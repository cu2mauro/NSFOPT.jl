"""
    PTeosMHzGWs

Computes frequency and amplitude of MHz gravitational waves from a first-order
quark-hadron phase transition in accreting neutron stars.

The pipeline, per accepted EOS sample `i`:

 1. [`prepare_eos`](@ref) builds, from the hadronic (`TOV`) and quark-branch
    (`TOVext`) mass sequences, the overpressure `Δp` driving nucleation
    and the quark-core radius `R_c(M)`.
 2. [`compute_row`](@ref) integrates the bubble-nucleation associated ODE for one point of
    the (accretion rate, surface tension, Λ, wall velocity) grid and returns the
    peak frequency, bubble radius, core radius and bubble count.
 3. [`process_eos`](@ref) sweeps the grid, then converts each surviving row
    into a strain spectrum, an SNR against a detector noise curve, and a
    characteristic-strain point, to be used for plotting.

Units are as follow: masses in M_☉, times in ms, pressures in GeV⁴
(converted from MeV/fm³ in the HDF5), lengths in m, frequencies in Hz
unless a name says otherwise.
"""
module PTeosMHzGWs

using DelimitedFiles: readdlm
using HDF5: h5open, read
using JSON: parsefile
using OrdinaryDiffEq: ODEProblem, ContinuousCallback, solve, Vern7, AutoVern7, Rodas5P #maybe consider Vern9?
using SciMLBase: terminate!, ReturnCode
using Roots: find_zero, Brent

export MonotoneCubic, Linear1D, domain, interp_loglog,
       Branch, load_branch, PTparams, load_PTparams, accepted_ids,
       M_fallback, t_crit_of_M,
       PreparedEoS, prepare_eos,
       Row, compute_row, safe_row, sweep,
       NoiseCurve, load_noise_curve, Pgw, trapz, geometric_prefactor,
       EOSResult, process_eos, process_eos_pt, envelope, PTtools

# ---------------------------------------------------------------------------
# Physical constants and pipeline settings
# ---------------------------------------------------------------------------

const GEV_TO_INVFM   = 5.067730756672362
const GEV_TO_INVMS   = 1.5192674479961276e21
const MEV_FM3_TO_SI  = 1.602176634e32     # 1 MeV/fm³ in J/m³
const G_NEWTON       = 6.6743e-11         # SI
const C_LIGHT        = 299792458.0        # SI

# Rest-mass density ρ = m_B n_B, in the convention of arXiv:2304.12316: that
# paper's AGILE-BOLTZTRAN runs assume m_B = 938 MeV (not the sampler's own
# 931.494 MeV), and its fits Eq. (3)/(5) take ρ in 10¹⁴ g cm⁻³, so n_B [fm⁻³]
# has to be turned into a *mass* density — hence the factor 1/c².
const M_BARYON_MEV     = 938.0
"1 MeV/fm³ as a mass density in 10¹⁴ g cm⁻³ (J/m³ → kg/m³ → g/cm³ → 10¹⁴)."
const MEV_FM3_TO_RHO14 = MEV_FM3_TO_SI / C_LIGHT^2 / 1e17
"n_B [fm⁻³] → ρ [10¹⁴ g cm⁻³]; ≈ 16.72, so n_sat = 0.16 fm⁻³ ↦ 2.7."
const N_FM3_TO_RHO14   = M_BARYON_MEV * MEV_FM3_TO_RHO14
const D_SOURCE       = 10 * 3.0857e19     # source distance, m

"""
    M_fallback(t) -> M⊙

Gravitational mass at `t` ms after core collapse under fallback accretion,
`M(t) = 1.2 + 0.63 (t/200)^0.3 - 20/(t + 200)`.

`t = 0` is core collapse, so `M_fallback(0)` is the birth mass — never write
that number out, it moves whenever this law does. Strictly increasing, hence
invertible by [`t_crit_of_M`](@ref). `Ṁ` diverges as `t^-0.7` at the origin,
but `M` itself is finite there, so only the inverse is ever needed.

This law governs core collapse → criticality only. Past criticality the star
switches to `M_crit + (t - t_crit) a / 1000` at the scanned rate `a`.
"""
M_fallback(t::Real) = 1.2 + 0.63 * (t / 200)^0.3 - 20 / (t + 200)

"Largest critical mass considered, M⊙. Nothing above this is a neutron star."
const M_MAX = 10.0

"Inversion bracket for [`t_crit_of_M`](@ref), ms; the assert keeps it valid if the law changes."
const T_MAX = 1.4e6
@assert M_fallback(T_MAX) ≥ M_MAX "T_MAX no longer brackets M_MAX for this M_fallback"

"""
    t_crit_of_M(M_crit) -> ms

Time from core collapse to criticality: the inverse of [`M_fallback`](@ref).

`M_crit` must lie in `[M_fallback(0), M_MAX]`; [`prepare_eos`](@ref) drops the
samples that do not (7.1% of the current 6516, already supercritical at birth).
"""
t_crit_of_M(M_crit::Real) =
    find_zero(t -> M_fallback(t) - M_crit, (0.0, T_MAX), Brent())

const RHO_KIN  = 3.2e33                   # J/m³
const OMEGA_GW = 0.012

# guesses for the spinning quadrupole emission
const ETA_F = 1.0
const SPIN = 0.2

"Upper frequency limit of the plotted/integrated band, Hz."
const F_HI = 5.0e7
"Points per plotted strain curve."
const N_F_CURVE = 60
"Points per SNR integral."
const N_F_SNR = 200

"`Chop` threshold: default for zeroing small reals."
const CHOP_TOL = 1.0e-16

# ---------------------------------------------------------------------------
# Interpolation
#
# Both interpolants used are immutable and their call sites allocate nothing and
# mutate nothing, so a single instance can be shared across threads.
# ---------------------------------------------------------------------------

"""
    MonotoneCubic(x, y)

Monotone cubic Hermite (PCHIP / Fritsch–Carlson) interpolant.

`x` must be strictly increasing. Evaluation outside `[x[1], x[end]]` continues
the end cubics.
"""
struct MonotoneCubic
    x::Vector{Float64}
    y::Vector{Float64}
    d::Vector{Float64}   # node slopes
end

function MonotoneCubic(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    n == length(y) || throw(ArgumentError("x and y must have equal length"))
    n ≥ 2 || throw(ArgumentError("need at least 2 nodes, got $n"))
    xs = collect(Float64, x)
    ys = collect(Float64, y)
    all(>(0), diff(xs)) || throw(ArgumentError("x must be strictly increasing"))

    h = diff(xs)
    δ = diff(ys) ./ h
    d = similar(ys)
    if n == 2
        d[1] = d[2] = δ[1]
    else
        for i in 2:n-1
            if δ[i-1] * δ[i] > 0
                w1 = 2h[i] + h[i-1]
                w2 = h[i] + 2h[i-1]
                d[i] = (w1 + w2) / (w1 / δ[i-1] + w2 / δ[i])
            else
                d[i] = 0.0
            end
        end
        d[1] = _end_slope(h[1], h[2], δ[1], δ[2])
        d[n] = _end_slope(h[n-1], h[n-2], δ[n-1], δ[n-2])
    end
    return MonotoneCubic(xs, ys, d)
end

# One-sided three-point slope, limited so the end interval stays monotone.
function _end_slope(h1, h2, δ1, δ2)
    d = ((2h1 + h2) * δ1 - h1 * δ2) / (h1 + h2)
    if sign(d) != sign(δ1)
        return 0.0
    elseif sign(δ1) != sign(δ2) && abs(d) > abs(3δ1)
        return 3δ1
    end
    return d
end

function (itp::MonotoneCubic)(t::Real)
    x, y, d = itp.x, itp.y, itp.d
    i = clamp(searchsortedlast(x, t), 1, length(x) - 1)
    h = x[i+1] - x[i]
    s = (t - x[i]) / h
    s2 = s * s
    s3 = s2 * s
    h00 =  2s3 - 3s2 + 1
    h10 =   s3 - 2s2 + s
    h01 = -2s3 + 3s2
    h11 =   s3 -  s2
    return h00 * y[i] + h10 * h * d[i] + h01 * y[i+1] + h11 * h * d[i+1]
end

"""
    Linear1D(x, y)

Piecewise-linear interpolant.
"""
struct Linear1D
    x::Vector{Float64}
    y::Vector{Float64}
end

function Linear1D(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(ArgumentError("x and y must have equal length"))
    length(x) ≥ 2 || throw(ArgumentError("need at least 2 nodes"))
    xs = collect(Float64, x)
    ys = collect(Float64, y)
    all(>(0), diff(xs)) || throw(ArgumentError("x must be strictly increasing"))
    return Linear1D(xs, ys)
end

function (itp::Linear1D)(t::Real)
    x, y = itp.x, itp.y
    i = clamp(searchsortedlast(x, t), 1, length(x) - 1)
    return y[i] + (y[i+1] - y[i]) * (t - x[i]) / (x[i+1] - x[i])
end

"""
    domain(itp) -> (lo, hi)

Node range of an interpolant.
"""
domain(itp::MonotoneCubic) = (first(itp.x), last(itp.x))
domain(itp::Linear1D) = (first(itp.x), last(itp.x))

"""
    interp_loglog(x, y, t)

Interpolate a tabulated positive curve in log-log space, clamping to the end
values outside `x`. Used for the sound shell spectrum, which spans decades in
both `z` and amplitude; returns `0` across any non-positive sample.
"""
function interp_loglog(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, t::Real)
    t ≤ first(x) && return float(first(y))
    t ≥ last(x) && return float(last(y))
    i = searchsortedlast(x, t)
    (y[i] > 0 && y[i+1] > 0) || return 0.0
    w = (log(t) - log(x[i])) / (log(x[i+1]) - log(x[i]))
    return exp((1 - w) * log(y[i]) + w * log(y[i+1]))
end

# ---------------------------------------------------------------------------
# EOS input
# ---------------------------------------------------------------------------

"""
    Branch

The mass sequences of one EOS sample: `M`/`pc`/`Rqm`/`stab` on the hadronic
(`TOV`) branch, the `x`-suffixed fields on the quark (`TOVext`) branch, and
`like`, the sample's total likelihood (`/params/ptot`).
"""
struct Branch
    id::Int
    M::Vector{Float64}
    pc::Vector{Float64}
    Rqm::Vector{Float64}
    stab::Vector{Float64}
    Λtd::Vector{Float64}
    Mx::Vector{Float64}
    pcx::Vector{Float64}
    stabx::Vector{Float64}
    Λtdx::Vector{Float64}
    nx::Vector{Float64}
    like::Float64
end

"""
    load_branch(file, id) -> Branch

Read sample `id` from an open HDF5 handle or a path.
"""
function load_branch(f, id::Integer)
    g = string(id)
    return Branch(
        id,
        read(f, "$g/TOV/M"),
        read(f, "$g/TOV/p_cent"),
        read(f, "$g/TOV/Rqm"),
        read(f, "$g/TOV/stab"),
        read(f, "$g/TOV/Lambda"),
        read(f, "$g/TOVext/M"),
        read(f, "$g/TOVext/p_cent"),
        read(f, "$g/TOVext/stab"),
        read(f, "$g/TOVext/Lambda"),
        read(f, "$g/TOVext/n_cent"),
        first(read(f, "$g/params/ptot")),
    )
end

load_branch(path::AbstractString, id::Integer) = h5open(f -> load_branch(f, id), path, "r")



struct PTparams
    id::Int
    cs2h::Float64     # hadron side of the transition = cs2PTl (lower density)
    cs2q::Float64     # quark  side of the transition = cs2PTr (higher density)
    pPT::Float64
    ePTh::Float64
    ePTq::Float64
end

function load_PTparams(f, id::Integer)
    g = string(id)

    ee = read(f, "$g/EOS/e")
    eh = first(read(f, "$g/params/ePTl"))
    eq = ee[findfirst(x->x==eh, ee)+1]

    return PTparams(
        id,
        first(read(f, "$g/params/cs2PTl")),
        first(read(f, "$g/params/cs2PTr")),
        first(read(f, "$g/params/pPT")),
        eh,
        eq
    )
end

load_PTparams(path::AbstractString, id::Integer) = h5open(f -> load_PTparams(f, id), path, "r")

"""
    PTtools(PTparams, vw) -> (K, Pgw, z)

Kinetic energy fraction `K`, the SSM spectral shape `Pgw`, and the dimensionless
wavenumbers `z` it is tabulated on, for one EOS and one wall velocity `vw`.

`z = k r_*`, so a physical frequency is `f = z c / (2π r_*)`; the strain uses
`(z^3/2π²) Pgw(z)` in place of the broken power law, and `ρ_kin = K e_h`.
"""
function PTtools(PT::PTparams, vw)
    alpha = (PT.ePTq - PT.ePTh) / (3 * (PT.ePTh + PT.pPT))
    # css2 is the phase ahead of the wall (hadron), csb2 the one behind (quark).
    return Base.invokelatest(_ssm_solver(), PT.cs2h, PT.cs2q, abs(alpha), Float64(vw))
end

# PythonCall is loaded on first use rather than at module load: importing it runs
# CondaPkg's resolver, which is slow and prints a dozen lines to stderr, and only
# `PTtools` needs it. The two `@eval`s must stay separate — the second is
# macro-expanded only once the first has brought `@pyexec` into scope.
const _SSM = Ref{Any}()

function _ssm_solver()
    if !isassigned(_SSM)
        @eval using PythonCall
        @eval _SSM[] = function (css2, csb2, alpha, vw)
            @pyexec (css2, csb2, alpha, vw) => """
                import logging
                import warnings
                logging.disable(logging.WARNING)
                warnings.filterwarnings("ignore")
                from pttools.bubble import Bubble
                from pttools.models import ConstCSModel
                from pttools.ssm import SSMSpectrum
                model = ConstCSModel(css2=css2, csb2=csb2, alpha_n_min=0.5*alpha, log_info=False)
                bubble = Bubble(model, v_wall=vw, alpha_n=alpha, theta_bar=True, log_invalid=False)
                bubble.solve()
                if bubble.no_solution_found or bubble.solver_failed or bubble.numerical_error:
                    raise RuntimeError(f"no reliable solution at v_wall={vw}, alpha={alpha}")
                K = float(bubble.kinetic_energy_fraction)
                spec = SSMSpectrum(bubble)
                Pgw = spec.spec_den_gw / (3*K**2*spec.source_lifetime_factor)
                z = spec.y
                """ => (K::Float64, Pgw::Vector{Float64}, z::Vector{Float64})
        end
    end
    return _SSM[]
end

"""
    accepted_ids(json_path, h5_name) -> Vector{Int}

Accepted sample ids for one HDF5 file, from the `accepted.json` written by
`FilterEOS.jl`. `h5_name` is the file's basename, e.g. `"0.h5"`.
"""
function accepted_ids(json_path::AbstractString, h5_name::AbstractString)
    acc = parsefile(json_path)
    haskey(acc, h5_name) || throw(KeyError("$h5_name not in $json_path (have: $(collect(keys(acc))))"))
    return Vector{Int}(acc[h5_name])
end

# ---------------------------------------------------------------------------
# Stage 1: Δp(t) and R_c(M)
# ---------------------------------------------------------------------------

"""
    PreparedEoS

Everything stage 2 needs from one EOS sample:

- `dp`: overpressure Δp (GeV⁴) against `1000 (M - M_crit)`, the mass accreted
  past criticality in units of 10⁻³ M⊙. At rate `a` that argument is `a (t -
  t_crit)` for `t` in ms, which is how [`compute_row`](@ref) drives it.
- `M_crit`: gravitational mass at which the quark branch takes over, M⊙. Read
  straight off the mass sequence — the EOS alone, no fallback input.
- `t_crit`: time from core collapse to `M_crit` under [`M_fallback`](@ref), ms.
  Fixed by the EOS, not by the sweep grid, hence computed once here.
- `Rc`: quark-core radius (m) as a function of gravitational mass (M⊙).
"""
struct PreparedEoS
    dp::MonotoneCubic
    M_crit::Float64
    t_crit::Float64
    Rc::MonotoneCubic
    Λq::MonotoneCubic
    Λh::MonotoneCubic
    rhoh::MonotoneCubic
end

"""
    prepare_eos(br::Branch) -> Union{PreparedEoS, Nothing}

Build the Δp and R_c interpolants for one sample, or `nothing` where there
are no stable models, a stable set that is not a contiguous prefix of
the sequence, no quark-over-hadron crossing, or a crossing outside
`[M_fallback(0), M_MAX]`.
"""
function prepare_eos(br::Branch)
    stab_idx  = findall(==(1.0), br.stab)
    stab_idxx = findall(==(1.0), br.stabx)
    (isempty(stab_idx) || isempty(stab_idxx)) && return nothing
    # The stable models must be a leading run; anything else means the sequence
    # is not a single rising branch and the Δp construction below is meaningless.
    stab_idx  == 1:length(stab_idx)   || return nothing
    stab_idxx == 1:length(stab_idxx)  || return nothing

    Mi   = br.M[stab_idx]
    pci  = br.pc[stab_idx]  ./ GEV_TO_INVFM^3 ./ 1000
    Mxi  = br.Mx[stab_idxx]
    pcxi = br.pcx[stab_idxx] ./ GEV_TO_INVFM^3 ./ 1000
    Rqmi = br.Rqm[stab_idx] .* 1000
    Lambda = br.Λtd[stab_idx]
    Lamdbax = br.Λtdx[stab_idxx]
    rhox = br.nx[stab_idxx] .* N_FM3_TO_RHO14

    K = min(length(Mi), length(Mxi))
    K ≥ 2 || return nothing

    MQ = Mi[1:K]
    MH = Mxi[1:K]

    pQ = Linear1D(_sorted_unique(MQ, pci[1:K])...)
    pH = Linear1D(_sorted_unique(MH, pcxi[1:K])...)

    pos = findfirst(k -> Mxi[k] - Mi[k] > 0, 1:K)
    (pos === nothing || pos ≤ 1) && return nothing
    pos -= 1

    # Criticality, read straight off the mass sequence. A star already
    # supercritical at birth never accretes up to it, so there is nothing to time.
    M_crit = Mi[pos]
    (M_fallback(0.0) ≤ M_crit ≤ M_MAX) || return nothing
    t_crit = t_crit_of_M(M_crit)

    Mlo  = max(first(MQ), first(MH))
    Mhi  = min(last(MQ), last(MH))
    Mgrid = filter(M -> Mlo ≤ M ≤ Mhi, sort!(union(MQ, MH)))
    length(Mgrid) ≥ 2 || return nothing

    dx, dy = _sorted_unique(1000 .* (Mgrid .- M_crit), [_chop(pQ(M) - pH(M)) for M in Mgrid])
    length(dx) ≥ 2 || return nothing
    dp = MonotoneCubic(dx, dy)

    # R_c(M) over the full stable hadronic branch.
    mx, my = _sorted_unique(Mi, Rqmi)
    length(mx) ≥ 2 || return nothing
    Rc = MonotoneCubic(mx, my)

    # Λq(M) over the quark branch.
    mx, my = _sorted_unique(Mi, Lambda)
    length(mx) ≥ 2 || return nothing
    Λq = MonotoneCubic(mx, my)

    # Λh(M) over the hadronic branch.
    mx, my = _sorted_unique(Mxi, Lamdbax)
    length(mx) ≥ 2 || return nothing
    Λh = MonotoneCubic(mx, my)

    # nh(M) over the hadronic branch.
    mx, my = _sorted_unique(Mxi, rhox)
    length(mx) ≥ 2 || return nothing
    rhoh = MonotoneCubic(mx, my)

    return PreparedEoS(dp, M_crit, t_crit, Rc, Λq, Λh, rhoh)
end

_chop(v::Float64) = abs(v) < CHOP_TOL ? 0.0 : v

"""
    _sorted_unique(x, y) -> (xs, ys)

Sort the pairs by `x` and drop later duplicates.

If a sample flagged stable sits just past the maximum-mass turning point and its mass has
already dipped by ~1e-4 M⊙, `M` is not quite increasing. Sorting keeps the
sample instead of discarding it over a rounding-scale non-monotonicity.
"""
function _sorted_unique(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    p = sortperm(x)
    xs = collect(Float64, @view x[p])
    ys = collect(Float64, @view y[p])
    keep = Int[]
    for i in eachindex(xs)
        (i == 1 || xs[i] != xs[i-1]) && push!(keep, i)
    end
    return xs[keep], ys[keep]
end

# ---------------------------------------------------------------------------
# Stage 2: bubble nucleation
# ---------------------------------------------------------------------------

"""
    Row

One point of the parameter sweep and its outcome.

`N_bubbles == 0` with a zero `fpeak_MHz` means Δp never reached the nucleation
threshold within the branch; `N_bubbles == -1` flags a numerical failure. Both
are excluded by the `N_bubbles ≥ 2` cut applied downstream.

All times are in ms. `t_nuc_ms` is measured from core collapse, like every
other time in the module:
`t_crit` under [`M_fallback`](@ref), then the linear stage at the scanned rate
`Ṁ = a`. Always ≥ `t_crit` > 0 — samples already supercritical at birth are
dropped by [`prepare_eos`](@ref), never reported with a negative clock. The
delay from criticality alone is `t_nuc_ms - prep.t_crit`.
"""
struct Row
    accretion::Float64      # Ṁ, M⊙/s, linear stage past criticality
    sigma_MeV_fm2::Float64  # surface tension
    Lambda_MeV::Float64     # nucleation energy scale
    v_wall::Float64         # wall velocity / c
    fpeak_MHz::Float64
    R_bubble_m::Float64     # c / f_peak
    R_core_m::Float64       # quark core radius at transition
    N_bubbles::Float64
    M_nuc::Float64          # gravitational mass at nucleation, M⊙
    Λq_nuc::Float64
    Λh_nuc::Float64
    rhoh_nuc::Float64
    t_nuc_ms::Float64       # core collapse -> nucleation
end

_failed(a, s, l, v) = Row(a, s, l, v, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_never(a, s, l, v)  = Row(a, s, l, v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

"""
    compute_row(prep, accretion, sigma_MeV_fm2, Lambda_MeV, v_wall) -> Row

Integrate the nucleation ODE for one grid point.

The state is the chain `M₀' = f`, `K' = M₀`, `J' = 2K`, `V' = 3J`,
`n' = Λ⁴ f exp(-pref·V)`, with `f(t) = exp(-13.5 π² σ⁴ / Δp(a t)³)` the
Boltzmann-suppressed nucleation rate and `V` the exclusion volume. Integration
stops once `pref·V > 50`, i.e. once the transition has saturated.
"""
function compute_row(prep::PreparedEoS, accretion::Real, sigma_MeV_fm2::Real,
                     Lambda_MeV::Real, v_wall::Real)
    a  = Float64(accretion)
    sg = Float64(sigma_MeV_fm2)
    lm = Float64(Lambda_MeV)
    v  = Float64(v_wall)

    dp, Rc, Λq, Λh, rhoh = prep.dp, prep.Rc, prep.Λq, prep.Λh, prep.rhoh
    M_crit, t_crit = prep.M_crit, prep.t_crit

    σ = sg / GEV_TO_INVFM^2 / 1000
    Λ = lm / 1000 * GEV_TO_INVMS

    # Δp is keyed on mass accreted past criticality, so the ODE runs in the
    # shifted variable τ = t - t_crit (ms since criticality) and `dp(a τ)` is
    # the overpressure there. τ is local to this function; the returned time is
    # shifted back to core collapse.
    τmax = last(domain(dp)) / a

    S_target  = 4log(Λ) + 3log(v) + log(π / 3)
    dp_target = cbrt(13.5π^2 * σ^4 / S_target)

    # Δp peaks at the end of the branch; if even that is below threshold, no
    # bubble ever nucleates.
    dp(a * τmax) < dp_target && return _never(a, sg, lm, v)

    τlo = 1.0e-8 * τmax
    froot(u) = dp(a * u) - dp_target
    froot(τlo) ≥ 0 && return _failed(a, sg, lm, v)

    τ_nuc = try
        find_zero(froot, (τlo, τmax), Brent())
    catch
        return _failed(a, sg, lm, v)
    end

    τstart = max(τ_nuc - 1, τ_nuc / 2)
    pref = (4π / 3) * v^3 * Λ^4
    Λ⁴ = Λ^4
    A = 13.5π^2 * σ^4

    # Δp ≤ 0 is pre-transition: no nucleation. (Δp → 0⁺ underflows to 0 anyway.)
    @inline function rate(τ)
        Δ = dp(a * τ)
        return Δ > 0 ? exp(-A / Δ^3) : 0.0
    end

    function rhs!(du, u, _p, τ)
        f = rate(τ)
        du[1] = f
        du[2] = u[1]
        du[3] = 2u[2]
        du[4] = 3u[3]
        du[5] = Λ⁴ * f * exp(-pref * u[4])
        return nothing
    end

    prob = ODEProblem(rhs!, zeros(5), (τstart, τmax))
    saturated = ContinuousCallback((u, _τ, _i) -> pref * u[4] - 50, terminate!)
    sol = _solve_nucleation(prob, saturated)
    sol === nothing && return _failed(a, sg, lm, v)

    n_final = sol.u[end][5]
    (isfinite(n_final) && n_final > 0) || return _failed(a, sg, lm, v)

    # n has units of ms⁻³, so its cube root is a frequency in kHz.
    fpeak_MHz = cbrt(n_final) / 1000
    R_bubble  = C_LIGHT * 1e-6 / fpeak_MHz

    # Back to time since core collapse, and the mass the linear stage reached.
    t_nuc = t_crit + τ_nuc
    M_nuc = M_crit + τ_nuc * a / 1000

    R_core   = M_nuc < last(domain(Rc))   ? Rc(M_nuc)   : 0.0
    Λq_nuc   = M_nuc < last(domain(Λq))   ? Λq(M_nuc)   : 0.0
    Λh_nuc   = M_nuc < last(domain(Λh))   ? Λh(M_nuc)   : 0.0
    rhoh_nuc = M_nuc < last(domain(rhoh)) ? rhoh(M_nuc) : 0.0

    return Row(a, sg, lm, v, fpeak_MHz, R_bubble, R_core, (R_core / R_bubble)^3,
               M_nuc, Λq_nuc, Λh_nuc, rhoh_nuc, t_nuc)
end

_ok(sol) = sol.retcode in (ReturnCode.Success, ReturnCode.Terminated)

"""
    _solve_nucleation(prob, cb) -> solution or `nothing`

Integrate with an explicit high-order Runge–Kutta method, falling back to a
stiffness-switching method only if that fails outright.
"""
function _solve_nucleation(prob, cb)
    sol = try
        solve(prob, Vern7(); callback = cb, abstol = 1e-8, reltol = 1e-8,
              save_everystep = false, save_start = false)
    catch
        nothing
    end
    (sol !== nothing && _ok(sol)) && return sol

    sol = try
        solve(prob, AutoVern7(Rodas5P()); callback = cb, abstol = 1e-8, reltol = 1e-8,
              save_everystep = false, save_start = false)
    catch
        return nothing
    end
    return _ok(sol) ? sol : nothing
end

"""
    safe_row(prep, args...) -> Row

[`compute_row`](@ref) with any thrown error turned into a `N_bubbles == -1`
sentinel row.
"""
function safe_row(prep::PreparedEoS, a, s, l, v)
    try
        return compute_row(prep, a, s, l, v)
    catch
        return _failed(Float64(a), Float64(s), Float64(l), Float64(v))
    end
end

"""
    sweep_old(prep, accretions, sigmas, lambdas, vs) -> Vector{Row}

Evaluate [`safe_row`](@ref) over the full parameter product, one thread per
chunk of grid points. Each ODE solve is independent and the interpolants are
read-only, so this needs no locking.
"""
function sweep_old(prep::PreparedEoS, accretions, sigmas, lambdas, vs)
    grid = collect(Iterators.product(accretions, sigmas, lambdas, vs))
    rows = Vector{Row}(undef, length(grid))
    Threads.@threads for k in eachindex(grid)
        a, s, l, v = grid[k]
        rows[k] = safe_row(prep, a, s, l, v)
    end
    return rows
end

"""
    sweep(prep, accretions, sigmas, lambdas, v_ladder; n_min = 2.0) -> Vector{Row}

Evaluate [`safe_row`](@ref) over the (Ṁ, σ, Λ) product, and at each of those
points climb `v_ladder` from the slowest wall upwards, stopping at the first
velocity that yields fewer than `n_min` bubbles.

A faster wall lets each bubble swallow more volume before the next nucleates, so
`N_bubbles` falls as `v_w` rises and each EOS has a largest velocity it can
support. Measured over 480 (Ṁ, σ) ladders, `N_bubbles` fell monotonically in
`v_w` in every one, so "climb until it drops below two" is unambiguous.

`v_ladder` must be ascending. Threads run over the (Ṁ, σ, Λ) product with the
climb serial inside, so each thread owns one ladder.
"""
function sweep(prep::PreparedEoS, accretions, sigmas, lambdas, v_ladder;
               n_min::Real = 2.0)
    vs = sort(collect(Float64, v_ladder))
    grid = collect(Iterators.product(accretions, sigmas, lambdas))
    out = Vector{Vector{Row}}(undef, length(grid))
    Threads.@threads for k in eachindex(grid)
        a, s, l = grid[k]
        rows = Row[]
        for v in vs
            r = safe_row(prep, a, s, l, v)
            r.N_bubbles < n_min && break
            push!(rows, r)
        end
        out[k] = rows
    end
    return reduce(vcat, out; init = Row[])
end

# ---------------------------------------------------------------------------
# Stage 2.1: quadrupole oscillations
# ---------------------------------------------------------------------------

function f_quad(M::Float64, Λq::Float64)

    ai = [0.1817, -0.006652, -0.004105, 0.0004072, 1.712e-05, -4.796e-06, 2.838e-07, -5.743e-09]

    f_f = C_LIGHT^3 / (2π * G_NEWTON * M * 1.98847e30) * sum([ai[i]*log(Λq)^(i-1) for i = eachindex(ai)])

    return f_f::Float64
end

function tau_quad(M::Float64, Λq::Float64)

    bi = [4.514e-05, 1.907e-05, 4.3e-06, -5.025e-06, 1.133e-06, -1.165e-07, 5.851e-09, -1.167e-10]

    tau_f = 1 / (C_LIGHT^3 / (G_NEWTON * M * 1.98847e30)) / sum([bi[i]*log(Λq)^(i-1) for i = eachindex(bi)])

    return tau_f::Float64
end

function Love_Q(Λ::Float64)

    ci = [0.194, 0.0936, 0.0474, -4.21e-3, 1.23e-4]

    Qbar = exp(sum([ci[i]*log(Λ)^(i-1) for i = eachindex(ci)]))

    return Qbar::Float64
end

function hcf_quad(M::Float64, Λq::Float64, Λh::Float64)

    f = f_quad(M, Λq)
    tau = tau_quad(M, Λq)
    χ = SPIN # * C_LIGHT / (G_NEWTON * M * 1.98847e30) since SPIN is already dimensionless

    h0 = ETA_F * 4*pi*f^2 / C_LIGHT^2 / D_SOURCE * (G_NEWTON * M * 1.98847e30 / C_LIGHT^2)^3 * χ^2 * (Love_Q(Λh) - Love_Q(Λq))

    return (h0*sqrt(tau/2))::Float64
end

# ---------------------------------------------------------------------------
# Stage 2.2: second antineutrino burst
# ---------------------------------------------------------------------------

"""
    RHO_COLL_FIT_RANGE

Range of ρ_collapse spanned by the supernova models that Eqs. (3) and (5) of
arXiv:2304.12316 were fitted to, in 10¹⁴ g cm⁻³ (Table 4/5 of that paper give
4.2–6.3; the text quotes 4.2–6.2). Both relations are *empirical* linear fits
over that band and nothing else — Eq. (5) in particular has a negative slope,
so extrapolating past ρ = d₃ = 5.890 returns a negative luminosity.
"""
const RHO_COLL_FIT_RANGE = (2, 8)

"""
    tL_burst(ρ_coll) -> [t_burst, L_peak]

Second (electron antineutrino) burst of a PNS collapse triggered by the quark
matter phase transition, from the linear relations of arXiv:2304.12316:

    ρ_collapse ≃ c₁ t_burst      + d₁    (Eq. 3)
    ρ_collapse ≃ c₃ L_ν̄e,peak    + d₃    (Eq. 5)

inverted for the observables. `ρ_coll` is the *rest-mass* density ρ = m_B n_B
at the onset of collapse in **10¹⁴ g cm⁻³** (see [`N_FM3_TO_RHO14`](@ref)),
`t_burst` is the post-bounce time in s and `L_peak` the peak ν̄ₑ luminosity in
10⁵³ erg s⁻¹. Outside [`RHO_COLL_FIT_RANGE`](@ref) both are `NaN`: the fits
carry no information there, and this also swallows the `0.0` that
[`compute_row`](@ref) writes when the nucleation mass falls off the branch.
`L_peak` alone is `NaN` for ρ > d₃ = 5.890, where Eq. (5) has already gone
negative while Eq. (3) is still inside its fitted band.
"""
function tL_burst(ρ_coll::Float64)

    cdt = [1.304, 3.922]
    cdL = [-0.172, 5.890]

    lo, hi = RHO_COLL_FIT_RANGE
    (isfinite(ρ_coll) && lo ≤ ρ_coll ≤ hi) || return [NaN, NaN]

    t_b = (ρ_coll-cdt[2]) / cdt[1]
    L_peak = (ρ_coll-cdL[2]) / cdL[1]

    # The two fits are independent, so their implied ρ ranges do not quite
    # agree: Eq. (5) crosses zero at ρ = d₃ = 5.890, inside the band Eq. (3)
    # still covers. A negative peak luminosity is meaningless, so drop it and
    # keep the (much tighter) timing relation.
    L_peak > 0 || (L_peak = NaN)

    return [t_b::Float64, L_peak::Float64]
end

# ---------------------------------------------------------------------------
# Stage 3: strain spectra and SNR
# ---------------------------------------------------------------------------

"""
    Pgw(s)

Universal broken-power-law spectral shape at `s = f r_* / c`, peaking at `s = 1`.

This is the ansatz the pipeline uses by default. [`process_eos_pt`](@ref)
replaces it with the sound shell model, where the same role is played by
`(z³/2π²) P̃_gw(z)` with `z = 2π s` — the factor of 2π belongs there and only
there.
"""
const KR_PEAK = 5.0
Pgw(s) = (x = 2π * s / KR_PEAK; x^3 * (7 / (4 + 3x^2))^3.5)

"""
    trapz(x, y)

Trapezoidal integral of `y` over `x`.
"""
function trapz(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    s = 0.0
    @inbounds @simd for i in 1:length(x)-1
        s += (x[i+1] - x[i]) * (y[i+1] + y[i])
    end
    return 0.5s
end

"""
    NoiseCurve

Detector amplitude spectral density √Sₙ(f), log-log interpolated from a
two-column `f, √Sₙ` CSV. Callable: `nc(f)`.
"""
struct NoiseCurve
    logs::Linear1D
    fmin::Float64
    fmax::Float64
    f::Vector{Float64}
    asd::Vector{Float64}
end

"""
    load_noise_curve(path) -> NoiseCurve
"""
function load_noise_curve(path::AbstractString, delim::AbstractChar; columnf::Integer = 1, columnasd::Integer = 2)
    data = readdlm(path,delim)
    f = data[:, columnf]
    asd = data[:, columnasd]
    typeof(f[1]) <: Real ? nothing : (popfirst!(f); popfirst!(asd))
    p = sortperm(f)
    f, asd = f[p], asd[p]
    return NoiseCurve(Linear1D(log10.(f), log10.(asd)), first(f), last(f), f, asd)
end

function load_noise_curve(path::AbstractString; columnf::Integer = 1, columnasd::Integer = 2)
    data = readdlm(path)
    f = data[:, columnf]
    asd = data[:, columnasd]
    typeof(f[1]) <: Real ? nothing : (popfirst!(f); popfirst!(asd))
    p = sortperm(f)
    f, asd = f[p], asd[p]
    return NoiseCurve(Linear1D(log10.(f), log10.(asd)), first(f), last(f), f, asd)
end

(nc::NoiseCurve)(f::Real) = 10^nc.logs(log10(f))

"""
    EOSResult

Per-EOS output of [`process_eos`](@ref).

- `curves[k]`: `(f, √Sₕ)` spectrum of surviving row `k`, as an `n × 2` matrix.
- `peaks[k, :]`: `(f_peak, √Sₕ(f_peak))`.
- `characteristic[k, :]`: `(f_c, h_c / √f_c)` — the log-frequency centroid.
- `snr[k]`: matched-filter SNR against the noise curve.
- `like`: the sample's likelihood, used to colour every plot.
"""
struct EOSResult
    id::Int
    like::Float64
    rows::Vector{Row}
    curves::Vector{Matrix{Float64}}
    peaks::Matrix{Float64}
    characteristic::Matrix{Float64}
    quadrupole::Matrix{Float64}
    snr::Vector{Float64}
    neutrinos::Matrix{Float64}
end

"""
    geometric_prefactor(r) -> kpref

The part of `S_h` fixed by the geometry of the source, shared by every
spectral model: `R_c³/d² · 128π G²/c⁹ · r_*⁴`. What multiplies it is the
kinetic energy density squared times a dimensionless spectral shape.
"""
geometric_prefactor(r::Row) =
    r.R_core_m^3 / D_SOURCE^2 * 128π * G_NEWTON^2 / C_LIGHT^9 * r.R_bubble_m^4

"""
    _reduce(id, like, keep, noise, Sh; f_hi, n_curve, n_snr) -> EOSResult

Turn surviving rows into curves, peaks, characteristic strain and SNR.

`Sh(k, f)` gives the strain spectral density of row `keep[k]` at frequency `f`;
it is the only thing that differs between the spectral models, so both
[`process_eos`](@ref) and [`process_eos_pt`](@ref) reduce through here.
"""
function _reduce(id::Integer, like::Real, keep::Vector{Row}, noise::NoiseCurve,
                 Sh::Function; f_hi::Real, n_curve::Integer, n_snr::Integer)
    nk = length(keep)
    curves = Vector{Matrix{Float64}}(undef, nk)
    peaks  = Matrix{Float64}(undef, nk, 2)
    quadr  = Matrix{Float64}(undef, nk, 2)
    neutrinos = Matrix{Float64}(undef, nk, 2)
    charac = Matrix{Float64}(undef, nk, 2)
    snrs   = Vector{Float64}(undef, nk)

    for (k, r) in pairs(keep)
        # Below f_min = c / R_core the source is not coherent.
        fmin = C_LIGHT / r.R_core_m
        fg = exp.(range(log(fmin), log(f_hi); length = n_curve + 1))
        lnf = log.(fg)
        Shg = [Sh(k, f) for f in fg]
        curves[k] = hcat(fg, sqrt.(Shg))

        # ∫Sₕ df = ∫f Sₕ dln f, and the centroid of the same measure.
        hc2 = trapz(lnf, fg .* Shg)
        fc = exp(trapz(lnf, fg .* lnf .* Shg) / hc2)
        charac[k, 1] = fc
        charac[k, 2] = sqrt(hc2) / sqrt(fc)

        flo = max(fmin, noise.fmin)
        snrs[k] = if flo ≥ noise.fmax
            0.0
        else
            # Union the log grid with the tabulated noise nodes so the
            # interpolant's kinks land on grid points.
            fs = sort!(union(exp.(range(log(flo), log(noise.fmax); length = n_snr + 1)),
                             filter(f -> flo < f < noise.fmax, noise.f)))
            sqrt(trapz(log.(fs), [Sh(k, f) / noise(f)^2 for f in fs]))
        end

        fpk = r.fpeak_MHz * 1e6
        peaks[k, 1] = fpk
        peaks[k, 2] = sqrt(Sh(k, fpk))

        quadr[k, 1] = f_quad(r.M_nuc, r.Λq_nuc)
        quadr[k, 2] = hcf_quad(r.M_nuc, r.Λq_nuc, r.Λh_nuc)

        neutrinos[k, :] = tL_burst(r.rhoh_nuc)

    end

    return EOSResult(id, like, keep, curves, peaks, charac, quadr, snrs, neutrinos)
end

"Rows worth keeping: enough bubbles to collide, and a frequency we can plot."
_survivors(rows) = filter(r -> r.N_bubbles ≥ 2 && r.fpeak_MHz ≥ 0.0001, rows)

"""
    process_eos(br, prep, noise; accretions, sigmas, lambdas, vs,
                f_hi, n_curve, n_snr) -> Union{EOSResult, Nothing}

Sweep the parameter grid for one sample and reduce the surviving rows to strain
curves, peaks, characteristic strain and SNR. Returns `nothing` if no row
survives.

The spectral model is the broken power law [`Pgw`](@ref), with
`RHO_KIN` and `OMEGA_GW`; see [`process_eos_pt`](@ref) for the sound shell
model, where both are replaced by the bubble solution.
"""
function process_eos(br::Branch, prep::PreparedEoS, noise::NoiseCurve;
                     accretions, sigmas, lambdas, vs,
                     f_hi::Real = F_HI, n_curve::Integer = N_F_CURVE,
                     n_snr::Integer = N_F_SNR)
    keep = _survivors(sweep(prep, accretions, sigmas, lambdas, vs))
    isempty(keep) && return nothing

    kpref = [geometric_prefactor(r) * RHO_KIN^2 * OMEGA_GW for r in keep]
    Sh(k, f) = kpref[k] * Pgw(f * keep[k].R_bubble_m / C_LIGHT)

    return _reduce(br.id, br.like, keep, noise, Sh; f_hi, n_curve, n_snr)
end

"""
    process_eos_pt(br, prep, ptp, noise; accretions, sigmas, lambdas, vs,
                   f_hi, n_curve, n_snr) -> Union{EOSResult, Nothing}

[`process_eos`](@ref) with the sound shell model in place of the two guesses:

    ρ_kin: RHO_KIN            ->  K(v_w) · e_h,      from the bubble solution
    shape: OMEGA_GW · Pgw(s)  ->  (z³/2π²) P̃_gw(z),  z = 2π s = 2π f r_*/c

`OMEGA_GW` is absorbed into the SSM factor, so it must not be carried as well.
Everything upstream — nucleation, f_peak, r_*, R_core — is untouched.

One [`PTtools`](@ref) solve is done per wall velocity, since everything it needs
is fixed by the EoS. Rows whose wall velocity has no reliable bubble solution
are dropped, which is how very slow walls remove themselves.
"""
function process_eos_pt(br::Branch, prep::PreparedEoS, ptp::PTparams,
                        noise::NoiseCurve; accretions, sigmas, lambdas, vs,
                        f_hi::Real = F_HI, n_curve::Integer = N_F_CURVE,
                        n_snr::Integer = N_F_SNR)
    keep = _survivors(sweep(prep, accretions, sigmas, lambdas, vs))
    isempty(keep) && return nothing

    ssm = Dict{Float64,Tuple{Float64,Vector{Float64},Vector{Float64}}}()
    for v in unique(r.v_wall for r in keep)
        try
            K, pgw, z = PTtools(ptp, v)
            ssm[v] = (K * ptp.ePTh * MEV_FM3_TO_SI, z, pgw)
        catch
            @warn "no reliable PTtools solution, dropping these rows" v_wall = v
        end
    end
    keep = filter(r -> haskey(ssm, r.v_wall), keep)
    isempty(keep) && return nothing

    kpref = [geometric_prefactor(r) * ssm[r.v_wall][1]^2 for r in keep]
    function Sh(k, f)
        _, z_tab, pgw_tab = ssm[keep[k].v_wall]
        z = 2π * f * keep[k].R_bubble_m / C_LIGHT
        return kpref[k] * z^3 / (2π^2) * interp_loglog(z_tab, pgw_tab, z)
    end

    return _reduce(br.id, br.like, keep, noise, Sh; f_hi, n_curve, n_snr)
end

"""
    envelope(curves; npts = 200) -> (f, upper, lower)

Pointwise max/min band across a set of `(f, √Sₕ)` curves, computed in log-log
space on a shared grid.

The grid spans `min(first f)` to `min(last f)`: the left edge is the union of
the curves (a curve only contributes once it has started, so the band widens as
curves switch on), while the right edge is the intersection, so no curve is
extrapolated.
"""
function envelope(curves::Vector{Matrix{Float64}}; npts::Integer = 200)
    isempty(curves) && throw(ArgumentError("need at least one curve"))
    f_left = [c[1, 1] for c in curves]
    f_lo = minimum(f_left)
    f_hi = minimum(c[end, 1] for c in curves)
    grid = range(log(f_lo), log(f_hi); length = npts + 1)
    itps = [Linear1D(log.(c[:, 1]), log.(c[:, 2])) for c in curves]

    f = similar(collect(grid))
    upper = similar(f)
    lower = similar(f)
    for (j, lnf) in pairs(grid)
        fj = exp(lnf)
        hi = -Inf
        lo = Inf
        for (itp, fl) in zip(itps, f_left)
            fl ≤ fj || continue
            v = itp(lnf)
            hi = max(hi, v)
            lo = min(lo, v)
        end
        f[j] = fj
        upper[j] = exp(hi)
        lower[j] = exp(lo)
    end
    return f, upper, lower
end

end # module
