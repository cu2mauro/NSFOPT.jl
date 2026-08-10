# Selects, from a sampled ensemble of dense-matter EOS, those that can produce
# MHz gravitational waves through bubble nucleation in an accreting neutron star.
#
# ---------------------------------------------------------------------------
# The physical picture
# ---------------------------------------------------------------------------
#
# Each sample carries two stellar sequences, built from two versions of the same
# EOS that coincide below the transition pressure pPT and part company above it:
#
#   EOS   / TOV     the full EOS: hadronic, then a first-order transition at
#                   pPT, then quark matter. Its stars carry a quark core, whose
#                   radius is TOV/Rqm (zero below the transition).
#   EOSext/ TOVext  the purely hadronic branch continued past pPT. Its stars
#                   never convert, so TOVext/Rqm is identically zero. Above pPT
#                   this branch is metastable: it is the state the star is
#                   actually in until a bubble nucleates.
#
# Both TOV sequences are integrated on a shared central-pressure grid, so the
# same index in the two arrays is the same central pressure — which is why the
# mass comparisons below may be done element by element. (Central *density* is
# not shared: the two EOS give different n at the same p above the transition.)
#
# An accreting star climbs its hadronic sequence. Once the centre passes pPT the
# hadronic phase is overpressed with respect to quark matter, and quark bubbles
# can nucleate. The driver is the overpressure at fixed accreted mass,
#
#   Δp(t) = p_cent(TOV) − p_cent(TOVext),
#
# which enters the thin-wall O(4) bounce action S = 27π²σ⁴/(2Δp³): the
# nucleation rate goes as exp(−S), so it is negligible until Δp has grown enough
# and then switches on sharply. The transition completes in a shell of bubbles
# whose size sets the GW peak frequency, and the quark-core radius at nucleation
# sets how many bubbles fit — hence the interest in TOV/Rqm.
#
# ---------------------------------------------------------------------------
# What the four clauses require, and why
# ---------------------------------------------------------------------------
#
# 1. At least two stable hybrid configurations have a quark core (Rqm > 0).
#    A transition that produces no stable quark core produces no signal. Two is
#    also the minimum for interpolating R_core(M) later without extrapolating —
#    in practice samples clear this by a wide margin.
#
# 2. A stable hadronic star reaches a central pressure above pPT.
#    This is the "the hadronic branch extends far enough" condition: the
#    metastable branch must actually be pushed past coexistence by accretion,
#    otherwise there is no overpressure and nothing nucleates. Asking it of
#    TOVext/p_cent rather than of the EOSext table matters in principle — a
#    tabulated branch can extend past pPT with no star sitting there — though on
#    the present dataset the two agree on every sample.
#
# 3. The hadronic branch is nowhere less massive than the hybrid one at equal
#    central pressure. A first-order transition softens the EOS, so the star
#    that converts must be the lighter of the two. A violation means the
#    sequence is not a clean pair of branches and the Δp construction below is
#    meaningless.
#
# 4. The hadronic branch strictly exceeds the hybrid one at more than two
#    (stable) central pressures. Clause 3 admits branches that merely touch;
#    this demands a resolved separation, which is both the signature of a real
#    transition and enough points to interpolate Δp past its onset.
#
# Clauses 1 and 4 are the numerical ones: their thresholds exist so that nothing
# downstream has to extrapolate R_core(M) or Δp(t).
#
# Note that no clause tests for first-order-ness directly. In this sampler a
# transition, when present, is always first order; a sample with no transition
# carries pPT = 0 and is rejected by clause 2.

using HDF5, JSON

# Datasets a sample must have to be evaluated at all. All but `EOSext/p` are
# read by the clauses; that one is kept as a guard, because a sample without an
# `EOSext` group has no quark branch and is broken rather than merely unlucky.
#
# These do go missing in practice: in a 60k-sample run exactly one entry had no
# `EOSext`/`TOVext` group, alongside pPT = ptot = 0 — the sampler recorded no
# phase transition for it. Such an entry is skipped and counted rather than
# being allowed to abort the whole scan.
const REQUIRED = ("TOV/stab", "TOV/Rqm", "TOV/M",
                  "TOVext/stab", "TOVext/M", "TOVext/p_cent",
                  "EOSext/p", "params/pPT")

"Numeric top-level group ids present in `f`, sorted."
function sample_ids(f)
    ids = [parse(Int, k) for k in keys(f) if all(isdigit, k)]
    return sort!(ids)
end

"True when every dataset in `REQUIRED` exists under group `g`."
function complete_sample(f, g::AbstractString)
    haskey(f, g) || return false
    grp = f[g]
    for path in REQUIRED
        sub, dset = split(path, '/')
        haskey(grp, sub) || return false
        haskey(grp[sub], dset) || return false
    end
    return true
end

"""
    accepted_for(path; ids = nothing) -> (accepted, incomplete)

Ids passing all four acceptance clauses, plus the ids skipped for missing data.

`ids` defaults to whatever numeric groups the file actually holds, so the scan
adapts to the sample count instead of assuming a fixed range.
"""
function accepted_for(path; ids = nothing)
    acc = Int[]
    incomplete = Int[]
    h5open(path, "r") do f
        for i in (ids === nothing ? sample_ids(f) : ids)
            g = string(i)
            if !complete_sample(f, g)
                push!(incomplete, i)
                continue
            end

            st    = read(f, "$g/TOV/stab")
            stx   = read(f, "$g/TOVext/stab")
            rqm   = read(f, "$g/TOV/Rqm")
            pcx   = read(f, "$g/TOVext/p_cent")
            pcrit = read(f, "$g/params/pPT")
            mx    = read(f, "$g/TOVext/M")
            mm    = read(f, "$g/TOV/M")

            sel  = st  .== 1.0
            selx = stx .== 1.0
            (count(sel) ≥ 1 && count(selx) ≥ 1) || continue

            # Clauses 3 and 4 walk the two branches index by index, which is
            # only meaningful up to the last *stable* model on each. Bounding by
            # `length(mx)` instead let clause 4 run past the hadronic branch's
            # stable run — true for ~4% of accepted samples — and compare
            # against configurations beyond the first instability.
            k = min(count(sel), count(selx))
            Lmx = length(mx)

            ok = count(>(0), rqm[sel]) >= 2 &&          # clause 1
                 maximum(pcx[selx]) > pcrit[1] &&       # clause 2
                 count(<(0), mx .- mm[1:Lmx]) == 0 &&   # clause 3
                 count(>(0), mx[1:k] .- mm[1:k]) > 2    # clause 4
            ok && push!(acc, i)
        end
    end
    return acc, incomplete
end

dir   = joinpath(@__DIR__, "..", "EOSsamplerLocal", "build")
files = sort(filter(endswith(".h5"), readdir(dir; join = true)))

results = Dict{String,Vector{Int}}()
for f in files
    acc, incomplete = accepted_for(f)
    results[basename(f)] = acc
    @info "filtered" file = basename(f) accepted = length(acc) incomplete = length(incomplete)
    isempty(incomplete) ||
        @warn "skipped samples with missing datasets" file = basename(f) ids = incomplete
end

open(joinpath(dir, "accepted.json"), "w") do io
    JSON.print(io, results)
end
@info "wrote accepted.json" path = joinpath(dir, "accepted.json") total_accepted = sum(length, values(results))
