#!/usr/bin/env julia
#
# Driver for the PTeosMHzGWs module.
#
# Sweeping and plotting are separate steps: `run_sweep` does the expensive work
# once per HDF5 file and writes `results_<id>.h5`; the plot functions read those
# back, so any set of files can be plotted together without re-sweeping.
#
# Interactive use (preferred — pays the ~4 s package load and the JIT once):
#
#     julia -t auto
#     julia> include("run_PTeosMHzGWs.jl")
#     julia> run_sweep(0)          # writes results_0.h5
#     julia> run_sweep(1)          # writes results_1.h5
#     julia> plot_ids(0)           # figures for one file
#     julia> plot_ids(0:1)         # one figure per kind, both files combined
#     julia> plot_ids(:all)        # every results_*.h5 in the folder
#     julia> plot_bubbles(1, 101)  # bubble plot, 101st accepted sample of 1.h5
#
# `-t auto` can only be given at startup: the sweep is threaded and there is no
# way to add threads from inside a running session.
#
# Batch use:
#
#     julia --project=. -t auto run_PTeosMHzGWs.jl 0 1 2
#
# `include` cannot take arguments, so the file id is a function argument rather
# than a constant to edit.

using Pkg
Pkg.activate(@__DIR__; io = devnull)

using Revise

# Included once per session: a second `include` would build a *new* module, and
# every name reachable through both copies (`EOSResult`, `Row`, …) would then
# resolve ambiguously. `includet` tracks the file instead, so edited *functions*
# take effect at once. Module-level `const`s do not — no `__revise_mode__` setting
# changes that — so re-tune one in place with
#     Core.eval(PTeosMHzGWs, :(const OMEGA_GW = 0.02))
# or restart. `struct`s always need a restart. The constants in *this* file are
# picked up by re-including it.
isdefined(@__MODULE__, :PTeosMHzGWs) || includet(joinpath(@__DIR__, "PTeosMHzGWs.jl"))
using .PTeosMHzGWs

using HDF5
using Plots
using Plots.PlotMeasures: mm
using Printf
using Colors: red, green, blue
using LaTeXStrings

gr()
# `pyplot()` also works — see `_savefig`, which repairs a log-axis bug in that
# backend. Set ENV["MPLBACKEND"] = "Agg" before including, or PyPlot tries to
# start a Qt backend and fails.

# ---------------------------------------------------------------------------
# Defaults — every one is overridable per call, none needs editing to switch file
# ---------------------------------------------------------------------------

const DATA_DIR = joinpath(@__DIR__, "..", "EOSsampler", "build")
const OUT_DIR  = joinpath(@__DIR__, "results")

const ACCRETIONS = 0.3:0.2:0.9      # Ṁ, M⊙/s
const SIGMAS     = 10.0:10.0:50.0   # surface tension, MeV/fm²
const LAMBDAS    = [200.0]          # energy scale, MeV
const VS_OLD     = 0.01:0.02:0.07   # wall velocity / c
const VS         = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

# ColorDatas. Low likelihood → pale, high → dark, so the interesting EOS stand
# out against the bulk; `by_z_order` reads the direction off the colour itself.
const LIKE_CMAP   = cgrad(:GnBu, rev = false)
const BUBBLE_CMAP = cgrad(:coolwarm)

# Figure format: "pdf" for vector output (papers), "png" for a quick look.
const FIG_EXT = "pdf"

const PLOT_XLIMS = (5e2, 5e7)
const PLOT_YLIMS = (1e-29, 1e-21)

results_path(h5_id; outdir = OUT_DIR) = joinpath(outdir, "results_$h5_id.h5")

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

"""
    run_sweep(h5_id; kwargs...)

Sweep every accepted sample of `<h5_id>.h5` and write `results_<h5_id>.h5`.

The likelihood range is taken over *all* accepted samples (not only those that
survive) and stored in the output file, so plots stay comparable across files.

Keywords: `accretions`, `sigmas`, `lambdas`, `vs`, `datadir`, `outdir`,
`save = true`.
"""
function run_sweep(h5_id::Integer;
                   accretions = ACCRETIONS, sigmas = SIGMAS,
                   lambdas = LAMBDAS, vs = VS,
                   datadir = DATA_DIR, outdir = OUT_DIR, save::Bool = true)
    Threads.nthreads() == 1 &&
        @warn "running single-threaded; restart julia with `-t auto` for the parallel sweep"

    h5path = joinpath(datadir, "$h5_id.h5")
    isfile(h5path) || error("no such file: $h5path")
    accepted = accepted_ids(joinpath(datadir, "accepted.json"), "$h5_id.h5")

    noise = first(noise_curves())      # the SNR is quoted against MWB-DMR

    @info "sweeping" file = "$h5_id.h5" n_accepted = length(accepted) threads = Threads.nthreads()

    results = EOSResult[]
    likes = Float64[]
    prep_failed = 0
    no_rows = 0

    t = @elapsed h5open(h5path, "r") do f
        for (k, id) in pairs(accepted)
            br = load_branch(f, id)
            push!(likes, br.like)
            prep = prepare_eos(br)
            if prep === nothing
                prep_failed += 1
                continue
            end
            res = process_eos(br, prep, noise;
                              accretions, sigmas, lambdas, vs)
            if res === nothing
                no_rows += 1
                continue
            end
            push!(results, res)
            k % 50 == 0 && @info "  swept" k kept = length(results)
        end
    end

    like_range = (0.0, maximum(likes))
    @info "sweep done" file = "$h5_id.h5" seconds = round(t; digits = 1) eos_with_signal = length(results) prep_failed no_surviving_rows = no_rows

    if save && !isempty(results)
        path = results_path(h5_id; outdir)
        save_results(path, results, like_range)
        @info "wrote" file = basename(path) rows = sum(length(r.rows) for r in results)
    end
    return
end

# ---------------------------------------------------------------------------
# Results I/O
#
# The saved file holds everything the plots need, so re-plotting never re-runs
# an ODE. Curves are uniform in length (n_curve + 1 points), so they store as
# one 3D array rather than a ragged set.
# ---------------------------------------------------------------------------

"""
    save_results(path, results, like_range)

Write a `Vector{EOSResult}` to HDF5: one group per sample id, plus a top-level
`like_range` over all accepted samples of the source file.
"""
function save_results(path::AbstractString, results::Vector{EOSResult},
                      like_range::Tuple{Float64,Float64})
    h5open(path, "w") do f
        f["like_range"] = collect(like_range)
        for r in results
            g = "eos/$(r.id)"
            f["$g/like"] = r.like
            f["$g/peaks"] = r.peaks
            f["$g/characteristic"] = r.characteristic
            f["$g/quadrupole"] = r.quadrupole
            f["$g/snr"] = r.snr
            # 11 × n table, columns in `Row` field order.
            f["$g/rows"] = reduce(hcat, [[x.accretion, x.sigma_MeV_fm2, x.Lambda_MeV,
                                          x.v_wall, x.fpeak_MHz, x.R_bubble_m,
                                          x.R_core_m, x.N_bubbles, x.M_at_nucleation,
                                          x.Λq_at_nucleation, x.Λh_at_nucleation] for x in r.rows])
            npt = size(first(r.curves), 1)
            curves = Array{Float64,3}(undef, npt, 2, length(r.curves))
            for (k, c) in pairs(r.curves)
                curves[:, :, k] = c
            end
            f["$g/curves"] = curves
        end
    end
    return
end

"""
    load_results(ids; outdir = OUT_DIR) -> (results, like_hi, sources)

Read back one or more `results_<id>.h5`.

`ids` may be an integer, any iterable of integers, or `:all` to take every
`results_*.h5` in `outdir`. `like_hi` is the maximum over all files loaded, so
a combined plot shares one colour scale. `sources[k]` is the file id that
`results[k]` came from — sample ids repeat across files, so this is the only
thing that identifies a result once several files are concatenated.
"""
function load_results(ids; outdir = OUT_DIR)
    file_ids = _resolve_ids(ids, outdir)
    isempty(file_ids) && error("no results files found; run `run_sweep(id)` first")

    results = EOSResult[]
    sources = Int[]
    like_hi = 0.0

    for fid in file_ids
        path = results_path(fid; outdir)
        isfile(path) || error("missing $(basename(path)); run `run_sweep($fid)` first")
        h5open(path, "r") do f
            like_hi = max(like_hi, read(f, "like_range")[2])
            for key in keys(f["eos"])
                g = f["eos"][key]
                m = read(g["rows"])
                rows = [Row(m[1, j], m[2, j], m[3, j], m[4, j],
                            m[5, j], m[6, j], m[7, j], m[8, j], 
                            m[9, j], m[10, j],m[11, j]) for j in axes(m, 2)]
                cs = read(g["curves"])
                curves = [cs[:, :, k] for k in axes(cs, 3)]
                push!(results, EOSResult(parse(Int, key), read(g["like"]), rows, curves,
                                         read(g["peaks"]), read(g["characteristic"]),
                                         read(g["quadrupole"]), read(g["snr"])))
                push!(sources, fid)
            end
        end
    end
    @info "loaded" files = ids eos = length(results) like_hi
    return results, like_hi, sources
end

function _resolve_ids(ids, outdir)
    if ids === :all
        found = Int[]
        for name in readdir(outdir)
            m = match(r"^results_(\d+)\.h5$", name)
            m === nothing || push!(found, parse(Int, m[1]))
        end
        return sort(found)
    end
    return collect(Int, ids isa Integer ? (ids,) : ids)
end

# ---------------------------------------------------------------------------
# Backend compatibility
# ---------------------------------------------------------------------------

"""
    _savefig(p, path)

Save `p`, repairing log axes first when the pyplot backend is active.

Plots' pyplot backend gives a log axis matplotlib's `symlog` scale instead of
`log`. `symlog` is *linear* below its `linthresh` (order 1), so an axis whose
whole range sits under that threshold is drawn linearly. That is every strain
axis here — 1e-20 down to 1e-30 — which collapses onto the bottom frame, while
the frequency axis at 1e4..1e8 is above the threshold and looks correct. The
asymmetry is what makes the points look "off the plot" and the y ticks wrong.

GR sets a true log scale and is untouched by any of this.
"""
function _savefig(p, path)
    if Plots.backend_name() === :pyplot
        Plots.prepare_output(p)                       # realise the matplotlib figure
        repaired = false
        for sp in p.subplots
            ax = sp.o
            ax === nothing && continue
            for (axsym, getscale, setscale, setlim) in
                    ((:xaxis, ax.get_xscale, ax.set_xscale, ax.set_xlim),
                     (:yaxis, ax.get_yscale, ax.set_yscale, ax.set_ylim))
                sp[axsym][:scale] === :log10 || continue
                getscale() == "symlog" || continue
                setscale("log")
                lims = sp[axsym][:lims]               # re-apply: set_*scale resets them
                lims isa Tuple && all(x -> x isa Real, lims) && setlim(lims...)
                repaired = true
            end
        end
        # Write through matplotlib rather than `savefig(p, ...)`: the Plots entry
        # point re-renders the figure from the plot object, which would restore
        # symlog and undo the repair above.
        if repaired
            p.o.savefig(path; bbox_inches = "tight", dpi = 200)
            return path
        end
    end
    savefig(p, path)
    return path
end

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

like_color(like, like_hi) = LIKE_CMAP[clamp(like / like_hi, 0, 1)]

# Draws the palest colours first so the darkest (highest-likelihood) land on top.
brightness(c) = red(c) + green(c) + blue(c)
by_z_order(rs, like_hi) = sort(rs; by = r -> -brightness(like_color(r.like, like_hi)))

"The two detector noise curves drawn under every strain figure."
noise_curves() = (load_noise_curve(joinpath(@__DIR__, "ExperimentSignal", "MWB-DMR-res.csv"), ','),
                  load_noise_curve(joinpath(@__DIR__, "ExperimentSignal", "levitatedSensors_100m.csv"), ','),
                  load_noise_curve(joinpath(@__DIR__, "ExperimentSignal", "LIGO5.tad"),columnf=1,columnasd=4),
                  load_noise_curve(joinpath(@__DIR__, "ExperimentSignal", "ET-0001A-18_ETDSensitivityCurveTxtFile.txt")),
                  load_noise_curve(joinpath(@__DIR__, "ExperimentSignal", "cosmic_explorer_strain.txt")))

"Decade ticks across `lims`, one every `step` orders of magnitude."
_log_ticks(lims, step) =
    10.0 .^ (round(Int, log10(lims[1])):step:round(Int, log10(lims[2])))

# Likelihoods from the sampler are relative, so they are plotted normalised to
# the most likely EOS in the set: `like_color` divides by `like_hi`, which puts
# the scale on 0..1 with the maximum at 1, and the colourbar says so.
function strain_axes(ylabel, title)
    plot(; xscale = :log10, yscale = :log10,
         xlims = PLOT_XLIMS, ylims = PLOT_YLIMS,
         xticks = _log_ticks(PLOT_XLIMS, 1), yticks = _log_ticks(PLOT_YLIMS, 2),
         xlabel = L"$f\;\left[\mathrm{Hz}\right]$", ylabel = ylabel,
         framestyle = :box, size = (900, 650), legend = false,
         colorbar = true, clims = (0.0, 1.0),
         colorbar_title = L"$\mathrm{Normalized~likelihood}$",
         colorbar_titlefontsize = 14,
         right_margin = 6mm, left_margin = 4mm)
end

"""
    _frame_on_top!(p)

Re-draw the axis box as the last series.

Series are painted in the order they are added, so the envelope fills and the
denser curve bundles cover the frame drawn with the axes — the edges of the
plot end up washed with colour instead of a clean black rule. Tracing the box
again at the end puts it back on top.
"""
function _frame_on_top!(p)
    x1, x2 = PLOT_XLIMS
    y1, y2 = PLOT_YLIMS
    plot!(p, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1];
          seriestype = :path, linecolor = :black, linewidth = 1.0,
          label = "", primary = false)
    return p
end

function add_noise!(p, noises)
    for nc in noises
        plot!(p, nc.f, nc.asd; color = :gray, linestyle = :dash, label = "")
    end
    # Dummy series carrying the colour scale, since every real series is drawn in
    # one flat colour and so never defines a colourbar of its own.
    #
    # The points sit *inside* the plot range at zero size: in range so they
    # cannot stretch the axis, zero size so they never show. Do not move them
    # outside the limits to hide them — that expands the axis and squashes the
    # real data, whatever `ylims` says.
    scatter!(p, fill(PLOT_XLIMS[1], 2), fill(PLOT_YLIMS[1], 2);
             marker_z = [0.0, 1.0], clims = (0.0, 1.0),
             c = LIKE_CMAP, markersize = 0, markerstrokewidth = 0, label = "")
    return _frame_on_top!(p)
end

# ---------------------------------------------------------------------------
# Dataset layers
#
# Every strain dataset lives on the same log-log axes, so each reduces to a
# `draw!(p, r, col)` that adds one EOS's contribution to an existing plot. The
# standalone figures (`plot_strain`) and the combined overlay (`plot_overlay`)
# both go through these, so a style lives in one place — and this is the surface
# to port if the backend ever moves to CairoMakie.
# ---------------------------------------------------------------------------

const _SH_LABEL  = L"$\sqrt{S_h}\;\left[\mathrm{Hz}^{-1/2}\right]$"
const _HC_LABEL  = L"$h_c/\sqrt{f_c}\;\left[\mathrm{Hz}^{-1/2}\right]$"
const _GEN_LABEL = L"$\left[\mathrm{Hz}^{-1/2}\right]$"

_draw_peaks!(p, r, col; marker = :circle) =
    scatter!(p, r.peaks[:, 1], r.peaks[:, 2]; color = col, markershape = marker,
             markersize = 2.5, markerstrokewidth = 0, label = "")

_draw_characteristic!(p, r, col; marker = :circle) =
    scatter!(p, r.characteristic[:, 1], r.characteristic[:, 2]; color = col,
             markershape = marker, markersize = 2.5, markerstrokewidth = 0, label = "")

_draw_quadrupole!(p, r, col; marker = :circle) =
    scatter!(p, r.quadrupole[:, 1], r.quadrupole[:, 2]; color = col,
             markershape = marker, markersize = 2.5, markerstrokewidth = 0, label = "")

# One series per EOS, not per curve: an EOS's curves share a colour, so they
# string into a single polyline with NaN between them (a NaN lifts the pen).
# Plots' per-series overhead dominates at tens of thousands of curves, and the
# drawn result is identical.
function _draw_curves!(p, r, col; marker = :none)
    xs = Float64[]
    ys = Float64[]
    for c in r.curves
        append!(xs, @view c[:, 1]); push!(xs, NaN)
        append!(ys, @view c[:, 2]); push!(ys, NaN)
    end
    plot!(p, xs, ys; color = col, linewidth = 0.5, label = "")
end

function _draw_envelopes!(p, r, col; marker = :none)
    f, upper, lower = envelope(r.curves)
    plot!(p, f, upper; fillrange = lower, fillalpha = 0.7,
          fillcolor = col, linecolor = col, linewidth = 0.3, label = "")
end

# Ordered back-to-front: fills, then curve bundles, then point clouds, so a
# scatter dataset always lands on top of an envelope it shares a figure with.
# `kind` picks the y-axis label; `marker` tells the point datasets apart when
# several are overlaid (ignored by curves/envelopes).
const STRAIN_LAYERS = (
    envelopes      = (draw = _draw_envelopes!,      kind = :sh, marker = :none),
    curves         = (draw = _draw_curves!,         kind = :sh, marker = :none),
    characteristic = (draw = _draw_characteristic!, kind = :hc, marker = :diamond),
    quadrupole     = (draw = _draw_quadrupole!,     kind = :hc, marker = :utriangle),
    peaks          = (draw = _draw_peaks!,          kind = :sh, marker = :circle),
)

_layer_label(kind) = kind === :sh ? _SH_LABEL : kind === :hc ? _HC_LABEL : _GEN_LABEL

"""
    _parse_layers(datasets) -> Vector{Symbol}

Normalise a dataset selection — a string like `"peaks, envelopes"`, or an
iterable of strings/symbols — to validated layer names.
"""
function _parse_layers(datasets)
    raw = datasets isa AbstractString ? split(datasets, r"[\s,]+"; keepempty = false) :
          datasets isa Union{AbstractString,Symbol} ? [datasets] : collect(datasets)
    names = Symbol[Symbol(lowercase(string(x))) for x in raw]
    isempty(names) && error("no datasets given")
    valid = keys(STRAIN_LAYERS)
    for n in names
        n in valid || error("unknown dataset :$n; choose from $(join(valid, ", "))")
    end
    return names
end

# ---------------------------------------------------------------------------
# Strain plots
# ---------------------------------------------------------------------------

"""
    plot_strain(results, like_hi; figdir, noises) -> Dict{String,String}

Write the four strain figures (peaks, curves, characteristic, envelopes) for a
collection of results, however many source files it spans.
"""
function plot_strain(results::Vector{EOSResult}, like_hi::Real;
                     figdir::AbstractString, noises = noise_curves(),
                     ext::AbstractString = FIG_EXT)
    isempty(results) && error("nothing to plot")
    mkpath(figdir)
    ordered = by_z_order(results, like_hi)
    written = Dict{String,String}()

    function save(p, name)
        path = joinpath(figdir, name)
        _savefig(p, path)
        written[splitext(name)[1]] = path
        @info "  wrote" figure = joinpath(basename(figdir), name)
    end

    let p = strain_axes(_SH_LABEL, "Peaks")
        for r in ordered
            _draw_peaks!(p, r, like_color(r.like, like_hi))
        end
        save(add_noise!(p, noises), "peaks.$ext")
    end

    let p = strain_axes(_SH_LABEL, "Curves")
        for r in ordered
            _draw_curves!(p, r, like_color(r.like, like_hi))
        end
        save(add_noise!(p, noises), "curves.$ext")
    end

    let p = strain_axes(_HC_LABEL, "Characteristic")
        for r in ordered
            col = like_color(r.like, like_hi)
            _draw_characteristic!(p, r, col)
            _draw_quadrupole!(p, r, col)
        end
        save(add_noise!(p, noises), "characteristic.$ext")
    end

    let p = strain_axes(_SH_LABEL, "Envelopes")
        for r in ordered
            _draw_envelopes!(p, r, like_color(r.like, like_hi))
        end
        save(add_noise!(p, noises), "envelopes.$ext")
    end

    return written
end

"""
    plot_ids(ids; outdir = OUT_DIR, figdir = nothing)

Load the results for `ids` and write one set of strain figures covering them
all. `ids` may be an integer, an iterable (`0:5`, `[0, 2, 5]`), or `:all`.

Figures go to `figures_<tag>/`, where the tag names the files plotted, so
per-file and combined runs do not overwrite each other.
"""
function plot_ids(ids; outdir = OUT_DIR, figdir = nothing, ext::AbstractString = FIG_EXT)
    results, like_hi, sources = load_results(ids; outdir)
    tag = _tag(unique(sources))
    dir = figdir === nothing ? joinpath(outdir, "figures_$tag") : figdir
    @info "plotting" files = ids eos = length(results) into = basename(dir)
    plot_strain(results, like_hi; figdir = dir, ext)
    return
end

# "0" | "0-5" for a contiguous run | "0_2_5" otherwise
function _tag(ids)
    length(ids) == 1 && return string(first(ids))
    s = sort(collect(ids))
    return s == first(s):last(s) ? "$(first(s))-$(last(s))" : join(s, "_")
end

# Gray key mapping marker/style to dataset name, so overlaid point clouds can be
# told apart (colour already encodes likelihood, not which dataset). The dummy
# series carry NaN coordinates: they never touch the axes, only the legend.
function _overlay_legend!(p, layers)
    length(layers) > 1 || return p
    for name in keys(STRAIN_LAYERS)
        name in layers || continue
        L = STRAIN_LAYERS[name]
        if name === :envelopes
            plot!(p, [NaN, NaN], [NaN, NaN]; seriestype = :shape,
                  fillcolor = :gray, fillalpha = 0.5, linecolor = :gray, label = string(name))
        elseif L.marker === :none
            plot!(p, [NaN, NaN], [NaN, NaN]; color = :gray, linewidth = 1.5, label = string(name))
        else
            scatter!(p, [NaN], [NaN]; markershape = L.marker, color = :gray,
                     markersize = 4, markerstrokewidth = 0, label = string(name))
        end
    end
    plot!(p; legend = :topright, legendfontsize = 9)
    return p
end

"""
    plot_overlay(datasets, ids; kwargs...) -> path

Overlay several strain datasets on one figure for the given result ids.

`datasets` selects the layers: a string like `"peaks, envelopes"`, or a vector
`["peaks", "envelopes", "quadrupole"]`. `ids` is as in [`plot_ids`](@ref) — an
integer, an iterable (`0:2`), or `:all`. Available layers: $(join(keys(STRAIN_LAYERS), ", ")).

Layers draw back-to-front (envelopes, curves, then point clouds), each in
likelihood order, so the highest-likelihood EOS sit on top. Point datasets get
distinct marker shapes; colour encodes likelihood throughout. A gray key naming
each layer is drawn unless `legend = false`.

Keywords: `outdir`, `figdir`, `ext`, `ylabel`, `title`, `name`, `noises`,
`legend`. The figure is written to `figures_<tag>/` (as `plot_ids`), named for
the datasets unless `name` is given.
"""
function plot_overlay(datasets, ids;
                      outdir = OUT_DIR, figdir = nothing, ext::AbstractString = FIG_EXT,
                      ylabel = nothing, title = "", name = nothing,
                      noises = noise_curves(), legend::Bool = true)
    layers = _parse_layers(datasets)
    results, like_hi, sources = load_results(ids; outdir)
    isempty(results) && error("nothing to plot")
    ordered = by_z_order(results, like_hi)

    if ylabel === nothing
        kinds = unique(STRAIN_LAYERS[n].kind for n in layers)
        ylabel = length(kinds) == 1 ? _layer_label(only(kinds)) : _GEN_LABEL
    end

    p = strain_axes(ylabel, title)
    # canonical back-to-front order, restricted to the requested layers
    for name_ in keys(STRAIN_LAYERS)
        name_ in layers || continue
        L = STRAIN_LAYERS[name_]
        for r in ordered
            L.draw(p, r, like_color(r.like, like_hi); marker = L.marker)
        end
    end
    legend && _overlay_legend!(p, layers)
    add_noise!(p, noises)

    base = name === nothing ? "overlay_" * join(string.(layers), "_") : name
    dir = figdir === nothing ? joinpath(outdir, "figures_$(_tag(unique(sources)))") : figdir
    mkpath(dir)
    path = joinpath(dir, "$base.$ext")
    _savefig(p, path)
    @info "overlay" datasets = layers files = ids eos = length(results) wrote = joinpath(basename(dir), "$base.$ext")
    return path
end

# ---------------------------------------------------------------------------
# BubblePlot: one EOS, a denser grid, points sized by bubble radius
# ---------------------------------------------------------------------------

"""
    _size_legend(rows, msize_lo, msize_hi) -> Plot

Bare panel showing what the marker sizes in the bubble plot mean: the smallest
and the largest bubble actually drawn, at exactly the sizes the main plot uses,
each labelled with its radius and peak frequency.

Both entries are real rows taken from the extremes of `rows`, so the two numbers
next to them describe points that are genuinely on the plot.
"""
function _size_legend(rows, msize_lo, msize_hi)
    lo_row = rows[argmin(x.R_bubble_m for x in rows)]
    hi_row = rows[argmax(x.R_bubble_m for x in rows)]

    # The panel spans y in 0..1; the two entries sit symmetric about the centre,
    # `GAP` apart, so shrinking GAP pulls them together without moving the pair
    # off centre. x runs past 1 so the left-aligned labels have room; anything
    # past the limit is clipped.
    #
    # The colourbar's label is annotated here (at the negative x) rather than set
    # as `colorbar_title` on the 3D panel: GR fixes that title a set step from
    # the bar, so lifting it off the tick numbers needed a wide right margin,
    # which opened a gap between bar and legend. As an annotation its distance
    # from the bar is just this coordinate, and the margin can stay tight.
    GAP      = 0.22    # vertical separation of the two markers, panel fraction
    marker_x = 0.18
    label_x  = 0.35
    y_big, y_small = 0.5 + GAP / 2, 0.5 - GAP / 2

    q = plot(; xlims = (0, 1.2), ylims = (0, 1),
             framestyle = :none, legend = false, grid = false)
    annotate!(q, -0.22, 0.5,
              text(L"$N_\mathrm{bubbles}$", 14, :center, rotation = 90))
    for (row, ms, y) in ((hi_row, msize_hi, y_big), (lo_row, msize_lo, y_small))
        scatter!(q, [marker_x], [y]; markersize = ms, markercolor = :gray,
                 markeralpha = 0.7, markerstrokecolor = :black,
                 markerstrokewidth = 0.5, label = "")
        annotate!(q, label_x, y + 0.035,
                  text(latexstring(@sprintf("\$R = %.4g\\,\\mathrm{m}\$", row.R_bubble_m)), 9, :left))
        annotate!(q, label_x, y - 0.035,
                  text(latexstring(@sprintf("\$f_\\mathrm{peak} = %.3g\\,\\mathrm{MHz}\$", row.fpeak_MHz)),
                       9, :left))
    end
    return q
end

"""
    plot_bubbles(h5_id, index; kwargs...)

3D scatter over the (Ṁ, σ, v_w) grid for a single sample, coloured by bubble
count and sized by bubble radius. `index` is the position in that file's
accepted list, not the sample id.

This re-sweeps on its own denser grid, so it does not use the saved results.
"""
function plot_bubbles(h5_id::Integer, index::Integer;
                      accretions = 0.1:0.1:0.9, sigmas = 10.0:5.0:40.0,
                      lambdas = LAMBDAS, vs = 0.01:0.01:0.5,
                      datadir = DATA_DIR, figdir = nothing, outdir = OUT_DIR,
                      ext::AbstractString = FIG_EXT)
    h5path = joinpath(datadir, "$h5_id.h5")
    accepted = accepted_ids(joinpath(datadir, "accepted.json"), "$h5_id.h5")
    1 ≤ index ≤ length(accepted) ||
        error("index $index out of range (file $h5_id.h5 has $(length(accepted)) accepted samples)")
    id = accepted[index]

    prep = prepare_eos(h5open(f -> load_branch(f, id), h5path, "r"))
    prep === nothing && error("prepare_eos failed for sample $id of $h5_id.h5")

    rows = sweep(prep, accretions, sigmas, lambdas, vs)
    hf = filter(r -> r.N_bubbles ≥ 2 && r.fpeak_MHz ≥ 0.0001, rows)
    isempty(hf) && error("no surviving rows for sample $id of $h5_id.h5")

    # Draw the points back-to-front for the viewing angle (painter's algorithm):
    # GR renders markers in array order and does not depth-sort, so without this
    # a background bubble can cover a foreground one. Project each point (after
    # normalising the three axes to a common 0..1 box, since they have unrelated
    # units) onto the direction of the camera, then plot farthest first so the
    # nearest bubbles land on top. `CAMERA` must match the `camera` kwarg below.
    CAMERA = (30, 30)   # (azimuth, elevation), degrees
    unit(v) = (lo = minimum(v); hi = maximum(v); hi > lo ? (v .- lo) ./ (hi - lo) : zero(v))
    let a = deg2rad(CAMERA[1]), e = deg2rad(CAMERA[2])
        toward = (cos(e) * cos(a), cos(e) * sin(a), sin(e))
        depth = unit([r.accretion for r in hf])     .* toward[1] .+
                unit([r.sigma_MeV_fm2 for r in hf])  .* toward[2] .+
                unit([r.v_wall for r in hf])         .* toward[3]
        hf = hf[sortperm(depth)]
    end

    nb = [r.N_bubbles for r in hf]
    rm = [r.R_bubble_m for r in hf]
    rm_lo, rm_hi = extrema(rm)
    msize_lo, msize_hi = 3.0, 14.0                              # marker points
    rescale(v, lo, hi, a, b) = a + (b - a) * (v - lo) / (hi - lo)
    sizes = [rescale(x, rm_lo, rm_hi, msize_lo, msize_hi) for x in rm]

    # Axis limits hugging the data with a small fractional margin. GR's automatic
    # 3D limits are not dependable here — inside the subplot, with the margins and
    # an explicit camera, it inflates the box well past the points — so the box is
    # pinned to the data extent plus 6% of the span on each side.
    #
    # Ticks have to be pinned too: given only limits, GR in a small 3D box drags
    # the box back out to whatever round value its own tick search lands on,
    # undoing the tight fit. Passing `optimize_ticks` values (the same nice
    # numbers it would pick, but computed over the *data* range) keeps both the
    # ticks and the box where we put them.
    X = [r.accretion for r in hf]
    Y = [r.sigma_MeV_fm2 for r in hf]
    Z = [r.v_wall for r in hf]
    pad(v; frac = 0.06) = begin
        lo, hi = extrema(v)
        d = hi > lo ? (hi - lo) * frac : abs(lo) * frac + eps()
        (lo - d, hi + d)
    end
    ticks(v) = (lo = minimum(v); hi = maximum(v); Plots.optimize_ticks(lo, hi)[1])
    p = scatter3d(X, Y, Z;
                  marker_z = nb, c = BUBBLE_CMAP, markersize = sizes,
                  markerstrokecolor = :black, markerstrokewidth = 0.5,
                  markeralpha = 0.9, camera = CAMERA,
                  xlabel = L"$\dot{M}\;\left[M_\odot/\mathrm{s}\right]$",
                  ylabel = L"$\sigma\;\left[\mathrm{MeV}/\mathrm{fm}^2\right]$",
                  zlabel = L"$v_w/c$",
                  xlims = pad(X), ylims = pad(Y), zlims = pad(Z),
                  xticks = ticks(X), yticks = ticks(Y), zticks = ticks(Z),
                  # No `colorbar_title`: it is annotated on the legend panel
                  # instead, which frees the right margin to be as tight as the
                  # tick numbers need. See `_size_legend`.
                  #
                  # GR gives the colourbar the full height of the panel, which
                  # made it tower over a 3D box that never fills its space
                  # vertically. The top/bottom margins trade that unused slack
                  # for a shorter bar; they shrink the axes and the bar together.
                  right_margin = 1mm, top_margin = 20mm, bottom_margin = 15mm, left_margin = 1mm,
                  label = "", size = (700, 700), title = "Sample $id of $h5_id.h5")

    # The size legend has to be its own panel rather than legend entries: GR
    # clamps the marker in a legend swatch to a fixed size, so a legend cannot
    # show that one bubble is five times another. Here the markers are drawn on a
    # bare axis at exactly the sizes `sizes` uses, so the scale is readable.
    legend_panel = _size_legend(hf, msize_lo, msize_hi)

    fig = plot(p, legend_panel;
               layout = grid(1, 2; widths = [0.7, 0.3]), size = (700, 500))

    dir = figdir === nothing ? joinpath(outdir, "figures_$h5_id") : figdir
    mkpath(dir)
    path = joinpath(dir, "bubbles_$id.$ext")
    _savefig(fig, path)

    @info "bubble plot" file = "$h5_id.h5" sample = id rows = length(rows) plotted = length(hf)
    @printf("  Rmin : %.4g m (f_peak %.4g MHz)\n  Rmax : %.4g m (f_peak %.4g MHz)\n",
            rm_lo, maximum(r.fpeak_MHz for r in hf),
            rm_hi, minimum(r.fpeak_MHz for r in hf))
    @info "  wrote" figure = joinpath(basename(dir), basename(path))
    return
end

# ---------------------------------------------------------------------------
# PTtools variant — one accepted sample at a time
#
# `process_eos_pt` (in the module) does the physics; these wrap it in the same
# select / sweep / plot shape as run_sweep and plot_ids.
#
#     julia> plot_compare(0; eos = 2672)          # both kernels, one figure
#     julia> plot_compare(0; like = 1.0)          # the most likely accepted sample
#     julia> plot_pt(0; eos = 2672)               # PTtools only
#     julia> id, old, pt = sweep_pt(0; like = 0.5)  # the numbers
# ---------------------------------------------------------------------------

"""
    pick_eos(h5_id; eos = nothing, like = nothing) -> id

Choose one accepted sample of `<h5_id>.h5`, either by its id or by normalized
likelihood: `like = 1` is the most likely accepted sample, `like = 0` the least.
"""
function pick_eos(h5_id::Integer; eos = nothing, like = nothing, datadir = DATA_DIR)
    accepted = accepted_ids(joinpath(datadir, "accepted.json"), "$h5_id.h5")
    if eos !== nothing
        eos in accepted ||
            error("EoS $eos is not accepted in $h5_id.h5 ($(length(accepted)) accepted)")
        return eos
    end
    like === nothing && error("give either eos = <id> or like = <0..1>")
    likes = h5open(joinpath(datadir, "$h5_id.h5"), "r") do f
        [first(read(f, "$id/params/ptot")) for id in accepted]
    end
    hi = maximum(likes)
    k = argmin(abs.(likes ./ hi .- like))
    @info "picked" eos = accepted[k] normalized_like = round(likes[k] / hi; digits = 4)
    return accepted[k]
end

"""
    sweep_pt(h5_id; eos = nothing, like = nothing, ...) -> (id, res_old, res_pt)

Sweep ONE accepted sample both ways: `res_old` with the current assumptions,
`res_pt` with the PTtools substitution. Either result may be `nothing`.
"""
function sweep_pt(h5_id::Integer; eos = nothing, like = nothing,
                  accretions = ACCRETIONS, sigmas = SIGMAS,
                  lambdas = LAMBDAS, vs = VS, datadir = DATA_DIR)
    id = pick_eos(h5_id; eos, like, datadir)
    noise = first(noise_curves())      # the SNR is quoted against MWB-DMR
    h5open(joinpath(datadir, "$h5_id.h5"), "r") do f
        br = load_branch(f, id)
        prep = prepare_eos(br)
        prep === nothing && error("prepare_eos failed for EoS $id")
        ptp = load_PTparams(f, id)
        old = process_eos(br, prep, noise; accretions, sigmas, lambdas, vs)
        pt  = process_eos_pt(br, prep, ptp, noise; accretions, sigmas, lambdas, vs)
        @info "swept" eos = id rows_old = old === nothing ? 0 : length(old.rows) rows_pt = pt === nothing ? 0 : length(pt.rows)
        return id, old, pt
    end
end

"Draw the strain curves of `rows`, coloured by wall velocity; `vws` sets the scale."
function _draw_curves!(p, rows, curves, vws, style, tag)
    # A plain colour vector: indexing a cgrad with an integer reads it as a
    # position in 0..1, so every curve would come out the same colour.
    cols = [cgrad(:viridis)[x] for x in range(0.1, 0.85; length = max(length(vws), 2))]
    seen = Set{Float64}()
    for (c, r) in zip(curves, rows)
        r.v_wall in vws || continue
        j = findfirst(==(r.v_wall), vws)
        lab = r.v_wall in seen ? "" : "v_w=$(r.v_wall) $tag"
        push!(seen, r.v_wall)
        plot!(p, c[:, 1], c[:, 2]; color = cols[j], linestyle = style,
              linewidth = 1.2, label = lab)
    end
    return p
end

"""
    plot_pt(h5_id; eos = nothing, like = nothing) -> path

Strain curves of one accepted sample, computed with PTtools.
"""
function plot_pt(h5_id::Integer; eos = nothing, like = nothing,
                 outdir = OUT_DIR, figdir = nothing, ext::AbstractString = FIG_EXT, kw...)
    id, _, pt = sweep_pt(h5_id; eos, like, kw...)
    pt === nothing && error("no surviving PTtools rows for EoS $id")
    p = strain_axes(L"$\sqrt{S_h}\;\left[\mathrm{Hz}^{-1/2}\right]$", "EoS $id — PTtools")
    plot!(p; colorbar = false, legend = :bottomleft)
    _draw_curves!(p, pt.rows, pt.curves, sort(unique(r.v_wall for r in pt.rows)), :solid, "")
    for nc in noise_curves()
        plot!(p, nc.f, nc.asd; color = :gray, linestyle = :dot, label = "")
    end
    _frame_on_top!(p)

    dir = figdir === nothing ? joinpath(outdir, "figures_$h5_id") : figdir
    mkpath(dir)
    return _savefig(p, joinpath(dir, "pttools_$id.$ext"))
end

"""
    plot_compare(h5_id; eos = nothing, like = nothing) -> path

Both strain kernels for one accepted sample on one figure: solid = current
assumptions, dashed = PTtools, coloured by wall velocity.
"""
function plot_compare(h5_id::Integer; eos = nothing, like = nothing,
                      outdir = OUT_DIR, figdir = nothing,
                      ext::AbstractString = FIG_EXT, kw...)
    id, old, pt = sweep_pt(h5_id; eos, like, kw...)
    (old === nothing || pt === nothing) && error("nothing to compare for EoS $id")
    # colour by the wall velocities PTtools kept, so both sets share a scale
    vws = sort(unique(r.v_wall for r in pt.rows))
    p = strain_axes(L"$\sqrt{S_h}\;\left[\mathrm{Hz}^{-1/2}\right]$",
                    "EoS $id — solid: current, dashed: PTtools")
    plot!(p; colorbar = false, legend = :bottomleft, legendfontsize = 7)
    _draw_curves!(p, old.rows, old.curves, vws, :solid, "current")
    _draw_curves!(p, pt.rows,  pt.curves,  vws, :dash,  "PTtools")
    for nc in noise_curves()
        plot!(p, nc.f, nc.asd; color = :gray, linestyle = :dot, label = "")
    end
    _frame_on_top!(p)

    # What actually moved. `peaks[:, 1]` is the nucleation f_peak, identical in
    # both by construction, so compare where each SPECTRUM peaks instead.
    @printf("%-6s %11s %11s %7s %10s %10s %7s\n",
            "v_w", "f_spec old", "f_spec PT", "ratio", "√Sh old", "√Sh PT", "ratio")
    for v in vws
        ko = findfirst(r -> r.v_wall == v, old.rows)
        kp = findfirst(r -> r.v_wall == v, pt.rows)
        (ko === nothing || kp === nothing) && continue
        co, cp = old.curves[ko], pt.curves[kp]
        io, ip = argmax(co[:, 2]), argmax(cp[:, 2])
        @printf("%-6.2f %11.4g %11.4g %7.2f %10.3e %10.3e %7.3f\n",
                v, co[io, 1], cp[ip, 1], cp[ip, 1] / co[io, 1],
                co[io, 2], cp[ip, 2], cp[ip, 2] / co[io, 2])
    end

    dir = figdir === nothing ? joinpath(outdir, "figures_$h5_id") : figdir
    mkpath(dir)
    return _savefig(p, joinpath(dir, "compare_$id.$ext"))
end

# ---------------------------------------------------------------------------
# Batch entry point: `julia --project=. -t auto run_PTeosMHzGWs.jl 0 1 2`
# Skipped when the file is `include`d interactively (ARGS empty).
# ---------------------------------------------------------------------------

if (abspath(PROGRAM_FILE) == @__FILE__()) && !isempty(ARGS)
    ids = parse.(Int, ARGS)
    for id in ids
        run_sweep(id)
    end
    plot_ids(ids)
end
