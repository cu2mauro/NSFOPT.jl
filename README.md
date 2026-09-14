# NS_FOPT.jl

Computes the frequency and amplitude of MHz gravitational waves radiated when a
first-order quark–hadron phase transition nucleates inside an accreting neutron
star, and compares the resulting strain against present and proposed detector
sensitivities.

This is the analysis code for the accompanying paper. It ships a small sample
dataset, so everything below runs on a fresh clone without any external input.

## Quick start

```bash
git clone https://github.com/cu2mauro/NS_FOPT.jl
cd NS_FOPT.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then, from an interactive session:

```julia
julia --project=. -t auto

julia> include("scripts/run_PTeosMHzGWs.jl")
julia> run_sweep(0)     # sweeps the sample dataset, writes results/results_0.h5
julia> plot_ids(0)      # writes the figures under results/figures_0/
```

`-t auto` can only be given at startup — the sweep is threaded and threads
cannot be added to a running session. The first call pays a few seconds of
package load and JIT; later calls in the same session do not, which is why
interactive use is preferred over the batch form:

```bash
julia --project=. -t auto scripts/run_PTeosMHzGWs.jl 0
```

On the shipped dataset (119 accepted samples) the sweep takes about 5 s on a
laptop; a full-size sampler file takes correspondingly longer.

## What is here

```
src/PTeosMHzGWs.jl          the physics: nucleation, bubble growth, strain, SNR
scripts/run_PTeosMHzGWs.jl  driver — sweeping, plotting, figure output
scripts/FilterEOS.jl        selects EOS samples that can produce a signal
data/noise_curves/          detector sensitivity curves (see Provenance)
data/sample/                a 1000-EOS sampler run, enough to run everything
docs/data_format.md         HDF5 layout of the input data
```

Sweeping and plotting are deliberately separate. `run_sweep` does the expensive
work once per input file and writes `results_<id>.h5`; the plot functions read
those back, so any set of files can be replotted or combined without sweeping
again.

```julia
plot_ids(0)        # figures for one file
plot_ids(0:1)      # one figure per kind, both files combined
plot_ids(:all)     # every results_*.h5 in the output folder
plot_bubbles(0, 101)   # bubble-evolution plot for one sample
```

## The physics, in brief

Each EOS sample carries two stellar sequences built from two versions of the
same equation of state, identical below the transition pressure `pPT` and
different above it. `TOV` is the full one, whose stars carry a quark core;
`TOVext` is the purely hadronic branch continued past `pPT`, metastable, and the
state the star is actually in until a bubble nucleates.

An accreting star climbs the hadronic sequence. Once its centre passes `pPT` the
hadronic phase is overpressed with respect to quark matter by

```
Δp(t) = p_cent(TOV) − p_cent(TOVext)
```

at fixed accreted mass. This drives the thin-wall O(4) bounce action
`S = 27π²σ⁴/(2Δp³)`: the nucleation rate goes as `exp(−S)`, so it is negligible
until `Δp` has grown enough and then switches on sharply. The transition
completes in a shell of bubbles whose size sets the GW peak frequency, while the
quark-core radius at nucleation sets how many bubbles fit.

The pipeline, per accepted sample:

1. `prepare_eos` builds the overpressure `Δp` driving nucleation and the
   quark-core radius `R_c(M)` from the two mass sequences.
2. `compute_row` integrates the nucleation ODE at one point of the
   (accretion rate, surface tension, Λ, wall velocity) grid, returning peak
   frequency, bubble radius, core radius and bubble count.
3. `process_eos` sweeps the grid and converts each surviving row into a strain
   spectrum, an SNR against a detector noise curve, and a characteristic-strain
   point.

Internally masses are in M☉, times in ms, pressures in GeV⁴, lengths in m and
frequencies in Hz, unless a name says otherwise.

## Parameters

The swept grid is set near the top of `scripts/run_PTeosMHzGWs.jl` and every
value is overridable per call, so none of them needs editing to change a run:

| Constant | Meaning | Default |
| --- | --- | --- |
| `ACCRETIONS` | accretion rate Ṁ past criticality, M☉/s | `0.2:0.2:1.0` |
| `SIGMAS` | surface tension, MeV/fm² | `10.0:10.0:50.0` |
| `LAMBDAS` | energy scale, MeV | `[200.0]` |
| `VS` | wall velocity, units of c | `0.01 … 0.65` |
| `FIG_EXT` | `"pdf"` for papers, `"png"` for a quick look | `"pdf"` |

```julia
run_sweep(0; sigmas = 20.0:5.0:40.0, vs = [0.1, 0.2])
```

## Using your own data

`data/sample/0.h5` is a complete EOSsampler run, not an extract — the file the
sampler wrote, untouched — so it is fully specified by the settings that
produced it:

| | |
| --- | --- |
| config | `config/default.yaml` with `n_eos: 1000` |
| seed | `12345` (EOSsampler's default) |
| entries | 1000 — 832 samples, 168 `rej_*` |
| accepted | 119 pass the four clauses in `scripts/FilterEOS.jl` |
| size | 53 MB |
| pipeline | EOSsampler's `EOS` → `scripts/filter1_eos.py` → `TOV` |

Re-running EOSsampler that way regenerates it, given the same build: the C++
`normal_distribution` is implementation-defined, so another standard library
draws a different sequence from the same seed. Because the file is genuine
output rather than a curated subset, `scripts/FilterEOS.jl` has real work to do
on it and `run_sweep(0)` produces a genuine `results_0.h5`.

The published figures use a far larger run of exactly the same kind — 50 files,
≈27 GB — so the figures you get here carry ~119 EOS where the paper's carry tens
of thousands. Nothing else differs: the same code paths run on either.

To run against a full dataset, point `NS_FOPT_DATA` at it:

```bash
NS_FOPT_DATA=/path/to/your/run julia --project=. -t auto
```

or pass `datadir = "..."` to any function. `NS_FOPT_OUT` relocates the output
directory the same way.

### Regenerating the full dataset

The input files are produced by Christian Ecker's
[EOSsampler](https://github.com/EckerChristian/EOSsampler) — build and run it
following its own README. Its sampler takes a random seed on the command line,
so a run is reproducible given the same seed, configuration and build.

Nothing here assumes a particular shape. `FilterEOS.jl` scans whatever `*.h5` it
finds and the sweep adapts to the sample count, so a smaller run works too, with
correspondingly fewer points in the figures. File names are the ids passed to
`run_sweep`.

Once the run exists, index it:

```bash
julia --project=. scripts/FilterEOS.jl /path/to/your/run
```

`FilterEOS.jl` writes `accepted.json` beside the HDF5 files, indexing the
samples that can produce a signal at all. See [docs/data_format.md](docs/data_format.md)
for the full HDF5 layout and the four selection clauses.

## Provenance of the noise curves

| File | Detector |
| --- | --- |
| `MWB-DMR-res.csv` | microwave-cavity / DMR concept, the curve the quoted SNRs use |
| `levitatedSensors_100m.csv` | levitated-sensor detector, 100 m |
| `LIGO5.tad` | LIGO |
| `ET-0001A-18_ETDSensitivityCurveTxtFile.txt` | Einstein Telescope, ET-D |
| `cosmic_explorer_strain.txt` | Cosmic Explorer |

## Requirements

Julia 1.10 or newer. `Pkg.instantiate()` uses the checked-in `Manifest.toml`,
which was resolved under Julia 1.12.6: on a 1.12 release you get exactly the
package versions the published results were produced with, and on 1.10 or 1.11
Pkg re-resolves to the nearest set valid for that release.

**Nothing above needs Python.** The dependencies are Julia-only and no conda
environment is built on first run. Two comparisons are opt-in:

| To use | Install |
| --- | --- |
| `process_eos_pt` / `PTtools`, the [pttools-gw](https://pypi.org/project/pttools-gw/) sound-shell comparison | `Pkg.add(["PythonCall", "CondaPkg"])`, then `using CondaPkg; CondaPkg.add_pip("pttools-gw")` |
| `pyplot()` instead of the default `gr()` plotting backend | `Pkg.add("PyPlot")` |

Calling `process_eos_pt` without them says so and names the command to run.

`Revise` is a real dependency, for the editing loop: `includet` means edited
functions take effect without a restart. Module-level `const`s do not — re-tune
one in place with

```julia
Core.eval(PTeosMHzGWs, :(const OMEGA_GW = 0.02))
```

or restart. `struct`s always need a restart.

## Licence

MIT — see [LICENSE](LICENSE).
