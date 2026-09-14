# Input data format

Everything in this repository reads the HDF5 files written by
[EOSsampler](https://github.com/EckerChristian/EOSsampler), plus one
`accepted.json` index produced by `scripts/FilterEOS.jl`. This page documents
both so the pipeline can be driven from data produced elsewhere.

## Directory

A data directory holds one or more sample files and the index:

```
<run directory>/
  0.h5            one group per sampled entry
  1.h5            …
  accepted.json   {"0.h5": [4390, 3129, …], "1.h5": […]}
```

`accepted.json` maps a file name to the ids of the samples that passed the
four physical clauses in `scripts/FilterEOS.jl`. It is regenerated, not edited
by hand.

## Sample file

Each `<n>.h5` has one top-level group per entry. Samples are named by their
integer id as a string (`"0"`, `"1"`, …). A file may also hold entries that are
not samples — groups named `rej_<n>`, with no complete `TOV`/`TOVext` pair — so
sample ids are not contiguous even before filtering. The shipped
`data/sample/0.h5` holds 832 samples and 168 such groups in its 1 000 entries.
Nothing here reads them: `sample_ids` keeps only all-digit group names and
`complete_sample` requires the TOV datasets. Read the ids from `accepted.json`
or from `sample_ids` rather than assuming a range.

Every sample group carries five subgroups. Array lengths differ between samples
and between the two branches; only lengths *within* a branch are tied together.

### `EOS/` and `EOSext/` — the equations of state

Tabulated on a shared pressure grid per branch.

| Dataset | Meaning |
| --- | --- |
| `p` | pressure, MeV/fm³ |
| `e` | energy density, MeV/fm³ |
| `n` | baryon number density, 1/fm³ |
| `mu` | baryon chemical potential, MeV |
| `cs2` | squared sound speed, units of c² |

`EOS` is the full equation of state: hadronic, a first-order transition at
`params/pPT`, then quark matter. `EOSext` is the purely hadronic branch
continued past `pPT` — the metastable state the star actually occupies until a
bubble nucleates. The two coincide below `pPT`.

### `TOV/` and `TOVext/` — the stellar sequences

One entry per central pressure, integrated on a grid shared between the two
branches: **the same index in `TOV` and `TOVext` is the same central pressure.**
That is what makes the element-by-element mass comparisons in `FilterEOS.jl`
meaningful. Central *density* is not shared, since the two equations of state
give different `n` at the same `p` above the transition.

| Dataset | Meaning |
| --- | --- |
| `M` | gravitational mass, M☉ |
| `Mb` | baryonic mass, M☉ |
| `Nb` | baryon number |
| `R` | circumferential radius, km |
| `Rqm` | quark-core radius, km — identically zero on `TOVext` |
| `Lambda` | dimensionless tidal deformability |
| `stab` | 1.0 if the configuration is stable, 0.0 otherwise |
| `p_cent` | central pressure, MeV/fm³ |
| `e_cent`, `n_cent`, `mu_cent`, `cs2_cent` | central values, units as in `EOS/` |

### `params/` — scalars

Length-1 arrays (read with `first`), except `pXray`.

| Dataset | Meaning |
| --- | --- |
| `pPT` | transition pressure, MeV/fm³. Zero when the sample has no transition |
| `ptot` | total likelihood — the weight used to colour the figures |
| `nPTl`, `ePTl`, `muPT` | density, energy density and chemical potential at the low-pressure side of the transition |
| `cs2PTl`, `cs2PTr` | squared sound speed either side of the transition |
| `dn` | density jump across the transition |
| `nbranches` | number of stable branches (`Int32`) |
| `pCET`, `pPQCD`, `pM`, `pBH`, `pGW` | individual likelihood factors |
| `pXray` | per-source X-ray likelihoods, length 15 |

## Minimum required datasets

A sample is evaluated only if it carries all of these; anything missing and it
is counted as incomplete and skipped rather than aborting the scan:

```
TOV/stab   TOV/Rqm   TOV/M
TOVext/stab   TOVext/M   TOVext/p_cent
EOSext/p   params/pPT
```

`EOSext/p` is never read by the clauses. It is a guard: a sample without an
`EOSext` group is broken rather than merely unlucky, and in a 60k-sample run
exactly one entry was, alongside `pPT = ptot = 0`.

## Units

The HDF5 stores pressures and energy densities in MeV/fm³ and radii in km. The
module converts on read — internally masses are in M☉, times in ms, pressures
in GeV⁴, lengths in m and frequencies in Hz, unless a name says otherwise.

## Producing your own

```bash
git clone https://github.com/EckerChristian/EOSsampler
# build and run per that repository's README, then index the output:
julia --project=. scripts/FilterEOS.jl /path/to/run
```

`FilterEOS.jl` writes `accepted.json` beside the HDF5 files. Any number of
files works; the sweep adapts to whatever each one holds. The settings behind
the shipped `data/sample/0.h5` are listed in the README.
