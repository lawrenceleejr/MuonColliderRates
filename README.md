# Muon Collider Rates

A compilation of cross sections and interaction rates for a multi-TeV muon
collider, from 1 to 10 TeV — the physics programme, the beam-induced
backgrounds, and the machine reference rates on one pair of axes.

**[→ Interactive version](https://lawrenceleejr.github.io/MuonColliderRates/)**
· [figure (PDF)](https://lawrenceleejr.github.io/MuonColliderRates/MuonColliderRates.pdf)

![The figure](https://lawrenceleejr.github.io/MuonColliderRates/MuonColliderRates.png)

The left axis is the cross section in femtobarns; the right axis is the same
number expressed as a rate at the nominal 10 TeV operating point,
`L = 2 × 10³⁵ cm⁻² s⁻¹`. Curves are grouped into machine-induced and inclusive
backgrounds (grey), Standard Model processes, and BSM benchmarks.

Compiled by L. Lee and T. Holmes. Corrections and additions are welcome — see
[Contributing](#contributing).

---

## Repository layout

| Path | What it is |
|------|------------|
| `data/` | One plain text file per dataset, each carrying its own provenance header |
| `mcrates.py` | Loader for those files plus the single registry of how each curve is drawn |
| `plot.py` | Builds `MuonColliderRates.pdf` / `.png` |
| `plotLuminosity.py` | Builds the separate instantaneous-luminosity figure |
| `make_web_data.py` | Exports `data/` to `web/data/curves.json` for the interactive page |
| `web/` | Source of the GitHub Pages site |
| `guineapig/` | Inputs and runner for the GuineaPig incoherent-pair calculation |
| `helperFunctions.py` | Small matplotlib helpers (axis "breathing", endpoint padding) |

## Running it

```bash
pip install matplotlib
pip install "matplotlib_tufte @ git+https://github.com/ninivert/matplotlib_tufte.git"

python plot.py            # -> MuonColliderRates.pdf and .png
python make_web_data.py   # -> web/data/curves.json
```

To preview the interactive page, serve `web/` over HTTP (it fetches
`data/curves.json`, which `file://` will not allow):

```bash
python -m http.server --directory web 8000   # then open http://localhost:8000
```

CI renders the figure on every push and redeploys the page from `main`.

## Data format

Every file in `data/` is comma-separated text with a small header. The header
is not decoration: `mcrates.py` parses the `# columns:` line and converts to
femtobarns, so nothing downstream carries a magic unit factor.

```
# title: VBF Z
# process: mu+ mu- -> Z nu nubar (vector-boson fusion)
# source: https://arxiv.org/abs/2005.10289 (digitised)
# notes: optional, anything a reader would want to know
# columns: sqrt_s [TeV], sigma [fb]
1.0144931295661115, 883.3407454482787
1.049350284931249, 928.6951375573958
...
```

Rules:

* The first column is always √s in TeV.
* Each column is declared as `name [unit]`. Cross sections may be quoted in
  `fb`, `pb`, `nb`, `ub`, `mb` or `b`; they are converted on load.
* Lines starting with `#` are comments; blank lines are ignored.
* Files may carry extra columns (uncertainties, or several related series, as
  `thermalwimp.txt` does) — name them in the `# columns:` line and select them
  from the curve registry.

## The datasets

| File | Process | Source |
|------|---------|--------|
| `vbfz.txt` | μ⁺μ⁻ → Z νν̄ | GGI lectures ([slides](https://indico.cern.ch/event/1564581/contributions/6591806/attachments/3097402/5487468/GGI-course-slides.pdf)), digitised |
| `vbfh.txt`, `vbfhh.txt`, `vbfhhh.txt` | VBF H, HH, HHH | GGI lectures, digitised |
| `vbftt.txt`, `vbftth.txt` | VBF tt̄, tt̄H | GGI lectures, digitised |
| `anntt.txt`, `anntth.txt` | s-channel annihilation to tt̄, tt̄H | GGI lectures, digitised |
| `vbfww.txt`, `vbfwwz.txt` | VBF WW, WWZ | [arXiv:2005.10289](https://arxiv.org/abs/2005.10289), digitised |
| `jj.txt` | Soft dijets, p_T > 5–7 GeV, \|η\| < 3.13 | [arXiv:2103.09844](https://arxiv.org/abs/2103.09844) fig. 5b, digitised |
| `lltohadrons.txt` | Inclusive μμ → hadrons | [arXiv:2103.09844](https://arxiv.org/abs/2103.09844) fig. 4, digitised |
| `mumu.txt` | μ⁺μ⁻ → μ⁺μ⁻, p_T > 10 GeV, \|η\| < 2.5 | MadGraph5_aMC@NLO, LO |
| `thermalwimp.txt` | Thermal-relic higgsino- and wino-like WIMP pairs | Z. Liu and X. Wang, private communication |
| `incoherentpairs.txt` | Incoherent e⁺e⁻ pairs, p_T(e) > 15 MeV | Modified GuineaPig — see below |
| `incoherentpairsecal.txt` | Incoherent e⁺e⁻ pairs, p_T(e) > 1.4 GeV (reach the ECAL) | Modified GuineaPig — see below |
| `vbfqq.txt` | VBF qq̄ | provenance not recorded — [tell us](https://github.com/lawrenceleejr/MuonColliderRates/issues) if you know it |
| `collisionrate.txt` | Reference collision rate (not a cross section) | — |

Please cite the original calculations, not this repository.

## Incoherent pair production

`data/incoherentpairs.txt` is computed here rather than quoted. Incoherent
e⁺e⁻ pair production in the beam-beam interaction is simulated with a modified
GuineaPig that treats the beams as muons, run from the published container
image at
[`ghcr.io/lawrenceleejr/guineapig_mumu`](https://github.com/lawrenceleejr/guineapig_mumu).

For each simulated bunch crossing we count the pair leptons produced above a
`p_T` threshold and read off that crossing's luminosity, then quote the
effective cross section

```
sigma_eff = N(e±, p_T > p_T,min) / L_crossing
```

so that `sigma_eff × L_collider` is the *particle* rate entering the detector —
which is what makes the curve line up with the rate axis of the figure. Beam
parameters are the IMCC interim-report targets
([arXiv:2407.12450](https://arxiv.org/abs/2407.12450), Table 1.1, Scenario 1):
Stage 1 for 3 TeV, Stage 2 for 10 TeV.

Two thresholds are counted on the same crossings, so the two curves are
statistically consistent with each other:

* **15 MeV** — roughly what it takes to get out of the beam pipe at all.
* **1.4 GeV** — the minimum `p_T` for a particle to reach the ECAL surface.

| √s | p_T > 15 MeV | | p_T > 1.4 GeV | |
|----|--------------|--|---------------|--|
| | leptons/crossing | σ_eff | leptons/crossing | σ_eff |
| 3 TeV | ~220 | 5.5 × 10¹⁰ fb (~11 MHz) | ~0.04 | 1.0 × 10⁷ fb (~2 kHz) |
| 10 TeV | ~7100 | 2.5 × 10¹¹ fb (~50 MHz) | ~1.3 | 4.6 × 10⁷ fb (~9 kHz) |

The spectrum is steep: of the ~5.6 × 10⁵ pair leptons produced per crossing at
10 TeV, only about one has enough `p_T` to reach the calorimeter.

To reproduce (needs Docker; nothing else):

```bash
cd guineapig
./run_pairs.sh mumu10tev pairs10tev 24 runs/10tev 2
./run_pairs.sh mumu3tev  pairs3tev  24 runs/3tev  2
python summarize.py 3:runs/3tev 10:runs/10tev \
    --pt-min 0.015 --output ../data/incoherentpairs.txt
python summarize.py 3:runs/3tev 10:runs/10tev \
    --pt-min 1.4 --output ../data/incoherentpairsecal.txt
```

`acc.dat` holds the beam and simulation parameters; `run_pairs.sh` runs the
crossings (in parallel chains, carrying GuineaPig's random state forward so each
crossing is independent) and reduces each one to a single line per threshold;
`summarize.py` averages them. The runs are deterministic, so the commands above
reproduce the committed numbers exactly. The per-crossing summaries behind them
are kept in `guineapig/runs/`.

`PT_MINS` sets the thresholds. To pool extra statistics, run a second batch with
`SKIP_BASE` past the chain count of the first — otherwise the chains replay the
same crossings — and pass both directories for the same energy:

```bash
SKIP_BASE=2 ./run_pairs.sh mumu3tev pairs3tev 120 runs/3tev-more 4
python summarize.py 3:runs/3tev 3:runs/3tev-more 10:runs/10tev --pt-min 1.4 ...
```

### Caveats

The quoted uncertainty is the crossing-to-crossing standard error only.

* **`track_pairs=0`.** Pairs are recorded at production, so the beam-field
  deflection is not included. The no-FFTW build segfaults with tracking on, even
  with `grids=1`. This barely matters at 15 MeV but is the dominant systematic at
  1.4 GeV, since the deflection is exactly what gives a pair lepton a large
  transverse kick.
* **The 1.4 GeV tail is rare**, so it is the statistics-hungry number: the
  hardest lepton in a typical crossing is only a few hundred MeV.
* Grid resolution and the beam parameters themselves are not varied.

GuineaPig references: D. Schulte, PhD thesis, Univ. Hamburg, TESLA-97-08 (1997);
D. Schulte, "Beam-beam simulations with GUINEA-PIG", CERN-PS-99-014-LP,
CLIC-Note-387 (1999); C. Rimbault *et al.*, "GUINEA-PIG++", PAC'07, THPMN010
(2007).

## Contributing

### Adding a curve

1. Try it out first: open the
   [interactive page](https://lawrenceleejr.github.io/MuonColliderRates/), drop
   your file into *Add your own curve*, and check that it lands where you expect.
   Nothing you add there leaves your browser.
2. Press *Download as data file* — you get a copy with the standard header
   already in place.
3. Fill in `process:` and `source:`, drop the file in `data/`, and add an entry
   to `CURVES` in `mcrates.py`:

   ```python
   dict(
       key="vbfzh",
       dataset="vbfzh",
       label=r"VBF $ZH$",              # matplotlib
       html_label="VBF ZH",            # the web page
       group="sm",                     # background / sm / bsm / extra
       color="#4c72b0",
       dash="-",
       shown=True,                     # on by default in the interactive figure
   ),
   ```

4. If it should appear in the PDF, add the corresponding block to `plot.py`
   (the label positions there are hand-placed).
5. Run `python plot.py && python make_web_data.py` and commit the regenerated
   `web/data/curves.json` — CI checks that it matches `data/`.

### Anything else

Corrections to numbers, missing citations, better provenance for `vbfqq.txt`, or
just a suggestion for a process worth adding — please
[open an issue](https://github.com/lawrenceleejr/MuonColliderRates/issues).

## Publishing

The site is published from a workflow, not from a branch: no `gh-pages` branch
and no `docs/` folder are involved. `.github/workflows/pages.yml` renders the
figure, exports the data and deploys on every push to `main`.

The workflow's `configure-pages` step turns Pages on with the **GitHub Actions**
source the first time it runs, so there is nothing to set by hand. If that step
is ever removed and Pages has never been enabled, the deploy fails with
`Get Pages site failed` / a 404 until the source is set in
**Settings → Pages → Build and deployment**.
