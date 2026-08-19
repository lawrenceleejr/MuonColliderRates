#!/usr/bin/env python3
"""Turn run_pairs.sh output into data/incoherentpairs.txt.

For each simulated bunch crossing GuineaPig gives us

    N   the number of incoherent pair leptons produced above the p_T threshold
    L   the luminosity of that crossing, in m^-2 (GuineaPig is run with
        n_b = f_rep = 1, so lumi_ee is per crossing)

so sigma_eff = N / L is the cross section that reproduces the true particle
rate when it is multiplied by the collider's luminosity:

    rate = sigma_eff * L_collider = (N / L_crossing) * L_collider = N * f_crossing

which is what makes the curve line up with the rate axis of the figure.

run_pairs.sh counts at several p_T thresholds in one pass, so one set of runs
feeds several curves. --pt-min picks which of them this file is about.

Usage:
    python summarize.py 3:out/3tev 10:out/10tev \
        --pt-min 0.015 --output ../data/incoherentpairs.txt
    python summarize.py 3:out/3tev 10:out/10tev \
        --pt-min 1.4 --output ../data/incoherentpairsecal.txt
"""

import argparse
import math
import os
import sys

M2_PER_FB = 1e-43


def read_summary(path, pt_min):
    """Read summary.txt (or a directory containing one) -> list of crossings.

    The header names one count column per p_T threshold; `pt_min` selects one.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "summary.txt")
    columns = None
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                if ":" in line:
                    field, _, value = line.lstrip("#").strip().partition(":")
                    if field.strip().lower() == "columns":
                        columns = [c.strip() for c in value.split(",")]
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            rows.append(fields)
    if not rows:
        raise SystemExit("no crossings found in %s" % path)

    if columns is None:
        raise SystemExit("%s has no '# columns:' header; re-run run_pairs.sh" % path)
    wanted = "n_pt_%s" % _trim(pt_min)
    available = [c for c in columns if c.startswith("n_pt_")]
    if wanted not in columns:
        raise SystemExit(
            "%s has no column %r (thresholds available: %s)"
            % (path, wanted, ", ".join(c[len("n_pt_"):] for c in available) or "none"))
    icut = columns.index(wanted)
    itot, ilumi = columns.index("n_stored"), columns.index("lumi_m2")

    return [
        {
            "chain": f[0],
            "event": int(f[1]),
            "n_total": int(f[itot]),
            "n_cut": int(f[icut]),
            "lumi_m2": float(f[ilumi]),
        }
        for f in rows
    ]


def _trim(value):
    """Format a threshold the way run_pairs.sh wrote it into the header."""
    text = ("%g" % float(value)) if not isinstance(value, str) else value
    return text


def summarize(rows):
    sigmas = [r["n_cut"] / r["lumi_m2"] / M2_PER_FB for r in rows]
    n = len(sigmas)
    mean = sum(sigmas) / n
    if n > 1:
        var = sum((s - mean) ** 2 for s in sigmas) / (n - 1)
        err = math.sqrt(var / n)
    else:
        err = mean / math.sqrt(max(1.0, rows[0]["n_cut"]))
    return {
        "n_crossings": n,
        "sigma_fb": mean,
        "sigma_err_fb": err,
        "particles_per_crossing": sum(r["n_cut"] for r in rows) / n,
        "stored_per_crossing": sum(r["n_total"] for r in rows) / n,
        "lumi_m2": sum(r["lumi_m2"] for r in rows) / n,
    }


def pretty_threshold(pt_min):
    """0.015 -> '15 MeV', 1.4 -> '1.4 GeV'."""
    gev = float(pt_min)
    return "%g MeV" % (gev * 1000) if gev < 1 else "%g GeV" % gev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("points", nargs="+", metavar="SQRTS:DIR",
                    help="centre-of-mass energy in TeV and the run directory")
    ap.add_argument("--output", default=None, help="write a data/ text file here")
    ap.add_argument("--pt-min", default="0.015",
                    help="which threshold column to summarise, in GeV "
                         "(must match one used by run_pairs.sh)")
    ap.add_argument("--title", default="Incoherent e+e- pairs",
                    help="title written into the data file header")
    ap.add_argument("--note", default=None,
                    help="extra sentence appended to the notes header line")
    args = ap.parse_args(argv)

    # The same energy may be given more than once; batches are pooled, which is
    # how extra statistics get folded in (see SKIP_BASE in run_pairs.sh).
    crossings = {}
    for spec in args.points:
        if ":" not in spec:
            ap.error("expected SQRTS:DIR, got %r" % spec)
        energy, path = spec.split(":", 1)
        crossings.setdefault(float(energy), []).extend(read_summary(path, args.pt_min))

    results = []
    for energy in sorted(crossings):
        summary = summarize(crossings[energy])
        summary["sqrt_s"] = energy
        results.append(summary)

    pt_text = pretty_threshold(args.pt_min)
    for r in results:
        print("%5.1f TeV : %d crossings, %.3g leptons above %s per crossing, "
              "L = %.4g m^-2, sigma_eff = %.4g +- %.2g fb"
              % (r["sqrt_s"], r["n_crossings"], r["particles_per_crossing"],
                 pt_text, r["lumi_m2"], r["sigma_fb"], r["sigma_err_fb"]))

    if not args.output:
        return 0

    lines = [
        "# title: %s" % args.title,
        "# process: incoherent e+ e- pair production in the mu+ mu- beam-beam "
        "interaction, p_T(e) > %s" % pt_text,
        "# source: Modified GUINEA-PIG (muon beams), run from "
        "ghcr.io/lawrenceleejr/guineapig_mumu; private communication. "
        "GUINEA-PIG: D. Schulte, PhD thesis, Univ. Hamburg, TESLA-97-08 (1997); "
        "D. Schulte, 'Beam-beam simulations with GUINEA-PIG', CERN-PS-99-014-LP, "
        "CLIC-Note-387 (1999); C. Rimbault et al., 'GUINEA-PIG++', PAC'07, "
        "THPMN010 (2007)",
        "# notes: effective cross section sigma_eff = N(e+-, p_T > %s) / L_crossing, "
        "so that sigma_eff x L_collider is the true rate of pair leptons entering "
        "the detector. Beam parameters from arXiv:2407.12450 Table 1.1, Scenario 1 "
        "(Stage 1 at 3 TeV, Stage 2 at 10 TeV); see guineapig/ for the inputs and "
        "the runner. Averaged over %s. Uncertainty is the crossing-to-crossing "
        "standard error only. This file holds the effective cross section; the "
        "figure plots it against the rate axis instead, rescaled by "
        "L(sqrt_s)/L_nominal so each stage reads its own true rate (see "
        "mcrates.rate_equivalent)." % (
            pt_text,
            " and ".join("%d bunch crossings at %g TeV" % (r["n_crossings"], r["sqrt_s"])
                         for r in results))
        + (" " + args.note if args.note else ""),
        "# columns: sqrt_s [TeV], sigma [fb], stat_unc [fb]",
    ]
    for r in results:
        lines.append("%g, %.5g, %.3g" % (r["sqrt_s"], r["sigma_fb"], r["sigma_err_fb"]))

    with open(args.output, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
