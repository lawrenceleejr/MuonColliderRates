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

Usage:
    python summarize.py 3:out/3tev 10:out/10tev --output ../data/incoherentpairs.txt
"""

import argparse
import math
import os
import sys

M2_PER_FB = 1e-43


def read_summary(path):
    """Read summary.txt (or a directory containing one) -> list of crossings."""
    if os.path.isdir(path):
        path = os.path.join(path, "summary.txt")
    rows = []
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 5:
                continue
            rows.append({
                "chain": fields[0],
                "event": int(fields[1]),
                "n_total": int(fields[2]),
                "n_cut": int(fields[3]),
                "lumi_m2": float(fields[4]),
            })
    if not rows:
        raise SystemExit("no crossings found in %s" % path)
    return rows


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("points", nargs="+", metavar="SQRTS:DIR",
                    help="centre-of-mass energy in TeV and the run directory")
    ap.add_argument("--output", default=None, help="write a data/ text file here")
    ap.add_argument("--pt-min", default="15 MeV", help="threshold quoted in the header")
    args = ap.parse_args(argv)

    results = []
    for spec in args.points:
        if ":" not in spec:
            ap.error("expected SQRTS:DIR, got %r" % spec)
        energy, path = spec.split(":", 1)
        summary = summarize(read_summary(path))
        summary["sqrt_s"] = float(energy)
        results.append(summary)
    results.sort(key=lambda r: r["sqrt_s"])

    for r in results:
        print("%5.1f TeV : %d crossings, %.0f leptons above %s per crossing, "
              "L = %.4g m^-2, sigma_eff = %.4g +- %.2g fb"
              % (r["sqrt_s"], r["n_crossings"], r["particles_per_crossing"],
                 args.pt_min, r["lumi_m2"], r["sigma_fb"], r["sigma_err_fb"]))

    if not args.output:
        return 0

    lines = [
        "# title: Incoherent e+e- pairs",
        "# process: incoherent e+ e- pair production in the mu+ mu- beam-beam "
        "interaction, p_T(e) > %s" % args.pt_min,
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
        "standard error only." % (
            args.pt_min,
            " and ".join("%d bunch crossings at %g TeV" % (r["n_crossings"], r["sqrt_s"])
                         for r in results)),
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
