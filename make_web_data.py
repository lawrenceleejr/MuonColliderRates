#!/usr/bin/env python3
"""Export data/ into web/data/curves.json for the interactive figure.

Run this after adding or editing anything under data/::

    python make_web_data.py

The CI workflow runs it too, so the published page never drifts from the
repository contents.
"""

import json
import os

import mcrates

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web", "data", "curves.json")


def build():
    datasets = {}

    def dataset(name):
        if name not in datasets:
            datasets[name] = mcrates.load(name)
        return datasets[name]

    curves = []
    for spec in mcrates.CURVES:
        data = dataset(spec["dataset"])
        column = spec.get("column", 1)
        x = data.x
        y = data.column(column)
        if spec.get("rate_equivalent"):
            # Really a rate: rescaled so the page's single rate axis reads the
            # true value at each stage. See mcrates.rate_equivalent.
            y = mcrates.rate_equivalent(x, y)
        points = [
            [float(xi), float(yi)]
            for xi, yi in zip(x, y)
            if yi > 0 and xi > 0
        ]
        if not points:
            continue
        entry = dict(
            key=spec["key"],
            label=spec["html_label"],
            latex=spec["label"],
            group=spec["group"],
            color=spec["color"],
            dash=spec["dash"],
            shown=spec["shown"],
            marker=spec.get("marker"),
            process=data.process,
            source=data.source,
            notes=data.notes,
            file="data/%s.txt" % spec["dataset"],
            points=points,
        )
        curves.append(entry)

    points = [
        dict(
            key=p["key"],
            label=p["html_label"],
            latex=p["label"],
            group=p["group"],
            color=p["color"],
            shown=p["shown"],
            source=p.get("source", ""),
            notes=p.get("notes", ""),
            points=[[p["x"], p["y"]]],
            marker="o",
            dash=None,
            process=p.get("process", ""),
            file=p.get("file", ""),
        )
        for p in mcrates.POINTS
    ]

    # Deliberately free of build stamps: this file must be a pure function of
    # data/ and mcrates.py so CI can check it against the repository contents.
    # The commit/date shown on the page comes from data/build.json, which the
    # Pages workflow writes at deploy time.
    return dict(
        repo="https://github.com/lawrenceleejr/MuonColliderRates",
        nominal_lumi_cm2_s=mcrates.NOMINAL_LUMI_CM2_S,
        nominal_sqrts_tev=mcrates.NOMINAL_SQRTS_TEV,
        reference_lines=mcrates.REFERENCE_LINES,
        curves=curves + points,
    )


if __name__ == "__main__":
    payload = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))
        handle.write("\n")
    n = len(payload["curves"])
    size = os.path.getsize(OUT) / 1024.0
    print("wrote %s (%d series, %.1f kB)" % (os.path.relpath(OUT, HERE), n, size))
