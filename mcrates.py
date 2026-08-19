"""Shared loading and bookkeeping for the muon collider rate compilation.

Every file in ``data/`` is a plain comma-separated text file that carries its own
provenance in a small ``# key: value`` header, e.g.::

    # title: VBF Z
    # process: mu+ mu- -> Z nu nubar (vector-boson fusion)
    # source: https://arxiv.org/abs/2005.10289 (digitised)
    # columns: sqrt_s [TeV], sigma [fb]
    1.0144931295661115, 883.3407454482787
    ...

The ``columns:`` line is the only one that is parsed mechanically: each entry is
``name [unit]``.  Cross sections may be quoted in any of the units below and are
converted to femtobarns on load, so nothing downstream has to carry a magic
factor around.

``CURVES`` then says how each dataset is drawn -- colour, dash pattern, label --
and is the single source of truth shared by ``plot.py`` (the PDF figure) and
``make_web_data.py`` (the interactive page).
"""

import os
import re

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

#: Cross-section units understood in a ``# columns:`` header, in femtobarns.
UNIT_TO_FB = {
    "fb": 1.0,
    "pb": 1e3,
    "nb": 1e6,
    "ub": 1e9,
    "mb": 1e12,
    "b": 1e15,
}

_COLUMN_RE = re.compile(r"^\s*(?P<name>[^\[]+?)\s*\[\s*(?P<unit>[^\]]+?)\s*\]\s*$")

# Reference operating points used to turn a cross section into a rate.
# Target luminosities from the IMCC interim report, arXiv:2407.12450 Table 1.1,
# Scenario 1: Stage 1 (3 TeV) and Stage 2 (10 TeV). They differ by exactly a
# factor of ten.
NOMINAL_SQRTS_TEV = 10.0
NOMINAL_LUMI_CM2_S = 2.1e35
# Per-stage targets, for reference: multiplying a cross section by the entry for
# its own energy gives the rate that stage would actually see. The figure's rate
# axis uses the 10 TeV value throughout, as its label says.
STAGE_LUMI_CM2_S = {3.0: 2.1e34, 10.0: 2.1e35}
LUMI_SOURCE = "arXiv:2407.12450 Table 1.1, Scenario 1"


class Dataset:
    """One ``data/*.txt`` file: its header metadata plus unit-converted columns."""

    def __init__(self, key, meta, column_names, column_units, values):
        self.key = key
        self.meta = meta
        self.column_names = column_names
        self.column_units = column_units
        self.values = values  # raw, as written in the file

    # -- header fields ----------------------------------------------------
    @property
    def title(self):
        return self.meta.get("title", self.key)

    @property
    def process(self):
        return self.meta.get("process", "")

    @property
    def source(self):
        return self.meta.get("source", "")

    @property
    def notes(self):
        return self.meta.get("notes", "")

    # -- columns ----------------------------------------------------------
    @property
    def x(self):
        """Centre-of-mass energy in TeV (always the first column)."""
        return self.values[:, 0]

    def _index(self, column):
        if isinstance(column, int):
            return column
        try:
            return self.column_names.index(column)
        except ValueError:
            raise KeyError(
                "%s has no column %r (columns: %s)"
                % (self.key, column, ", ".join(self.column_names))
            )

    def column(self, column):
        """Return a column converted to fb if it is a cross section, else raw."""
        i = self._index(column)
        return self.values[:, i] * UNIT_TO_FB.get(self.column_units[i], 1.0)

    @property
    def y(self):
        """The default cross section column, in fb."""
        return self.column(1)

    def unit(self, column):
        return self.column_units[self._index(column)]

    def __repr__(self):
        return "<Dataset %s: %d rows, columns %s>" % (
            self.key,
            len(self.values),
            ", ".join(self.column_names),
        )


def load(key):
    """Load ``data/<key>.txt``."""
    path = os.path.join(DATA_DIR, key + ".txt")
    meta = {}
    rows = []
    with open(path) as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line.lstrip("#").strip()
                if ":" in body:
                    field, _, value = body.partition(":")
                    meta.setdefault(field.strip().lower(), value.strip())
                continue
            try:
                rows.append([float(field) for field in line.split(",")])
            except ValueError:
                raise ValueError("%s:%d: cannot parse %r" % (path, lineno, line))

    if "columns" not in meta:
        raise ValueError("%s: missing '# columns:' header line" % path)
    names, units = [], []
    for spec in meta["columns"].split(","):
        match = _COLUMN_RE.match(spec)
        if not match:
            raise ValueError("%s: column spec %r is not 'name [unit]'" % (path, spec))
        names.append(match.group("name"))
        units.append(match.group("unit"))

    values = np.array(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(
            "%s: header declares %d columns but the data has %s"
            % (path, len(names), values.shape)
        )
    return Dataset(key, meta, names, units, values)


def load_all(keys=None):
    if keys is None:
        keys = sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(DATA_DIR)
            if name.endswith(".txt")
        )
    return {key: load(key) for key in keys}


# ---------------------------------------------------------------------------
# Rate conversion
# ---------------------------------------------------------------------------


def fb_to_hz(cross_section_fb, lumi_cm2_s=NOMINAL_LUMI_CM2_S):
    """Cross section in fb -> interaction rate in Hz at the given luminosity."""
    return cross_section_fb * 1e-39 * lumi_cm2_s


def hz_to_fb(rate_hz, lumi_cm2_s=NOMINAL_LUMI_CM2_S):
    return rate_hz / lumi_cm2_s * 1e39


# ---------------------------------------------------------------------------
# How each curve is drawn
# ---------------------------------------------------------------------------

GREY = "grey"

#: Palette, in the order the signal curves are introduced in the figure.
PALETTE = [
    "#d62728",  # red
    "#ff7f0e",  # orange
    "#8c564b",  # brown
    "#2ca02c",  # green
    "#bcbd22",  # lime
    "#17becf",  # cyan
    "#17a398",  # teal
    "#1f77b4",  # blue
    "#9467bd",  # purple
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]

# group:   "background"  detector-occupancy / machine-induced processes (grey)
#          "sm"          Standard Model signal processes
#          "bsm"         beyond-the-Standard-Model benchmarks
#          "extra"       available in data/ but not drawn in the reference figure
#
# shown:   whether the curve is on by default in the interactive figure
#          (matches what appears in MuonColliderRates.pdf)
CURVES = [
    dict(
        key="lltohadrons",
        dataset="lltohadrons",
        label=r"Incl. $\mu\mu\to$Hadrons",
        html_label="Incl. &mu;&mu; &rarr; hadrons",
        group="background",
        color=GREY,
        dash=":",
        shown=True,
    ),
    dict(
        key="jj",
        dataset="jj",
        label=r"jj ($p_{T,j}>5-7 \text{ GeV}$, $|\eta_{j}|<3.13$)",
        html_label="jj (p<sub>T,j</sub> &gt; 5&ndash;7 GeV, |&eta;<sub>j</sub>| &lt; 3.13)",
        group="background",
        color=GREY,
        dash="--",
        shown=True,
    ),
    dict(
        key="incoherentpairs",
        dataset="incoherentpairs",
        label=r"Incoh. $ee$ ($p_{T,e}>15$ MeV)",
        html_label="Incoh. ee (p<sub>T,e</sub> &gt; 15 MeV)",
        group="background",
        color=GREY,
        dash="-.",
        marker="o",
        shown=True,
    ),
    dict(
        key="incoherentpairsecal",
        dataset="incoherentpairsecal",
        label=r"Incoh. $ee$ ($p_{T,e}>1.4$ GeV)",
        html_label="Incoh. ee (p<sub>T,e</sub> &gt; 1.4 GeV)",
        group="background",
        color=GREY,
        dash="-.",
        marker="o",
        shown=True,
    ),
    dict(
        key="vbfz",
        dataset="vbfz",
        label=r"VBF Z",
        html_label="VBF Z",
        group="sm",
        color=PALETTE[0],
        dash="-",
        shown=True,
    ),
    dict(
        key="vbfh",
        dataset="vbfh",
        label=r"VBF $H$",
        html_label="VBF H",
        group="sm",
        color=PALETTE[1],
        dash="-",
        shown=True,
    ),
    dict(
        key="mumu",
        dataset="mumu",
        label=r"$\mu\mu$ ($p_{T,\mu}>10$ GeV, $|\eta_{\mu}|<2.5$)",
        html_label="&mu;&mu; (p<sub>T,&mu;</sub> &gt; 10 GeV, |&eta;<sub>&mu;</sub>| &lt; 2.5)",
        group="sm",
        color=PALETTE[2],
        dash="-",
        shown=True,
    ),
    dict(
        key="vbfww",
        dataset="vbfww",
        label=r"VBF $WW$",
        html_label="VBF WW",
        group="sm",
        color=PALETTE[3],
        dash="-",
        shown=True,
    ),
    dict(
        key="vbftt",
        dataset="vbftt",
        label=r"VBF $t\bar{t}$",
        html_label="VBF t&#773;t",
        group="sm",
        color=PALETTE[4],
        dash="-",
        shown=True,
    ),
    dict(
        key="vbfhh",
        dataset="vbfhh",
        label=r"VBF $HH$",
        html_label="VBF HH",
        group="sm",
        color=PALETTE[5],
        dash="-",
        shown=True,
    ),
    dict(
        key="wimp_higgsino",
        dataset="thermalwimp",
        column="sigma_higgsino_charged_pair",
        label=r"Thermal $\tilde{H}$-like WIMP",
        html_label="Thermal H&#771;-like WIMP",
        group="bsm",
        color=PALETTE[6],
        dash="-",
        shown=True,
    ),
    dict(
        key="wimp_wino",
        dataset="thermalwimp",
        column="sigma_wino_charged_pair",
        label=r"Thermal $\tilde{W}$-like WIMP",
        html_label="Thermal W&#771;-like WIMP",
        group="bsm",
        color=PALETTE[7],
        dash="-",
        shown=True,
    ),
    dict(
        key="vbfwwz",
        dataset="vbfwwz",
        label=r"VBF $WWZ$",
        html_label="VBF WWZ",
        group="sm",
        color=PALETTE[8],
        dash="-",
        shown=True,
    ),
    dict(
        key="vbftth",
        dataset="vbftth",
        label=r"VBF $t\bar{t}H$",
        html_label="VBF t&#773;tH",
        group="sm",
        color=PALETTE[9],
        dash="-",
        shown=True,
    ),
    dict(
        key="vbfhhh",
        dataset="vbfhhh",
        label=r"VBF $HHH$",
        html_label="VBF HHH",
        group="sm",
        color=PALETTE[10],
        dash="-",
        shown=True,
    ),
    # -- present in data/, off by default in the interactive figure ---------
    dict(
        key="vbfqq",
        dataset="vbfqq",
        label=r"VBF $q\bar{q}$",
        html_label="VBF q&#773;q",
        group="extra",
        color="#4c72b0",
        dash="-",
        shown=False,
    ),
    dict(
        key="anntt",
        dataset="anntt",
        label=r"Ann. $t\bar{t}$",
        html_label="Annihilation t&#773;t",
        group="extra",
        color="#dd8452",
        dash="-",
        shown=False,
    ),
    dict(
        key="anntth",
        dataset="anntth",
        label=r"Ann. $t\bar{t}H$",
        html_label="Annihilation t&#773;tH",
        group="extra",
        color="#937860",
        dash="-",
        shown=False,
    ),
    dict(
        key="wimp_higgsino_neutral",
        dataset="thermalwimp",
        column="sigma_higgsino_neutral_pair",
        label=r"Thermal $\tilde{H}$-like WIMP (neutral pair)",
        html_label="Thermal H&#771;-like WIMP (neutral pair)",
        group="extra",
        color="#55a868",
        dash="-",
        shown=False,
    ),
    dict(
        key="wimp_higgsino_gamma",
        dataset="thermalwimp",
        column="sigma_higgsino_charged_pair_gamma",
        label=r"Thermal $\tilde{H}$-like WIMP $+\gamma$",
        html_label="Thermal H&#771;-like WIMP + &gamma;",
        group="extra",
        color="#8172b3",
        dash="-",
        shown=False,
    ),
    dict(
        key="wimp_wino_gamma",
        dataset="thermalwimp",
        column="sigma_wino_charged_pair_gamma",
        label=r"Thermal $\tilde{W}$-like WIMP $+\gamma$",
        html_label="Thermal W&#771;-like WIMP + &gamma;",
        group="extra",
        color="#c44e52",
        dash="-",
        shown=False,
    ),
]

CURVES_BY_KEY = {curve["key"]: curve for curve in CURVES}

#: Standalone points that are not part of any scan.
POINTS = [
    dict(
        key="neutrino_slice",
        label="Neutrino Slice Interaction",
        html_label="Neutrino slice interaction",
        group="background",
        color=GREY,
        x=10.0,
        # 30 kHz crossing rate x 0.44 x 0.21, from arXiv:2412.14115
        y=hz_to_fb(29979) * 0.44 * 0.21,
        source="https://arxiv.org/abs/2412.14115",
        shown=True,
    ),
]

#: Horizontal guide lines, quoted as rates in Hz at the nominal luminosity.
REFERENCE_LINES = [
    dict(key="lhc_bx", label="40 MHz Collision Rate (LHC)", rate_hz=40e6, muted=True),
    dict(key="lhc_l1", label="100 kHz L1 Trigger (LHC)", rate_hz=1e5, muted=True),
    dict(key="mc_bx", label="30 kHz Collision Rate (10 km)", rate_hz=29979, muted=False),
    dict(key="one_per_year", label="1 Event / Snowmass Year", rate_hz=1e-7, muted=False),
]
