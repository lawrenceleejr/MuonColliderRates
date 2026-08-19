"""Build MuonColliderRates.pdf -- the reference cross-section/rate figure.

Data live in ``data/*.txt``; how each curve is drawn (colour, dash, label) lives
in ``mcrates.CURVES``.  Only the hand-tuned label placement is local to this
file.
"""

from matplotlib_tufte import *
setup()

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, ScalarFormatter
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.colors import to_rgba
import matplotlib.patheffects as pe
import matplotlib.transforms as mtrans
from matplotlib.transforms import Affine2D


from helperFunctions import *

import mcrates
from mcrates import CURVES_BY_KEY, fb_to_hz, hz_to_fb


# A thin white outline behind every in-plot label, so a curve passing behind
# text never fights it for legibility.
HALO = [pe.withStroke(linewidth=3.0, foreground="white")]


def mark_crossing(line, x_val, **marker_kwargs):
    x_data, y_data = line.get_data()
    # Find where x_data crosses x_val
    for i in range(len(x_data) - 1):
        if (x_data[i] - x_val) * (x_data[i+1] - x_val) <= 0:
            # Linear interpolation for y at x_val
            x0, x1 = x_data[i], x_data[i+1]
            y0, y1 = y_data[i], y_data[i+1]
            y_cross = y0 + (y1 - y0) * (x_val - x0) / (x1 - x0)
            ax.plot(x_val, y_cross, marker='o',clip_on=False, **marker_kwargs)
            break  # Only mark the first crossing

def print_crossing(line, x_val, **marker_kwargs):
    x_data, y_data = line.get_data()
    # Find where x_data crosses x_val
    for i in range(len(x_data) - 1):
        if (x_data[i] - x_val) * (x_data[i+1] - x_val) <= 0:
            # Linear interpolation for y at x_val
            x0, x1 = x_data[i], x_data[i+1]
            y0, y1 = y_data[i], y_data[i+1]
            y_cross = y0 + (y1 - y0) * (x_val - x0) / (x1 - x0)
            print(fb_to_hz(y_cross))
            break  # Only mark the first crossing


# ---------------------------------------------------------------------------
# Data.  Everything comes out of the loader already converted to fb.
# ---------------------------------------------------------------------------

datasets = {}


def curve(key):
    """(x [TeV], sigma [fb], colour) for a curve declared in mcrates.CURVES."""
    spec = CURVES_BY_KEY[key]
    name = spec["dataset"]
    if name not in datasets:
        datasets[name] = mcrates.load(name)
    dataset = datasets[name]
    return dataset.x, dataset.column(spec.get("column", 1)), spec["color"]


baselength=4
fig, ax = plt.subplots(1,1, figsize=(1.5*baselength, 2*baselength))


# Add manually scaled Y axis on the right
ax2 = ax.secondary_yaxis('right', functions=(fb_to_hz,hz_to_fb))
ax2.set_ylabel(r'Rate (at $\sqrt{s}=$10 TeV, L=$2\times10^{35}$ cm$^{-2}$ s$^{-1}$) [Hz]', color='black')
ax2.set_yscale('log',base=10)
# ax2.spines.right.set_position(('data', 20))
ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=None))
ax2.tick_params(axis='y', labelcolor='black')
# ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))


ax.annotate(
    '40 MHz Collision Rate (LHC)',       # Text
    xy=(10, hz_to_fb(40000000)),                 # Point to annotate
    xytext=(8, hz_to_fb(40000000)),           # Position of the text (to the left)
    arrowprops=dict(arrowstyle='-|>',color="grey"), va="center", ha="right",
    color="grey", path_effects=HALO
)

ax.annotate(
    '100 kHz L1 Trigger (LHC)',       # Text
    xy=(10, hz_to_fb(100000)),                 # Point to annotate
    xytext=(8, hz_to_fb(100000)),           # Position of the text (to the left)
    arrowprops=dict(arrowstyle='-|>',color="grey"), va="center", ha="right",
    color="grey", path_effects=HALO
)





#ax.annotate(
#    '11 kHz Collision Rate (27 km)',       # Text
#    xy=(10, hz_to_fb(11245)),                 # Point to annotate
#    xytext=(8, hz_to_fb(11245)),           # Position of the text (to the left)
#    arrowprops=dict(arrowstyle='-|>'), va="center", ha="right"
#)

ax.annotate(
    '30 kHz Collision Rate (10 km)',       # Text
    xy=(10, hz_to_fb(29979)),                 # Point to annotate
    xytext=(8, hz_to_fb(29979)),           # Position of the text (to the left)
    arrowprops=dict(arrowstyle='-|>'), va="center", ha="right", path_effects=HALO
)


ax.annotate(
    '1 Event / Snowmass Year',       # Text
    xy=(10, hz_to_fb(1e-7)),                 # Point to annotate
    xytext=(8, hz_to_fb(1e-7)),           # Position of the text (to the left)
    arrowprops=dict(arrowstyle='-|>'), va="center", ha="right", path_effects=HALO
)

# ax.text(10, 1e6, r'$\{$',
#         fontsize=120,
#         ha='right', va='center',
#         fontfamily='serif')  # Try 'monospace' or 'sans-serif' too


### Actual Curves:


### Machine-induced and inclusive backgrounds, all in grey

x, y, color = curve("lltohadrons")
line, = ax.plot(x, y, ":", color=color, lw=1, alpha=0.5)
ax.text( 1.2, 0.12*y[69],
    CURVES_BY_KEY["lltohadrons"]["label"],
    color=color, fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='left'
)
mark_crossing(line, 10, color=color)

x, y, color = curve("jj")
line, = ax.plot(x, y, "--", color=color, lw=1, alpha=0.5)
ax.text( 0.95*10, 1.3*y[69],
    CURVES_BY_KEY["jj"]["label"],
    color=color, fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=color)


x, y, color = curve("incoherentpairs")
line, = ax.plot(x, y, "-.", marker='o', markersize=3, color=color, lw=1, alpha=0.5)
# Sits just under its own curve: the band above belongs to the 40 MHz guide.
ax.text( 0.95*10, 0.28*y[-1],
    CURVES_BY_KEY["incoherentpairs"]["label"],
    color=color, fontsize=8.5, path_effects=HALO, verticalalignment='top',horizontalalignment='right'
)
mark_crossing(line, 10, color=color)



# ax.annotate(
#     'Beam-Induced Neutrino Interaction Rate',       # Text
#     xy=(10, hz_to_fb(29979)*0.44*0.21),                 # Point to annotate
#     xytext=(8, hz_to_fb(29979)*0.44*0.21),           # Position of the text (to the left)
#     arrowprops=dict(arrowstyle='-|>'), va="center", ha="right"
# )
# ax.text(8, 0.5*hz_to_fb(29979)*0.44*0.21, r'[2412.14115]',
#         fontsize=9,
#         ha='right', va='center',
#         fontfamily='serif')  # Try 'monospace' or 'sans-serif' too


ax.text( 0.95*10, 1.12*hz_to_fb(29979)*0.44*0.21,
    "Neutrino Slice Interaction\n[2412.14115]",
    color="grey", fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='right'
)
ax.plot(10, hz_to_fb(29979)*0.44*0.21, marker='o',clip_on=False, color="grey")




### Standard Model and BSM benchmark processes

alpha=1

x, y, color = curve("vbfz")
line, = ax.plot(x, y, "-", color=to_rgba(color,alpha), lw=1)
ax.text( 0.95*10, 1.05*y[69],
    CURVES_BY_KEY["vbfz"]["label"],
    color=to_rgba(color,alpha), fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=to_rgba(color,alpha))
print("VBF Z")
print_crossing(line, 10, color=to_rgba(color,alpha))


# x, y, color = curve("vbfqq")
# line, = ax.plot(x, y, "-", color=to_rgba(color,alpha), lw=1)
# ax.text( 0.95*10, 1.05*y[3],
#     CURVES_BY_KEY["vbfqq"]["label"],
#     color=to_rgba(color,alpha), fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='right'
# )
# mark_crossing(line, 10, color=to_rgba(color,alpha))




x, y, color = curve("vbfh")
line, = ax.plot(x, y, "-", color=to_rgba(color,alpha), lw=1)
ax.text( 0.95*10, 1.05*y[65],
    CURVES_BY_KEY["vbfh"]["label"],
    color=to_rgba(color,alpha), fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=to_rgba(color,alpha))

print("VBF H")
print_crossing(line, 10, color=to_rgba(color,alpha))




x, y, color = curve("mumu")
line, = ax.plot(x, y, "-", color=to_rgba(color,alpha), lw=1)
ax.text( 2.2, 1.0*10000,
    CURVES_BY_KEY["mumu"]["label"],
    color=to_rgba(color,alpha), fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='left'
)
mark_crossing(line, 10, color=to_rgba(color,alpha))




x, y, color = curve("vbfww")
line, = ax.plot(x, y, "-", color=to_rgba(color,alpha), lw=1)
ax.text( 2, 30,
    CURVES_BY_KEY["vbfww"]["label"],
    color=to_rgba(color,alpha), fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='left'
)
mark_crossing(line, 10, color=to_rgba(color,alpha))




x, y, color = curve("vbftt")
line, = ax.plot(x, y, "-", color=to_rgba(color,alpha), lw=1)
ax.text( 1.2, 2,
    CURVES_BY_KEY["vbftt"]["label"],
    color=to_rgba(color,alpha), fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='left'
)
mark_crossing(line, 10, color=to_rgba(color,alpha))


x, y, color = curve("vbfhh")
line, = ax.plot(x, y, "-", color=to_rgba(color,alpha), lw=1)
ax.text( 0.95*10, 1.05*y[66],
    CURVES_BY_KEY["vbfhh"]["label"],
    color=to_rgba(color,alpha), fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=to_rgba(color,alpha))

print("VBF HH")
print_crossing(line, 10, color=to_rgba(color,alpha))


# https://arxiv.org/pdf/2102.11292
x, y, color = curve("wimp_higgsino")
line, = ax.plot(x, y, "-", color=to_rgba(color,alpha), lw=1)
ax.text( 2.25, 3e0,
    CURVES_BY_KEY["wimp_higgsino"]["label"],
    color=to_rgba(color,alpha), fontsize=10, path_effects=HALO, verticalalignment='top',horizontalalignment='right',rotation=90
)
mark_crossing(line, 10, color=to_rgba(color,alpha))


x, y, color = curve("wimp_wino")
line, = ax.plot(x, y, "-", color=to_rgba(color,alpha), lw=1)
ax.text( 5.7, 3e0,
    CURVES_BY_KEY["wimp_wino"]["label"],
    color=to_rgba(color,alpha), fontsize=10, path_effects=HALO, verticalalignment='top',horizontalalignment='right', rotation=90
)
mark_crossing(line, 10, color=to_rgba(color,alpha))









x, y, color = curve("vbfwwz")
line, = ax.plot(x, y, "-", color=to_rgba(color,alpha), lw=1)
ax.text( 0.95*10, 1.05*y[64],
    CURVES_BY_KEY["vbfwwz"]["label"],
    color=to_rgba(color,alpha), fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=to_rgba(color,alpha))




x, y, color = curve("vbftth")
line, = ax.plot(x, y, "-", color=to_rgba(color,alpha), lw=1)
ax.text( 0.95*10, 1.05*y[75],
    CURVES_BY_KEY["vbftth"]["label"],
    color=to_rgba(color,alpha), fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=to_rgba(color,alpha))



x, y, color = curve("vbfhhh")
line, = ax.plot(x, y, "-", color=to_rgba(color,alpha), lw=1)
ax.text( 0.95*10, 1.05*y[73],
    CURVES_BY_KEY["vbfhhh"]["label"],
    color=to_rgba(color,alpha), fontsize=10, path_effects=HALO, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=to_rgba(color,alpha))








# Title block above the axes rather than over the data, so the whole interior
# is left to the curves and their labels.  matplotlib_tufte turns on
# figure.autolayout, which ignores subplots_adjust but does reserve room for a
# title -- hence set_title with a pad large enough to clear the source line.

ax.text(0.0, 1.012,
    r"$\sigma$ from 2005.10289; 2103.09844; Z. Liu, X. Wang;"
    + "\nModified GUINEA-PIG; and MadGraph5_aMC@NLO",
    transform=ax.transAxes, color="0.35", fontsize=9,
    verticalalignment='bottom', horizontalalignment='left', linespacing=1.4
)
ax.set_title(r"Muon Collider Rates", loc='left', fontsize=21, color="k", pad=40)



ax.set_xlabel(r'$\sqrt{s}$ [TeV]',)
ax.set_ylabel(r'$\sigma$ [fb]',)
ax.set_xscale('log',base=10)
ax.set_yscale('log',base=10)
ax.set_ylim([1e-4,1e12])
ax.set_xlim([1,10])

# Use FormatStrFormatter for clean labels
formatter = FormatStrFormatter('%g')
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_minor_formatter(formatter)
# Optional: control tick placement
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))

# ax.tick_params(axis='x', which='both', pad=8)

ax.spines['top'].set_visible(False)


tick_padding = 7  # choose a value that looks good to you
ax.tick_params(axis='x', which='major', length=5, pad=tick_padding-5)
ax.tick_params(axis='x', which='minor', length=2, pad=tick_padding-2)

breathe_logxy(ax)


fig.text(0.97, 0.03, 'L. Lee, T. Holmes', ha='right', va='top', fontsize=10)


# Force figure to render, so transforms are accurate
fig.subplots_adjust(left=0.18, right=0.85, bottom=0.08, top=0.96)
fig.canvas.draw()

fig.savefig("MuonColliderRates.pdf")
fig.savefig("MuonColliderRates.png", dpi=200)
