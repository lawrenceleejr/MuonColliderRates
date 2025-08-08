from matplotlib_tufte import *
setup()

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, ScalarFormatter
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.colors import to_rgba
import matplotlib.transforms as mtrans
from matplotlib.transforms import Affine2D


import sys
import ast
sys.path.insert(0, "data")
from helperFunctions import *
from thermalwimp import tmpdata as thermalwimp_data

# plt.subplots_adjust(wspace=0.03)
def fb_to_hz(cross_section_fb, lumi_cm2_s=2e35):
    # Convert fb to cm^2
    sigma_cm2 = cross_section_fb * 1e-39
    # Rate in Hz
    return sigma_cm2 * lumi_cm2_s
def hz_to_fb(rate_hz, lumi_cm2_s=2e35):
    return rate_hz / lumi_cm2_s * 1e39

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


# colors = ["#FF595E",  "#1982C4", "#8AC926", "#F2CC8F", "#1982C4", "#1982C4"]

colors = [
    "#d62728",  # red
    "#ff7f0e",  # orange
    "#8c564b",  # brown
    "#2ca02c",  # green
    "#bcbd22",  # lime (new)
    "#17becf",  # cyan
    "#17a398",  # teal (new)
    "#1f77b4",  # blue
    "#9467bd",  # purple
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]
# colors = ["#E07A5F",  # Terra Cotta
# 		"#F2CC8F",  # Sand
# 		"#81B29A"]  # Sage

# colors = ["#FF0000", "#00FF00", "#0000FF"]
# colors = ["#FF6B6B", "#6BCB77", "#4D96FF"]
# colors = ["#3A86FF", "#8338EC", "#FB5607"]
# colors = ["#FF6B6B", "#4ECDC4", "#1A535C"]
# colors = ["#4477AA", "#CC6677", "#117733"]

data = {}

# data["collisionrate"] = np.genfromtxt("data/collisionrate.txt", delimiter=",", skip_header=0, names=["x","y"])

data["mumu"] = np.genfromtxt("data/mumu.txt", delimiter=",", skip_header=1, names=["x","y","sigma"])
data["vbfqq"] = np.genfromtxt("data/vbfqq.txt", delimiter=",", skip_header=0, names=["x","y"])
data["vbfz"] = np.genfromtxt("data/vbfz.txt", delimiter=",", skip_header=1, names=["x","y"])
data["vbfww"] = np.genfromtxt("data/vbfww.txt", delimiter=",", skip_header=1, names=["x","y"])
data["vbfwwz"] = np.genfromtxt("data/vbfwwz.txt", delimiter=",", skip_header=1, names=["x","y"])
data["vbftt"] = np.genfromtxt("data/vbftt.txt", delimiter=",", skip_header=1, names=["x","y"])
data["vbftth"] = np.genfromtxt("data/vbftth.txt", delimiter=",", skip_header=1, names=["x","y"])
data["vbfh"] = np.genfromtxt("data/vbfh.txt", delimiter=",", skip_header=1, names=["x","y"])
data["vbfhh"] = np.genfromtxt("data/vbfhh.txt", delimiter=",", skip_header=1, names=["x","y"])
data["vbfhhh"] = np.genfromtxt("data/vbfhhh.txt", delimiter=",", skip_header=1, names=["x","y"])

data["thermalwimp"] = np.array(thermalwimp_data)

data["jj"] = np.genfromtxt("data/jj.txt", delimiter=",", skip_header=1, names=["x","y"])
data["lltohadrons"] = np.genfromtxt("data/lltohadrons.txt", delimiter=",", skip_header=1, names=["x","y"])


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
    color="grey"
)

ax.annotate(
    '100 kHz L1 Trigger (LHC)',       # Text
    xy=(10, hz_to_fb(100000)),                 # Point to annotate
    xytext=(8, hz_to_fb(100000)),           # Position of the text (to the left)
    arrowprops=dict(arrowstyle='-|>',color="grey"), va="center", ha="right",
    color="grey"
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
    arrowprops=dict(arrowstyle='-|>'), va="center", ha="right"
)


ax.annotate(
    '1 Event / Snowmass Year',       # Text
    xy=(10, hz_to_fb(1e-7)),                 # Point to annotate
    xytext=(8, hz_to_fb(1e-7)),           # Position of the text (to the left)
    arrowprops=dict(arrowstyle='-|>'), va="center", ha="right"
)

# ax.text(10, 1e6, r'$\{$',
#         fontsize=120,
#         ha='right', va='center',
#         fontfamily='serif')  # Try 'monospace' or 'sans-serif' too


### Actual Curves:



line, = ax.plot(data["lltohadrons"]['x'], (data["lltohadrons"]['y']*1e6),":", color="grey", lw=1, alpha=0.5)
ax.text( 1.2, 0.12*data["lltohadrons"]['y'][69]*1e6,
    r"Incl. $\mu\mu\to$Hadrons",
    color="grey", fontsize=10, verticalalignment='bottom',horizontalalignment='left'
)
mark_crossing(line, 10, color="grey")

line, = ax.plot(data["jj"]['x'], (data["jj"]['y']*1e3),"--", color="grey", lw=1, alpha=0.5)
ax.text( 0.95*10, 1.3*data["jj"]['y'][69]*1e3,
    r"jj ($p_{T,j}>5-7 \text{ GeV}$, $|\eta_{j}|<3.13$)",
    color="grey", fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color="grey")



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


ax.text( 0.95*10, hz_to_fb(29979)*0.44*0.21,
    "Neutrino Slice Interaction\n[2412.14115]",
    color="grey", fontsize=10, verticalalignment='top',horizontalalignment='right'
)
ax.plot(10, hz_to_fb(29979)*0.44*0.21, marker='o',clip_on=False, color="grey")





i=0
alpha=1

line, = ax.plot(data["vbfz"]['x'], (data["vbfz"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbfz"]['y'][69],
    r"VBF Z",
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=to_rgba(colors[i],alpha))
print("VBF Z")
print_crossing(line, 10, color=to_rgba(colors[i],alpha))


# i=i+1
# line, = ax.plot(data["vbfqq"]['x'], (data["vbfqq"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
# ax.text( 0.95*10, 1.05*data["vbfqq"]['y'][3],
#     r"VBF $q\bar{q}$",
#     color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
# )
# mark_crossing(line, 10, color=to_rgba(colors[i],alpha))




i=i+1
line, = ax.plot(data["vbfh"]['x'], (data["vbfh"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbfh"]['y'][65],
    r"VBF $H$",
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=to_rgba(colors[i],alpha))

print("VBF H")
print_crossing(line, 10, color=to_rgba(colors[i],alpha))




i=i+1
line, = ax.plot(data["mumu"]['x'], (1000*data["mumu"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 2.2, 1.0*10000,
    r"$\mu\mu$ ($p_{T,\mu}>10$ GeV, $|\eta_{\mu}|<2.5$)",
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='left'
)
mark_crossing(line, 10, color=to_rgba(colors[i],alpha))




i=i+1
line, = ax.plot(data["vbfww"]['x'], (data["vbfww"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 2, 30,
    r"VBF $WW$",
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='left'
)
mark_crossing(line, 10, color=to_rgba(colors[i],alpha))




i=i+1
line, = ax.plot(data["vbftt"]['x'], (data["vbftt"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 1.2, 2,
    r"VBF $t\bar{t}$",
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='left'
)
mark_crossing(line, 10, color=to_rgba(colors[i],alpha))


i=i+1
line, = ax.plot(data["vbfhh"]['x'], (data["vbfhh"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbfhh"]['y'][66],
    r"VBF $HH$",
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=to_rgba(colors[i],alpha))

print("VBF HH")
print_crossing(line, 10, color=to_rgba(colors[i],alpha))


# https://arxiv.org/pdf/2102.11292
# i=i+1
# ax.text( 0.95*10, 0.65*(2.2+0.039),
#     r"Ther. $\tilde{W}$ WIMP",
#     color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='center',horizontalalignment='right'
# )
# ax.plot(10, 2.2+0.039, marker='o',clip_on=False, color=to_rgba(colors[i],alpha))


# i=i+1
# ax.text( 0.95*10, 0.65*1.18436,
#     r"Ther. $\tilde{H}$ WIMP",
#     color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='center',horizontalalignment='right'
# )
# ax.plot(10, 1.18436, marker='o',clip_on=False, color=to_rgba(colors[i],alpha))


i=i+1
line, = ax.plot(data["thermalwimp"][0], (data["thermalwimp"][1]*1000.),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 2.25, 3e0,
    r"Thermal $\tilde{H}$-like WIMP",
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='top',horizontalalignment='right',rotation=90
)
mark_crossing(line, 10, color=to_rgba(colors[i],alpha))


i=i+1
line, = ax.plot(data["thermalwimp"][0], (data["thermalwimp"][3]*1000.),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 5.7, 3e0,
    r"Thermal $\tilde{W}$-like WIMP",
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='top',horizontalalignment='right', rotation=90
)
mark_crossing(line, 10, color=to_rgba(colors[i],alpha))









i=i+1
line, = ax.plot(data["vbfwwz"]['x'], (data["vbfwwz"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbfwwz"]['y'][64],
    r"VBF $WWZ$",
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=to_rgba(colors[i],alpha))




i=i+1
line, = ax.plot(data["vbftth"]['x'], (data["vbftth"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbftth"]['y'][75],
    r"VBF $t\bar{t}H$",
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=to_rgba(colors[i],alpha))



i=i+1
line, = ax.plot(data["vbfhhh"]['x'], (data["vbfhhh"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbfhhh"]['y'][73],
    r"VBF $HHH$",
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)
mark_crossing(line, 10, color=to_rgba(colors[i],alpha))








# Labels

ax.text( 1, 1e10,
    r"Muon Collider Rates",
    color="k", fontsize=22, verticalalignment='bottom',horizontalalignment='left'
)
ax.text( 1, 0.8e10,
    r"$\sigma$ from 2005.10289; 2103.09844; Z. Liu, X. Wang;"+ "\nand MadGraph5_aMC@NLO",
    color="k", fontsize=10, verticalalignment='top',horizontalalignment='left'
)





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

# fig.savefig("MuonColliderRates.pdf")
# plt.show()



fig.savefig("MuonColliderRates.pdf")
