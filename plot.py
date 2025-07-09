from matplotlib_tufte import *
setup()

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.colors import to_rgba
import matplotlib.transforms as mtrans
from matplotlib.transforms import Affine2D


import sys
# sys.path.insert(0, "..")
from helperFunctions import *

# plt.subplots_adjust(wspace=0.03)
def fb_to_hz(cross_section_fb, lumi_cm2_s=2e35):
    # Convert fb to cm^2
    sigma_cm2 = cross_section_fb * 1e-39
    # Rate in Hz
    return sigma_cm2 * lumi_cm2_s
def hz_to_fb(rate_hz, lumi_cm2_s=2e35):
    return rate_hz / lumi_cm2_s * 1e39



colors = ["#FF595E",  "#1982C4", "#8AC926", "#F2CC8F", "#1982C4", "#1982C4"] 
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

data["vbfqq"] = np.genfromtxt("data/vbfqq.txt", delimiter=",", skip_header=0, names=["x","y"])
data["vbfz"] = np.genfromtxt("data/vbfz.txt", delimiter=",", skip_header=1, names=["x","y"])
data["vbftt"] = np.genfromtxt("data/vbftt.txt", delimiter=",", skip_header=1, names=["x","y"])
data["vbftth"] = np.genfromtxt("data/vbftth.txt", delimiter=",", skip_header=1, names=["x","y"])
data["vbfh"] = np.genfromtxt("data/vbfh.txt", delimiter=",", skip_header=1, names=["x","y"])
data["vbfhh"] = np.genfromtxt("data/vbfhh.txt", delimiter=",", skip_header=1, names=["x","y"])
data["vbfhhh"] = np.genfromtxt("data/vbfhhh.txt", delimiter=",", skip_header=1, names=["x","y"])


baselength=4
fig, ax = plt.subplots(1,1, figsize=(1.5*baselength, 2.5*baselength))

# Add manually scaled Y axis on the right
ax2 = ax.secondary_yaxis('right', functions=(fb_to_hz,hz_to_fb))
ax2.set_ylabel(r'Rate (at L=$2\times10^{35}$ cm$^{-2}$ s$^{-1}$) [Hz]', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_yscale('log',base=10)


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


i=0
alpha=1

ax.plot(data["vbfqq"]['x'], (data["vbfqq"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbfqq"]['y'][3], 
    r"VBF $q\bar{q}$", 
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)


i=i+1
ax.plot(data["vbfz"]['x'], (data["vbfz"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbfz"]['y'][69], 
    r"VBF Z", 
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)


i=i+1
ax.plot(data["vbftt"]['x'], (data["vbftt"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbftt"]['y'][71], 
    r"VBF $t\bar{t}$", 
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)

i=i+1
ax.plot(data["vbftth"]['x'], (data["vbftth"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbftth"]['y'][75], 
    r"VBF $t\bar{t}H$", 
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)

i=i+1
ax.plot(data["vbfh"]['x'], (data["vbfh"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbfh"]['y'][65], 
    r"VBF $H$", 
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)


i=i
ax.plot(data["vbfhh"]['x'], (data["vbfhh"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbfhh"]['y'][66], 
    r"VBF $HH$", 
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)

i=i
ax.plot(data["vbfhhh"]['x'], (data["vbfhhh"]['y']),"-", color=to_rgba(colors[i],alpha), lw=1)
ax.text( 0.95*10, 1.05*data["vbfhhh"]['y'][73], 
    r"VBF $HHH$", 
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)










ax.set_xlabel(r'$\sqrt{s}$ [TeV]',)
ax.set_ylabel(r'$\sigma$ [fb]',)
ax.set_xscale('log',base=10)
ax.set_yscale('log',base=10)
ax.set_ylim([1e-4,1e12])
ax.set_xlim([1,10])



ax.spines['top'].set_visible(False)


breathe_logxy(ax)


# Force figure to render, so transforms are accurate
fig.subplots_adjust(left=0.15, right=0.93, bottom=0.18, top=0.96)
fig.canvas.draw()

fig.savefig("MuonColliderRates.pdf")
# plt.show()