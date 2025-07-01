from matplotlib_tufte import *
setup()

import matplotlib.font_manager as fm
fm.fontManager.addfont("MyriadPro-Regular.ttf")
fm.fontManager.addfont("MyriadPro-Bold.ttf")
from matplotlib import rcParams
rcParams['font.family'] = 'Myriad Pro'

import matplotlib.pyplot as plt
import numpy as np

import ROOT

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.colors import to_rgba

import sys
sys.path.insert(0, "..")
from helperFunctions import *

# plt.subplots_adjust(wspace=0.03)
def fb_to_hz(cross_section_fb, lumi_cm2_s=1e34):
    # Convert fb to cm^2
    sigma_cm2 = cross_section_fb * 1e-39
    # Rate in Hz
    return sigma_cm2 * lumi_cm2_s
def hz_to_fb(rate_hz, lumi_cm2_s=1e34):
    return rate_hz / lumi_cm2_s * 1e39



colors = ["#FF595E",  "#1982C4", "#8AC926", "#F2CC8F"] 
# colors = ["#E07A5F",  # Terra Cotta
# 		"#F2CC8F",  # Sand
# 		"#81B29A"]  # Sage

# colors = ["#FF0000", "#00FF00", "#0000FF"]
# colors = ["#FF6B6B", "#6BCB77", "#4D96FF"]
# colors = ["#3A86FF", "#8338EC", "#FB5607"]
# colors = ["#FF6B6B", "#4ECDC4", "#1A535C"]
# colors = ["#4477AA", "#CC6677", "#117733"]

data = {}

data["collisionrate"] = np.genfromtxt("data/collisionrate.txt", delimiter=",", skip_header=0, names=["x","y"])

data["qq"] = np.genfromtxt("data/qq.txt", delimiter=",", skip_header=0, names=["x","y"])



baselength=4
fig, ax = plt.subplots(1,1, figsize=(1.5*baselength, 2*baselength))

# Add manually scaled Y axis on the right
ax2 = ax.secondary_yaxis('right', functions=(fb_to_hz,hz_to_fb))
ax2.set_ylabel('Rate [Hz]', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_yscale('log',base=10)

# Put several collision rate curves for different ring assumptions.
ax.plot(data["collisionrate"]['x'], hz_to_fb(data["collisionrate"]['y']),":", color="black")

ax.text( 0.95*10, 1.05*hz_to_fb(data["collisionrate"]['y'])[-1], 
    "Collision Rate", 
    color="black", fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)

### Actual Curves:


i=0
alpha=1

ax.plot(data["qq"]['x'], (data["qq"]['y']),"--", color=to_rgba(colors[i],alpha), mew=1)

ax.text( 0.95*10, 1.05*data["qq"]['y'][-1], 
    r"$q\bar{q}$", 
    color=to_rgba(colors[i],alpha), fontsize=10, verticalalignment='bottom',horizontalalignment='right'
)


ax.set_xlabel(r'$\sqrt{s}$ [TeV]',)
ax.set_ylabel(r'$\sigma$ [fb]',)
ax.set_xscale('log',base=10)
ax.set_yscale('log',base=10)
ax.set_ylim([1e2,2e10])
ax.set_xlim([1,10])



ax.spines['top'].set_visible(False)


breathe_logxy(ax)


# Force figure to render, so transforms are accurate
fig.subplots_adjust(left=0.15, right=0.93, bottom=0.18, top=0.96)
fig.canvas.draw()

fig.savefig("MuonColliderRates.pdf")
# plt.show()