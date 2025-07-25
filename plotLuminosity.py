from matplotlib_tufte import *
setup()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L_avg = 1e34              # Desired average luminosity [cm^-2 s^-1]
tau_mu = 0.104            # Muon lab-frame lifetime [s]
refill_interval = 0.2     # Seconds between refills
total_time = 1.0          # Simulate for 1 second
dt = 1e-3                 # Time step [s]

# Compute required peak luminosity for given average
decay_factor = 1 - np.exp(-2 * refill_interval / tau_mu)
L0 = (2 * refill_interval / (tau_mu * decay_factor)) * L_avg

# Time array
t = np.arange(0, total_time, dt)

# Luminosity and BIB arrays
L = np.zeros_like(t)
BIB = np.zeros_like(t)
for i in range(len(t)):
    t_since_refill = t[i] % refill_interval
    L[i] = L0 * np.exp(-2 * t_since_refill / tau_mu)
    BIB[i] = np.exp(-t_since_refill / tau_mu)  # normalized to 1.0 at t=0

# Plot
fig, ax1 = plt.subplots(figsize=(10, 5))

# Left axis: normalized luminosity
ax1.plot(t, L / 1e34, label=r'$L(t)$', color='tab:blue')
ax1.axhline(L_avg / 1e34, color='gray', linestyle='--', label='Average $L$')
ax1.set_xlabel("Time [s]")
ax1.set_ylabel(r"Instantaneous Luminosity [$10^{34}$ cm$^{-2}$ s$^{-1}$]", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim([0, 5])
#ax1.grid(True)

# Right axis: relative BIB
ax2 = ax1.twinx()
ax2.plot(t, BIB, label='Beam-Induced Background', color='tab:red')
ax2.set_ylabel("Fraction of Peak Beam-Induced Background", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 1.5)

# Legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

#plt.title("Instantaneous Luminosity and Beam-Induced Background Over Time")
plt.tight_layout()
#plt.show()
fig.savefig("InstantaneousLuminosity.pdf")
