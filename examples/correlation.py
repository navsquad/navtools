import navtools as nt
import numpy as np
import matplotlib.pyplot as plt

prn1 = 2 * np.random.randint(low=0, high=2, size=1000) - 1
prn2 = 2 * np.random.randint(low=0, high=2, size=1000) - 1

rx_baseband_prn1 = np.roll(prn1, 400)
rx_baseband_prn2 = np.roll(prn2, 700)

correlation_prn1 = nt.dsp.parcorr(rx_baseband_prn1, prn1)
correlation_prn2 = nt.dsp.parcorr(rx_baseband_prn2, prn2)

ax = nt.plot.correlation(correlation_prn1, title="PRN Correlations", label="PRN 1")
ax.plot(correlation_prn2, label="PRN 2")
plt.legend()
plt.show()
