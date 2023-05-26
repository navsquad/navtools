# %%
import navtools as nt
import numpy as np
import matplotlib.pyplot as plt
import mpld3

mpld3.enable_notebook()

# %%
ideal_baseband_signal = 2 * np.random.randint(low=0, high=2, size=int(100)) - 1
baseband_signal = np.roll(ideal_baseband_signal, 50)
replica = ideal_baseband_signal
correlation = nt.dsp.parcorr(baseband_signal, replica)

# %%
ax = nt.plot.correlation(correlation, title="GPS Correlations")
ax.plot(correlation * 2)
plt.show()

# %%
