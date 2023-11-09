import numpy as np
import matplotlib.pyplot as plt

from navtools.dsp import carrier, apply_carrier_to_noise

# currently validating CN0 application to carrier wave

fcarrier = 1
fsamp = 100
duration = 10
carrier = carrier(fcarrier=fcarrier, fsamp=fsamp, duration=duration)
noisy_carrier = apply_carrier_to_noise(samples=carrier, cn0=45, fsamp=fsamp)

_, cn0_ax = plt.subplots()
cn0_ax.plot(np.real(noisy_carrier))
cn0_ax.plot(np.real(carrier))
plt.show()
