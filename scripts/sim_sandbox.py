# %%
import navtools.simulation as ntsim
import navtools.emitters as ntem

from tqdm import tqdm
from datetime import datetime

emitters = ntem.SatelliteEmitters(constellations=["gps", "galileo"], mask_angle=0)
emitters.rx_pos = [423756, -5361363, 3417705]

for _ in tqdm(range(10)):
    states = emitters.compute_states(datetime=datetime(2022, 12, 22))

# %%
LEAP_SECONDS = 18

config = ntsim.ReceiverSimulationConfiguration(
    fsim=50,
    datetime=datetime(2023, 7, 31, 0, 0, LEAP_SECONDS),
    constellations="gps",
    mask_angle=15,
)

# %%
sim = ntsim.ReceiverSimulation(config=config)
sim.simulate_duration(rx_pos=[423756, -5361363, 3417705])
