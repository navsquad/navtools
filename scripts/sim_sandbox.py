# %%
import navtools.simulation as ntsim
import navtools.emitters as ntem

from tqdm import tqdm
from datetime import datetime, timedelta

emitters = ntem.SatelliteEmitters(constellations=["gps", "iridium-next"], mask_angle=10)
emitters.rx_pos = [423756, -5361363, 3417705]

datetimes = [datetime(2023, 8, 14) + timedelta(0, sec) for sec in range(1000)]
states = [emitters.compute_states(datetime=time) for time in tqdm(datetimes)]


# %%
# LEAP_SECONDS = 18

# config = ntsim.ReceiverSimulationConfiguration(
#     fsim=50,
#     datetime=datetime(2023, 7, 31, 0, 0, LEAP_SECONDS),
#     constellations="gps",
#     mask_angle=15,
# )

# # %%
# sim = ntsim.ReceiverSimulation(config=config)
# sim.simulate_duration(rx_pos=[423756, -5361363, 3417705])
