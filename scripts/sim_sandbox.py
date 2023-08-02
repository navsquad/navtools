# %%
import navtools.simulation as ntsim

from datetime import datetime

LEAP_SECONDS = 18

config = ntsim.SimulationConfiguration(
    duration=30,
    fsim=2,
    datetime=datetime(2023, 7, 31, 0, 0, LEAP_SECONDS),
    constellations=["gps", "galileo"],
    mask_angle=15,
)

# %%
sim = ntsim.Simulation(config=config)
sim.simulate_constellations(rx_pos=[423756, -5361363, 3417705])
