# %%
import navtools.emitters as ntem
import numpy as np

from datetime import datetime, timedelta

LEAP_SECONDS = 18
INITIAL_TIME = datetime(2023, 8, 19, 18, 50, LEAP_SECONDS)
DURATION = 2000  # [s]
RX_POS = np.array([423756, -5361363, 3417705])
CONSTELLATIONS = ["GPS", "GALILEO", "GLONASS", "IRIDIUM-NEXT", "ORBCOMM"]

emitters = ntem.SatelliteEmitters(constellations=CONSTELLATIONS, mask_angle=10)

# %% single epoch

print("\nsingle epoch processing...\n")
all_states = emitters.from_datetime(
    datetime=INITIAL_TIME,
    rx_pos=RX_POS,
    is_only_visible_emitters=False,
)
in_view_states = emitters.from_gps_time(
    gps_time=ntem.GPSTime.from_datetime(datetime=INITIAL_TIME),
    rx_pos=RX_POS,
)

# %% multiple epochs

print("\nmulti-epoch processing...\n")
datetimes = [datetime(2023, 8, 14) + timedelta(0, sec) for sec in range(DURATION)]

all_duration_states = emitters.from_datetimes(
    datetimes=datetimes, rx_pos=RX_POS, is_only_visible_emitters=False
)

print("\n")
in_view_duration_states = emitters.from_datetimes(datetimes=datetimes, rx_pos=RX_POS)
