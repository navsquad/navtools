import numpy as np

from datetime import datetime, timedelta
from navtools.emitters.satellite import SatelliteEmitters, GPSTime

# user-defined parameters
CONSTELLATIONS = ["gps", "galileo", "glonass", "iridium", "orbcomm"]

LEAP_SECONDS = 18  # current gps leap seconds
INITIAL_TIME = datetime(2023, 9, 22, 18, 50, LEAP_SECONDS)
MULTIPLE_EPOCHS_DURATION = 3600  # [s]

RX_POS = np.array([423756, -5361363, 3417705])  # auburn, al ECEF position [m]
MASK_ANGLE = 10  # [deg]

# define SatelliteEmitters object
emitters = SatelliteEmitters(constellations=CONSTELLATIONS, mask_angle=MASK_ANGLE)

# process single epoch using datetime and GPSTime
print("\nsingle epoch processing...\n")

# returns all satellites
all_states = emitters.from_datetime(
    datetime=INITIAL_TIME,
    rx_pos=RX_POS,
    is_only_visible_emitters=False,
)
# returns satellites in view
in_view_states = emitters.from_gps_time(
    gps_time=GPSTime.from_datetime(datetime=INITIAL_TIME),
    rx_pos=RX_POS,
)

# process multiple epochs using datetimes
print("\nmulti-epoch processing...\n")
datetimes = [
    INITIAL_TIME + timedelta(0, sec) for sec in range(MULTIPLE_EPOCHS_DURATION)
]

# returns all satellites
all_duration_states = emitters.from_datetimes(
    datetimes=datetimes, rx_pos=RX_POS, is_only_visible_emitters=False
)

# returns satellites in view
in_view_duration_states = emitters.from_datetimes(datetimes=datetimes, rx_pos=RX_POS)
