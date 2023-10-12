import numpy as np

from datetime import datetime
from navtools.emitters.satellite import SatelliteEmitters
from navtools.error_models.error_models import (
    compute_carrier_to_noise,
    get_ionosphere_delay_model,
    get_troposphere_delay_model,
    IonosphereModelParameters,
    TroposphereModelParameters,
)

# from navtools.signals.tools import get_signal_properties
from laika import AstroDog
from tqdm import tqdm

# satellite parameters
CONSTELLATIONS = "iridium"
LEAP_SECONDS = 18  # current gps leap seconds
INITIAL_TIME = datetime(2023, 9, 22, 18, 50, LEAP_SECONDS)
TRANSMIT_POWER = 50  # [W]
TRANSMIT_ANTENNA_GAIN = 12  # [dBi]

# receiver parameters
RX_POS = np.array([423756, -5361363, 3417705])  # [m - ECEF] auburn, al position
MASK_ANGLE = 10  # [deg]

# error parameters
JS = 40  # [dB] jammer-to-signal ratio

emitters = SatelliteEmitters(constellations=CONSTELLATIONS, mask_angle=MASK_ANGLE)
in_view_emitters = emitters.from_datetime(
    datetime=INITIAL_TIME,
    rx_pos=RX_POS,
)
l1ca = get_signal_properties(signal_type="gpsl1ca")

positions = np.array([state.pos for state in in_view_emitters.values()])
ranges = np.array([state.range for state in in_view_emitters.values()])
els = np.array([state.el for state in in_view_emitters.values()])
azs = np.array([state.az for state in in_view_emitters.values()])

signal_cn0 = compute_carrier_to_noise(
    range=ranges,
    transmit_power=10 * np.log10(TRANSMIT_POWER),
    transmit_antenna_gain=TRANSMIT_ANTENNA_GAIN,
    js=JS,
    fcarrier=l1ca.fcarrier,
)
print(f"channel carrier-to-noise [dB-Hz] with J/S of {JS} dB: {signal_cn0}")

tparams = TroposphereModelParameters(
    rx_pos=RX_POS,
    el=0,
)
iparams = IonosphereModelParameters(
    time=INITIAL_TIME,
    rx_pos=RX_POS,
    emitter_pos=0,
    az=0,
    el=0,
    fcarrier=l1ca.fcarrier,
)

ionosphere = get_ionosphere_delay_model(model_type="tec_map_delay")
troposphere = get_troposphere_delay_model(model_type="saastamoinen")

iono_delay = []
tropo_delay = []

for pos, az, el in zip(positions, azs, els):
    tparams.el = el
    iparams.emitter_pos = pos
    iparams.el = el
    iparams.az = az

    iono_delay.append(ionosphere.get_delay(params=iparams))
    tropo_delay.append(troposphere.get_delay(params=tparams))

print(f"ionosphere channel delays: {iono_delay}")
print(f"troposphere channel delays: {tropo_delay}")
