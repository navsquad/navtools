import importlib
import numpy as np
import pymap3d as pmap
import itertools

from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timezone
from skyfield.api import load, EarthSatellite
from numba import njit

from navtools.exceptions import UnsetReceiverState, UnsupportedConstellation
from navtools.kinematics import ecef2lla, ecef2enu

from laika import AstroDog
from laika.gps_time import GPSTime
from laika.helpers import get_el_az


@njit
def compute_visibility_status(
    rx_pos: np.array, emitter_pos: np.array, mask_angle: float = 10.0
):
    emitter_az, emitter_el = compute_az_and_el(rx_pos=rx_pos, emitter_pos=emitter_pos)
    is_visible = np.degrees(emitter_el) >= mask_angle

    return is_visible, emitter_az, emitter_el


@njit
def compute_az_and_el(rx_pos: np.array, emitter_pos: np.array):
    range, _ = compute_range_and_unit_vector(rx_pos=rx_pos, emitter_pos=emitter_pos)
    lla = ecef2lla(x=rx_pos[0], y=rx_pos[1], z=rx_pos[2])
    enu = ecef2enu(
        x=emitter_pos[0],
        y=emitter_pos[1],
        z=emitter_pos[2],
        lat0=lla.lat,
        lon0=lla.lon,
        alt0=lla.alt,
    )
    el = np.arcsin(enu.up / range)
    az = np.arctan2(enu.east, enu.north)

    return az, el


@njit
def compute_range_and_unit_vector(rx_pos: np.array, emitter_pos: np.array):
    rx_pos_rel_sat = rx_pos - emitter_pos
    range = np.sqrt(np.sum(rx_pos_rel_sat**2))
    unit_vector = rx_pos_rel_sat / range

    return range, unit_vector


@njit
def compute_range_rate(rx_vel: np.array, emitter_vel: np.array, unit_vector: np.array):
    rx_vel_rel_sat = rx_vel - emitter_vel
    range_rate = np.sum(rx_vel_rel_sat * unit_vector)

    return range_rate


@dataclass(frozen=True)
class SatelliteEmitterState:
    prn: str
    gps_time: GPSTime
    pos: float
    vel: float
    clock_bias: float
    clock_drift: float
    range: float
    range_rate: float
    az: float
    el: float


class SatelliteEmitters:
    def __init__(self, constellations: list, mask_angle: float = 10.0):
        self._filter_constellations(constellations=constellations)

        if self._laika_constellations:
            laika_literals = self._get_laika_literals()
            self._dog = AstroDog(valid_const=laika_literals)

        if self._skyfield_constellations:
            self._skyfield_satellites = self._get_skyfield_satellites()
            self._ts = load.timescale()

        self._mask_angle = mask_angle
        self._is_rx_state_unset = True
        self._is_multiple_epoch = False
        self._rx_pos = np.zeros(3)
        self._rx_vel = np.zeros(3)

    @property
    def states(self):
        return self._emitter_states

    @property
    def rx_pos(self):
        return self._rx_pos

    @property
    def rx_vel(self):
        return self._rx_vel

    @rx_pos.setter
    def rx_pos(self, pos: np.array):
        self._is_rx_state_unset = False
        self._rx_pos = np.array(pos)

    @rx_vel.setter
    def rx_vel(self, vel: np.array):
        self._is_rx_state_unset = False
        self._rx_vel = vel

    def compute_states(self, datetime: datetime, is_only_visible_emitters: bool = True):
        if self._is_rx_state_unset:
            raise UnsetReceiverState

        if isinstance(datetime, list):
            self._is_multiple_epoch = True

        emitter_states = {}

        if self._is_multiple_epoch:  # TODO: this could be rewritten but i'm tired
            if self._laika_constellations:
                pass

            if self._skyfield_constellations:
                datetime = [time.replace(tzinfo=timezone.utc) for time in datetime]
                time = self._ts.from_datetimes(datetime_list=datetime)
                skyfield_states = self._get_all_skyfield_states(time=time)
                emitter_states.update(skyfield_states)

        else:
            if self._laika_constellations:
                self._gps_time = GPSTime.from_datetime(datetime=datetime)
                laika_states = self._dog.get_all_sat_info(time=self._gps_time)
                emitter_states.update(laika_states)

            if self._skyfield_constellations:
                datetime = datetime.replace(tzinfo=timezone.utc)
                time = self._ts.from_datetime(datetime=datetime)
                skyfield_states = self._get_all_skyfield_states(time=time)
                emitter_states.update(skyfield_states)

        if self._is_multiple_epoch:
            # TODO: process all skyfield at one time then compare
            pass
        else:
            self._emitter_states = self._compute_single_epoch_los_states(
                emitter_states=emitter_states,
                is_only_visible_emitters=is_only_visible_emitters,
            )
        return self._emitter_states

    def _compute_single_epoch_los_states(
        self, emitter_states: dict, is_only_visible_emitters: bool
    ):
        emitters = defaultdict()

        for emitter_prn, emitter_state in emitter_states.items():
            emitter_pos = emitter_state[0]
            emitter_vel = emitter_state[1]
            emitter_clock_bias = emitter_state[2]
            emitter_clock_drift = emitter_state[3]

            if is_only_visible_emitters:
                is_visible, emitter_az, emitter_el = compute_visibility_status(
                    rx_pos=self._rx_pos,
                    emitter_pos=emitter_pos,
                    mask_angle=self._mask_angle,
                )

                if not is_visible:
                    continue

            range, unit_vector = compute_range_and_unit_vector(
                rx_pos=self._rx_pos, emitter_pos=emitter_pos
            )
            range_rate = compute_range_rate(
                rx_vel=self._rx_vel, emitter_vel=emitter_vel, unit_vector=unit_vector
            )

            emitter_state = SatelliteEmitterState(
                prn=emitter_prn,
                gps_time=1,
                pos=emitter_pos,
                vel=emitter_vel,
                clock_bias=emitter_clock_bias,
                clock_drift=emitter_clock_drift,
                range=range,
                range_rate=range_rate,
                az=emitter_az,
                el=emitter_el,
            )
            emitters[emitter_prn] = emitter_state

        return emitters

    # def _compute_time_array_laika_info(self):
    def _filter_constellations(self, constellations: list):
        if isinstance(constellations, str):
            constellations = constellations.split()

        GNSS = ["gps", "glonass", "galileo", "beidou", "qznss"]
        LEO = ["iridium", "iridium-next", "orbcomm", "globalstar", "oneweb", "starlink"]

        self._laika_constellations = []
        self._skyfield_constellations = []

        for constellation in constellations:
            if constellation.lower() in GNSS:
                self._laika_constellations.append(constellation)
            elif constellation.lower() in LEO:
                self._skyfield_constellations.append(constellation)
            else:
                raise UnsupportedConstellation(constellation=constellation)

    def _get_laika_literals(self):
        constellations = self._laika_constellations
        literals = []

        module = importlib.import_module("laika.helpers")
        obj = getattr(module, "ConstellationId")

        if isinstance(constellations, str):
            constellations = constellations.split()

        for constellation in constellations:
            literal = getattr(obj, constellation.upper())
            literals.append(literal)

        return literals

    def _get_skyfield_satellites(self):
        constellations = self._skyfield_constellations
        satellites = [
            load.tle_file(
                url=f"https://celestrak.org/NORAD/elements/gp.php?GROUP={constellation}&FORMAT=tle",
            )
            for constellation in constellations
        ]
        [print(constellation) for constellation in constellations]

        satellites = list(itertools.chain(*satellites))  # flatten list

        return satellites

    def _get_all_skyfield_states(self, time):
        emitters = defaultdict()

        for emitter in self._skyfield_satellites:
            emitter_state = emitter.at(time)
            state = [emitter_state.xyz.m, emitter_state.velocity.m_per_s, 0, 0]
            emitters[emitter.name] = state

        return emitters
