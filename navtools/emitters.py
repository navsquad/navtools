import importlib
import warnings
import numpy as np

from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime

from navtools.exceptions import UnsetReceiverState

from laika import AstroDog
from laika.gps_time import GPSTime
from laika.helpers import get_el_az


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


class SatelliteEmitters:
    def __init__(self, constellations: list = None, mask_angle: float = 10.0):
        constellation_literals = self._get_constellation_literals(
            constellations=constellations
        )
        self._mask_angle = mask_angle
        self._is_rx_state_unset = True
        self._rx_pos = np.zeros(3)
        self._rx_vel = np.zeros(3)

        self._dog = AstroDog(valid_const=constellation_literals)

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
        self._rx_pos = pos

    @rx_vel.setter
    def rx_vel(self, vel: np.array):
        self._is_rx_state_unset = False
        self._rx_vel = vel

    def compute_states(self, datetime: datetime, is_only_visible_emitters: bool = True):
        if self._is_rx_state_unset:
            raise UnsetReceiverState

        self._gps_time = GPSTime.from_datetime(datetime=datetime)

        emitter_states = self._dog.get_all_sat_info(time=self._gps_time)
        self._emitter_states = self._compute_los_states(
            emitter_states=emitter_states,
            is_only_visible_emitters=is_only_visible_emitters,
        )
        return self._emitter_states

    def _compute_los_states(self, emitter_states: dict, is_only_visible_emitters: bool):
        emitters = defaultdict()

        for emitter_prn, emitter_state in emitter_states.items():
            emitter_pos = emitter_state[0]
            emitter_vel = emitter_state[1]
            emitter_clock_bias = emitter_state[2]
            emitter_clock_drift = emitter_state[3]

            if is_only_visible_emitters:
                is_visible = self.compute_visibility_status(
                    rx_pos=self._rx_pos,
                    emitter_pos=emitter_pos,
                    mask_angle=self._mask_angle,
                )

                if not is_visible:
                    continue

            range, unit_vector = self.compute_range_and_unit_vector(
                rx_pos=self._rx_pos, emitter_pos=emitter_pos
            )
            range_rate = self.compute_range_rate(
                rx_vel=self._rx_vel, emitter_vel=emitter_vel, unit_vector=unit_vector
            )

            emitter_state = SatelliteEmitterState(
                prn=emitter_prn,
                gps_time=self._gps_time,
                pos=emitter_pos,
                vel=emitter_vel,
                clock_bias=emitter_clock_bias,
                clock_drift=emitter_clock_drift,
                range=range,
                range_rate=range_rate,
            )
            emitters[emitter_prn] = emitter_state

        return emitters

    @staticmethod
    def compute_visibility_status(
        rx_pos: np.array, emitter_pos: np.array, mask_angle: float = 10.0
    ):
        sat_el, _ = get_el_az(pos=rx_pos, sat_pos=emitter_pos)
        is_visible = np.degrees(sat_el) >= mask_angle

        return is_visible

    @staticmethod
    def compute_range_and_unit_vector(rx_pos: np.array, emitter_pos: np.array):
        rx_pos_rel_sat = rx_pos - emitter_pos
        range = np.linalg.norm(rx_pos_rel_sat)
        unit_vector = rx_pos_rel_sat / range

        return range, unit_vector

    @staticmethod
    def compute_range_rate(
        rx_vel: np.array, emitter_vel: np.array, unit_vector: np.array
    ):
        rx_vel_rel_sat = rx_vel - emitter_vel
        range_rate = np.sum(rx_vel_rel_sat * unit_vector)

        return range_rate

    def _get_constellation_literals(self, constellations: list):
        self._constellations = []
        literals = []

        module = importlib.import_module("laika.helpers")
        obj = getattr(module, "ConstellationId")

        if isinstance(constellations, str):
            constellations = constellations.split()

        for constellation in constellations:
            self._constellations.append(constellation.lower())
            literal = getattr(obj, constellation.upper())
            literals.append(literal)

        return literals
