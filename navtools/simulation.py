from dataclasses import dataclass
from collections import defaultdict
from typing import Union
from datetime import datetime
from laika import AstroDog
from laika.gps_time import GPSTime
from laika.helpers import get_el_az

import importlib
import numpy as np


@dataclass(frozen=True)
class SimulationConfiguration:
    duration: float
    fsim: float
    datetime: datetime
    constellations: list
    mask_angle: float = 10.0

    def __post_init__(self):
        object.__setattr__(self, "constellations", self._get_constellation_objects())

    def _get_constellation_objects(self):
        output = []
        module = importlib.import_module("laika.helpers")
        obj = getattr(module, "ConstellationId")

        if isinstance(self.constellations, str):
            constellations = self.constellations.split()
        else:
            constellations = self.constellations

        for constellation in constellations:
            literal = getattr(obj, constellation.upper())
            output.append(literal)

        return output


@dataclass(frozen=True)
class SatelliteState:
    epoch: int
    gps_time: GPSTime
    pos: float
    vel: float
    range: float
    range_rate: float


@dataclass(frozen=True)
class SatelliteObservables:
    epoch: int
    psr: float
    psr_rate: float


class Simulation:
    NUM_STATIC_POS_ELEMENTS = 3

    def __init__(self, config: SimulationConfiguration):
        self._config = config
        self._dog = AstroDog(valid_const=self._config.constellations)
        self._time = GPSTime.from_datetime(datetime=self._config.datetime)

        self.ref_states = defaultdict(lambda: [])
        self.observables = defaultdict(lambda: [])

    def simulate_constellations(self, rx_pos: np.array, rx_vel: np.array = None):
        is_rx_static = np.asarray(rx_pos).size == Simulation.NUM_STATIC_POS_ELEMENTS

        if is_rx_static:
            rx_pos, rx_vel = self._build_static_trajectory(rx_pos=rx_pos)

        for epoch, (pos, vel) in enumerate(zip(rx_pos, rx_vel)):
            all_sat_info_per_epoch = self._dog.get_all_sat_info(time=self._time)

            for sat_prn, sat_info in all_sat_info_per_epoch.items():
                sat_el, _ = get_el_az(pos=pos, sat_pos=sat_info[0])
                is_visible = np.degrees(sat_el) > self._config.mask_angle

                if is_visible:
                    self._compute_state(
                        epoch=epoch,
                        sat_prn=sat_prn,
                        sat_info=sat_info,
                        rx_pos=pos,
                        rx_vel=vel,
                    )
                    # self._get_observables()

            self._time += 1 / self._config.fsim

    def _build_static_trajectory(self, rx_pos):
        num_epochs = int(self._config.duration * self._config.fsim) + 1
        trajectory = np.tile(rx_pos, (num_epochs, 1))
        velocity = np.zeros_like(trajectory)

        return trajectory, velocity

    def _compute_state(self, epoch, sat_prn, sat_info, rx_pos, rx_vel):
        reference_sat_states = self.ref_states[sat_prn]

        # extract states
        sat_pos = sat_info[0]
        sat_vel = sat_info[1]

        # compute truth observables
        rx_pos_rel_sat = rx_pos - sat_pos
        range = np.linalg.norm(rx_pos_rel_sat)

        unit_vector = rx_pos_rel_sat / range
        rx_vel_rel_sat = rx_vel - sat_vel
        range_rate = np.sum(rx_vel_rel_sat * unit_vector)

        epoch_state = SatelliteState(
            epoch=epoch,
            gps_time=self._time,
            pos=sat_pos,
            vel=sat_vel,
            range=range,
            range_rate=range_rate,
        )
        reference_sat_states.append(epoch_state)


@dataclass
class NavigationClock:
    h0: float
    h1: float
    h2: float


LOW_QUALITY_TCXO = NavigationClock(h0=2e-19, h1=7e-21, h2=2e-20)
HIGH_QUALITY_TCXO = NavigationClock(h0=2e-21, h1=1e-22, h2=2e-20)
OCXO = NavigationClock(h0=2e-25, h1=7e-25, h2=6e-25)
RUBIDIUM = NavigationClock(h0=2e-22, h1=4.5e-26, h2=1e-30)
CESIUM = NavigationClock(h0=2e-22, h1=5e-27, h2=1.5e-33)
