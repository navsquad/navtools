import numpy as np

from dataclasses import dataclass
from datetime import datetime

from navtools.emitters.satellite import SatelliteEmitters
from navtools.error_models.signal import compute_carrier_to_noise
from navtools.error_models.atmosphere import (
    get_ionosphere_delay_model,
    get_troposphere_delay_model,
    IonosphereModelParameters,
    TroposphereModelParameters,
)
from navtools.signals.types import get_signal_properties


@dataclass
class MeasurementLevelConfiguration:
    constellations: list


class MeasurementLevelSimulation:
    def __init__(self) -> None:
        self._emitters = SatelliteEmitters(
            constellations=self._constellations, mask_angle=self._mask_angle
        )
        pass

    def simulate(
        self, times: list[datetime], rx_pos: np.array, rx_vel: np.array = np.zeros(3)
    ):
        self._compute_emitter_states(times=times, rx_pos=rx_pos, rx_vel=rx_vel)

    def _compute_emitter_states(
        self, times: list[datetime], rx_pos: np.array, rx_vel: np.array
    ):
        self._truth = self._emitters.from_datetimes(
            datetimes=times, rx_pos=rx_pos, rx_vel=rx_vel
        )

    def _compute_carrier_to_noise(self, truth_epoch: dict):
        ranges = {emitter: state.range for (emitter, state) in truth_epoch.items()}
        cn0 = {
            emitter: compute_carrier_to_noise(range=range)
            for (emitter, range) in ranges.items()
        }

    def _compute_observables():
        pass
