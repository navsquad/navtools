import numpy as np

from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from numba import njit

from navtools.constants import SPEED_OF_LIGHT, BOLTZMANN_CONSTANT
from navtools.conversions import ecef2lla
from laika import AstroDog
from laika.gps_time import GPSTime


@njit
def compute_carrier_to_noise(
    range: float,
    transmit_power: float,
    transmit_antenna_gain: float,
    fcarrier: float,
    js: float = 0,
    temperature: float = 290,
):
    """computes carrier-to-noise density ratio from free space path loss (derived from Friis equation)

    Parameters
    ----------
    range : float
        range to emitter [m]
    transmit_power : float
        trasmit power [dBW]
    transmit_antenna_gain : float
        isotropic antenna gain [dBi]
    fcarrier : float
        signal carrier frequency [Hz]
    js : float, optional
        jammer-to-signal ratio [dB], by default 0
    temperature : float, optional
        noise temperature [K], by default 290

    Returns
    -------
    float
        carrier-to-noise ratio [dB-Hz]

    Reference
    -------
    A. Joseph, “Measuring GNSS Signal Strength: What is the difference between SNR and C/N0?,” InsideGNSS, Nov. 2010.
    """

    ADDITIONAL_NOISE_FIGURE = 3  # [dB-Hz] cascaded + band-limiting/quantization noise

    EIRP = transmit_power + transmit_antenna_gain  # [dBW]
    wavelength = SPEED_OF_LIGHT / fcarrier  # [m]
    FSPL = 20 * np.log10(4 * np.pi * range / wavelength)  # [dB] free space path loss

    received_carrier_power = EIRP - FSPL - js  # [dBW]
    thermal_noise_density = 10 * np.log10(BOLTZMANN_CONSTANT * temperature)  # [dBW/Hz]

    nominal_cn0 = received_carrier_power - thermal_noise_density  # [dB-Hz]
    cn0 = nominal_cn0 - ADDITIONAL_NOISE_FIGURE

    return cn0


@njit
def cn02snr(cn0: float, front_end_bw: float = 4e6, noise_figure: float = 0.0):
    snr = cn0 - 10 * np.log10(front_end_bw) - noise_figure  # dB

    return snr


@njit
def snr2cn0(snr: float, front_end_bw: float = 4e6, noise_figure: float = 0.0):
    cn0 = snr + 10 * np.log10(front_end_bw) + noise_figure  # dB-Hz

    return cn0


@dataclass
class IonosphereModelParameters:
    time: datetime
    rx_pos: np.array
    emitter_pos: np.array
    az: float
    el: float
    fcarrier: float


@dataclass
class TroposphereModelParameters:
    rx_pos: np.array
    el: float


def get_ionosphere_delay_model(model_type: str):
    IONOSPHERE_MODELS = {
        "tec_map_delay": TotalElectronCountMapDelayModel(),
        "klobuchar": KlobucharModel(),
    }
    model = IONOSPHERE_MODELS.get(
        model_type.casefold(), TotalElectronCountMapDelayModel()
    )

    return model


def get_troposphere_delay_model(model_type: str):
    TROPOSPHERE_MODELS = {"saastamoinen": SaastamoinenModel()}
    model = TROPOSPHERE_MODELS.get(model_type.casefold(), SaastamoinenModel())

    return model


class IonosphereModel(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_delay(self, params: IonosphereModelParameters):
        pass


class TotalElectronCountMapDelayModel(IonosphereModel):
    def __init__(self) -> None:
        super().__init__()

    def get_delay(self, params: IonosphereModelParameters):
        gps_time = GPSTime.from_datetime(datetime=params.time)

        if not hasattr(self, "dog"):
            self.dog = AstroDog()
            self.ionex = self.dog.get_ionex(time=gps_time)

        delay = self.ionex.get_delay(
            time=gps_time,
            freq=params.fcarrier,
            rcv_pos=params.rx_pos,
            sat_pos=params.emitter_pos,
            az=params.az,
            el=params.el,
        )

        return delay


class KlobucharModel(IonosphereModel):
    def __init__(self) -> None:
        super().__init__()

    def get_delay(self, params: IonosphereModelParameters):
        if not hasattr(self, "dog"):
            self.dog = AstroDog()
            self.ionex = self.dog.get_ionex(time=params.time)

        delay = self.ionex.get_delay(
            time=params.time,
            freq=params.fcarrier,
            rcv_pos=params.rx_pos,
            sat_pos=params.emitter_pos,
            az=params.az,
            el=params.el,
        )

        return delay


class TroposphereModel(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_delay(self, params: TroposphereModelParameters):
        pass


class SaastamoinenModel(TroposphereModel):
    def __init__(self) -> None:
        super().__init__()

    def get_delay(self, params: TroposphereModelParameters):
        delay = compute_saastamoinen_delay(rx_pos=params.rx_pos, el=params.el)

        return delay


@njit
def compute_saastamoinen_delay(
    rx_pos, el, humidity=0.75, temperature_at_sea_level=15.0
):
    """function from RTKlib: https://github.com/tomojitakasu/RTKLIB/blob/master/src/rtkcmn.c#L3362-3362
        with no changes by way of laika: https://github.com/commaai/laika

    Parameters
    ----------
    rx_pos : _type_
        receiver ECEF position
    el : _type_
        elevation to emitter [rad]
    humidity : float, optional
        relative humidity, by default 0.75
    temperature_at_sea_level : float, optional
        temperature at sea level [C], by default 15.0

    Returns
    -------
    _type_
        sum of wet and dry tropospheric delay [m]
    """
    rx_pos_lla = ecef2lla(x=rx_pos[0], y=rx_pos[1], z=rx_pos[2])
    if rx_pos_lla[2] < -1e3 or 1e4 < rx_pos_lla[2] or el <= 0:
        return 0.0

    # /* standard atmosphere */
    hgt = 0.0 if rx_pos_lla[2] < 0.0 else rx_pos_lla[2]

    pres = 1013.25 * pow(1.0 - 2.2557e-5 * hgt, 5.2568)
    temp = temperature_at_sea_level - 6.5e-3 * hgt + 273.16
    e = 6.108 * humidity * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))

    # /* saastamoninen model */
    z = np.pi / 2.0 - el
    trph = (
        0.0022768
        * pres
        / (1.0 - 0.00266 * np.cos(2.0 * rx_pos_lla[0]) - 0.00028 * hgt / 1e3)
        / np.cos(z)
    )
    trpw = 0.002277 * (1255.0 / temp + 0.05) * e / np.cos(z)
    return trph + trpw
