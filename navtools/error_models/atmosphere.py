import numpy as np

from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from numba import njit

from laika import AstroDog
from laika.gps_time import GPSTime
from navtools.conversions import ecef2lla


### ionosphere ###
@dataclass
class IonosphereModelParameters:
    time: datetime
    rx_pos: np.array
    emitter_pos: np.array
    az: float
    el: float
    fcarrier: float


class IonosphereModel(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_delay(self, params: IonosphereModelParameters):
        pass


class TotalElectronCountMapModel(IonosphereModel):
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
        pass  # TODO: implement Klobuchar with default alpha and beta


### troposphere ###
@dataclass
class TroposphereModelParameters:
    rx_pos: np.array
    el: float


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
    # TODO: clean this up
    rx_pos_lla = ecef2lla(x=rx_pos[0], y=rx_pos[1], z=rx_pos[2])
    if rx_pos_lla[2] < -1e3 or 1e4 < rx_pos_lla[2] or el <= 0:
        return 0.0

    # /* standard atmosphere */
    hgt = 0.0 if rx_pos_lla.alt < 0.0 else rx_pos_lla.alt

    pres = 1013.25 * pow(1.0 - 2.2557e-5 * hgt, 5.2568)
    temp = temperature_at_sea_level - 6.5e-3 * hgt + 273.16
    e = 6.108 * humidity * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))

    # /* saastamoninen model */
    z = np.pi / 2.0 - el
    trph = (
        0.0022768
        * pres
        / (1.0 - 0.00266 * np.cos(2.0 * rx_pos_lla.lat) - 0.00028 * hgt / 1e3)
        / np.cos(z)
    )
    trpw = 0.002277 * (1255.0 / temp + 0.05) * e / np.cos(z)
    return trph + trpw


### factories ###
def get_ionosphere_model(model_name: str):
    """factory function that retrieves requested ionosphere model

    Parameters
    ----------
    model_name : str
        name of ionosphere model

    Returns
    -------
    IonosphereModel
        ionosphere model
    """
    IONOSPHERE_MODELS = {
        "tecmap": TotalElectronCountMapModel(),
        "klobuchar": KlobucharModel(),
    }

    model_name = "".join([i for i in model_name if i.isalnum()]).casefold()
    model = IONOSPHERE_MODELS.get(
        model_name.casefold(),
        TotalElectronCountMapModel(),  # defaults to tec map
    )

    return model


def get_troposphere_model(model_name: str):
    """factory function that retrieves requested troposphere model

    Parameters
    ----------
    model_name : str
        name of troposphere model

    Returns
    -------
    TroposphereModel
        troposphere model
    """
    TROPOSPHERE_MODELS = {"saastamoinen": SaastamoinenModel()}

    model_name = "".join([i for i in model_name if i.isalnum()]).casefold()
    model = TROPOSPHERE_MODELS.get(
        model_name.casefold(), SaastamoinenModel()
    )  # defaults to saastamoinen

    return model
