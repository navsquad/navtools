import numpy as np

from numba import njit
from ..conversions import ecef2lla, ecef2enu
from navtools.constants import SPEED_OF_LIGHT, BOLTZMANN_CONSTANT


@njit
def compute_visibility_status(
    rx_pos: np.array, emitter_pos: np.array, mask_angle: float = 10.0
) -> tuple[bool, float, float]:
    """computes visibility of emitter from receiver location in ECEF reference frame

    Parameters
    ----------
    rx_pos : np.array
        receiver position(s) in ECEF reference frame
    emitter_pos : np.array
        emitter position(s) in ECEF reference frame
    mask_angle : float, optional
        mask angle experienced at receiver position(s), by default 10.0

    Returns
    -------
    tuple[bool, float, float]
        emitter visibility status, emitter azimuth (rad), emitter elevation(rad)
    """
    emitter_az, emitter_el = compute_az_and_el(rx_pos=rx_pos, emitter_pos=emitter_pos)
    is_visible = np.degrees(emitter_el) >= mask_angle

    return is_visible, emitter_az, emitter_el


@njit
def compute_az_and_el(rx_pos: np.array, emitter_pos: np.array) -> tuple[float, float]:
    """computes azimuth and elevation of emitter from receiver location in ECEF reference frame

    Parameters
    ----------
    rx_pos : np.array
        receiver position(s) in ECEF reference frame
    emitter_pos : np.array
        emitter position(s) in ECEF reference frame

    Returns
    -------
    tuple[float, float]
        emitter azimuth (rad), emitter elevation (rad)
    """
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
def compute_range_and_unit_vector(
    rx_pos: np.array, emitter_pos: np.array
) -> tuple[float, np.array]:
    """computes range and unit vector to emitter from receiver location in ECEF reference frame

    Parameters
    ----------
    rx_pos : np.array
        receiver position(s) in ECEF reference frame
    emitter_pos : np.array
        emitter position(s) in ECEF reference frame

    Returns
    -------
    tuple[float, np.array]
        range (m), unit vector in ECEF reference frame
    """
    rx_pos_rel_sat = rx_pos - emitter_pos
    range = np.sqrt(np.sum(rx_pos_rel_sat**2))
    unit_vector = rx_pos_rel_sat / range

    return range, unit_vector


@njit
def compute_range_rate(
    rx_vel: np.array, emitter_vel: np.array, unit_vector: np.array
) -> float:
    """computes range rate from receiver and emitter velocities in ECEF reference frame

    Parameters
    ----------
    rx_vel : np.array
        receiver velocity(s) in ECEF reference frame
    emitter_vel : np.array
        emitter velocity(s) in ECEF reference frame
    unit_vector : np.array
        unit vector in ECEF reference frame

    Returns
    -------
    float
        range rate (m/s)
    """
    rx_vel_rel_sat = rx_vel - emitter_vel
    range_rate = np.sum(rx_vel_rel_sat * unit_vector)

    return range_rate


@njit
def compute_carrier_to_noise(
    range: float,
    transmit_power: float,
    antenna_gain: float,
    fcarrier: float,
    temperature: float = 290,
):
    """computes carrier-to-noise density from free space path loss (derived from Friis equation)

    Parameters
    ----------
    range : float
        range to emitter [m]
    transmit_power : float
        transmit power [dBW]
    antenna_gain : float
        iostropic tx antenna gain [dBi]
    fcarrier : float
        signal carrier frequency [Hz]
    temperature : float
        ambient temperature [K]


    Returns
    -------
    _type_
        carrier-to-noise ratio
    """
    CASCADED_NOISE = 2  # dB-Hz
    BL_AND_QUANTIZATION_NOISE = 1  # dB-Hz

    eirp = transmit_power + antenna_gain
    thermal_noise = 10 * np.log10(BOLTZMANN_CONSTANT * temperature)  # in dBW
    wavelength = SPEED_OF_LIGHT / fcarrier

    cn0 = (
        eirp
        - 20 * np.log10(4 * np.pi * range / wavelength)
        - thermal_noise
        - CASCADED_NOISE
        - BL_AND_QUANTIZATION_NOISE
    )

    return cn0


@njit
def cn02snr(cn0: float, front_end_bw: float = 4e6, noise_figure: float = 0.0):
    snr = cn0 - 10 * np.log10(front_end_bw) - noise_figure  # dB

    return snr


@njit
def snr2cn0(snr: float, front_end_bw: float = 4e6, noise_figure: float = 0.0):
    cn0 = snr + 10 * np.log10(front_end_bw) + noise_figure

    return cn0
