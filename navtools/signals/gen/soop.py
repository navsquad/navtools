"""
|============================================ soop.py =============================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     navtools/signals/gen/soop.py                                                         |
|   @brief    Default signal of opportunity classes.                                               |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     March 2024                                                                           |
|                                                                                                  |
|==================================================================================================|
"""

__all__ = []

from navtools.signals.signals import SignalOfOpportunity
from navtools.constants import SPEED_OF_LIGHT

# * LEO Signals of Opprotunity
IRIDIUM = SignalOfOpportunity(
    fcarrier=1626.25e6,  # nominal
    wavelength=SPEED_OF_LIGHT / 1626.25e6,
    transmit_power=12,  # ~21-28 nominal EIRP [dBw]
    transmit_antenna_gain=13,
)

ORBCOMM = SignalOfOpportunity(
    fcarrier=400.1e6,
    wavelength=SPEED_OF_LIGHT / 400.1e6,
    transmit_power=11,  # ~20-27 nominal EIRP [dBw]
    transmit_antenna_gain=13,
)

GLOBALSTAR = SignalOfOpportunity(
    fcarrier=2491.77e6,
    wavelength=SPEED_OF_LIGHT / 2491.77e6,
    transmit_power=15,  # ~26-30 nominal EIRP [dBw]
    transmit_antenna_gain=13,
)

ONEWEB = SignalOfOpportunity(
    fcarrier=19300e6,  # 10700e6, 12750e6, 14000e6, 19700e6, 27500e6
    wavelength=SPEED_OF_LIGHT / 19300e6,
    transmit_power=20,  # ~22-44 nominal EIRP (max 51.6) [dBw]
    transmit_antenna_gain=13,
)

STARLINK = SignalOfOpportunity(
    fcarrier=10700e6,
    wavelength=SPEED_OF_LIGHT / 10700e6,
    transmit_power=19.5,  # ~21-44 nominal EIRP (max 66.89) [dBw]
    transmit_antenna_gain=13,
)

BUOY = SignalOfOpportunity(
    fcarrier=162.475e6,  # 162.400, 162.425, 162.450, 162.475, 162.500, 162.525, 162.550 MHz
    wavelength=SPEED_OF_LIGHT / 162.475e6,
    transmit_power=1,
    transmit_antenna_gain=5,
)
