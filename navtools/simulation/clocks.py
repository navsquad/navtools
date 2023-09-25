from dataclasses import dataclass


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
