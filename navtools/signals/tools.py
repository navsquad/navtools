from dataclasses import dataclass

from navtools.signals import gps


@dataclass(frozen=True)
class PhaseShiftKeyedSignal:
    fcarrier: float

    fbit_data: float
    msg_length_data: int
    fchip_data: float
    code_length_data: int
    prn_generator_data: any

    fbit_pilot: float = None
    msg_length_pilot: int = None
    fchip_pilot: float = None
    code_length_pilot: float = None
    prn_generator_pilot: any = None


def get_signal_properties(signal_type: str):
    signal_type = "".join([i for i in signal_type if i.isalnum()]).casefold()

    match signal_type:
        case "gpsl1ca":
            return gps.GPS_L1CA
