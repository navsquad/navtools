from dataclasses import dataclass


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
