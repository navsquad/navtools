from navtools.signals import diy, gps


def get_signal_properties(signal_name: str):
    """factory function that retrieves requested signal properties

    Parameters
    ----------
    signal_name : str
        name of signal

    Returns
    -------
    _type_
        signal properties
    """
    SIGNALS = {"gpsl1ca": gps.L1CA, "freedom": diy.FREEDOM}

    signal_name = "".join([i for i in signal_name if i.isalnum()]).casefold()
    properties = SIGNALS.get(signal_name, gps.L1CA)  # defaults to gps-l1ca

    return properties
