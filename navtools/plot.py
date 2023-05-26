import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def correlation(
    correlation: np.array,
    title: str = "Parallel Correlation",
    context: str = "talk",
    label=None,
) -> plt.axes:
    """Plots correlation output from navtools.dsp.parcorr.

    Parameters
    ----------
    correlation : np.array
        Correlation values across sample lags
    title : str, optional
        Title of plot, by default "Parallel Correlation"
    context : str, optional
        Seaborn context for plot scaling, by default "talk"
    label : _type_, optional
        Plot trace label for legend, by default None

    Returns
    -------
    plt.axes
        Plot axes
    """
    sns.set_context(context)

    _, ax = plt.subplots()
    ax.plot(correlation, label=label)

    plt.title(title)
    plt.xlabel("Sample Lags")
    plt.ylabel("Correlation Magnitude")

    return ax
