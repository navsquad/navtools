import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import planar


def correlation(
    correlation: np.array,
    x: np.array = np.empty(0),
    xlabel: str = "Sample Lags",
    title: str = "Parallel Correlation",
    context: str = "talk",
    label=None,
) -> plt.axes:
    """Plots a correlation output

    Parameters
    ----------
    correlation : np.array
        Correlation values across sample lags
    x : np.array, optional
        x values (eg. fractional chips), by default np.empty(0)
    xlabel : str, optional
        x axis label, by default "Sample Lags"
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

    if x.size == 0:
        x = np.arange(0, correlation.size)

    sns.set_context(context)

    _, ax = plt.subplots()
    ax.plot(x, correlation, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Correlation Magnitude")

    return ax


def __compute_zoom_level(latitudes, longitudes):
    all_pairs = []

    for lon, lat in zip(longitudes, latitudes):
        all_pairs.append((lon, lat))

    b_box = planar.BoundingBox(all_pairs)
    if b_box.is_empty:
        return 0, (0, 0)

    area = b_box.height * b_box.width
    zoom = np.interp(
        area,
        [0, 5**-10, 4**-10, 3**-10, 2**-10, 1**-10, 1**-5],
        [20, 17, 16, 15, 14, 7, 5],
    )
    return zoom, b_box.center


def geoplot(
    df: pd.DataFrame,
    source: str = None,
    hover_data: dict = None,
    labels: dict = None,
    title: str = "geoplot",
    color_sequence: px.colors = px.colors.sequential.Rainbow,
) -> go.Figure:
    """Plots latitude and longitude trajectories

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with latitude and longitude labeled as "lat" and "lon", respectively
    source : str, optional
        String associated with source of positions in df (eg. sensors, filters, etc.), by default None
    hover_data : dict, optional
        True/False key value pairs with df entries, by default None
    labels : dict, optional
        Formatted labels for hover_data, by default None
    color_sequence : px.colors, optional
        Plotly color sequence, by default px.colors.sequential.Rainbow


    Returns
    -------
    go.Figure
        geoplot figure
    """
    zoom, center = __compute_zoom_level(df.lat, df.lon)

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color=source,
        hover_data=hover_data,
        labels=labels,
        color_discrete_sequence=color_sequence,
        template="seaborn",
    )
    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ],
            }
        ],
    )
    fig.update_layout(
        title=title,
        font=dict(
            family="Courier New, monospace",
            size=18,
        ),
    )
    fig.update_layout(
        mapbox=dict(
            center=go.layout.mapbox.Center(lat=center.y, lon=center.x), zoom=zoom
        )
    )

    return fig
