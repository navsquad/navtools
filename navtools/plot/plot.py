import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import cartopy
import cartopy.geodesic as cgeo
import cartopy.crs as ccrs

import cartopy.io.img_tiles as cimgt
import io
from urllib.request import urlopen, Request
from PIL import Image
import shapely
from planar import BoundingBox


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


# TODO: make into class to append new data
def geoplot(
    lon,
    lat,
    style="satellite",
    context="talk",
    ax=None,
    single_coordinate_radius=500,
    **kwargs
):
    """This function makes OpenStreetMap satellite or map image with circle and random points.
    Change np.random.seed() number to produce different (reproducable) random patterns of points.
    Also review 'scale' variable"""

    if style == "map":
        ## MAP STYLE
        cimgt.OSM.get_image = (
            image_spoof  # reformat web request for street map spoofing
        )
        img = cimgt.OSM()  # spoofed, downloaded street map
    elif style == "satellite":
        # SATELLITE STYLE
        cimgt.QuadtreeTiles.get_image = (
            image_spoof  # reformat web request for street map spoofing
        )
        img = cimgt.QuadtreeTiles()  # spoofed, downloaded street map
    else:
        print("invalid style")

    ############################################################################
    sns.set_context(context=context)
    if ax is None:
        ax = plt.axes(projection=img.crs)

    # project using coordinate reference system (CRS) of street map
    data_crs = ccrs.PlateCarree()

    # or change scale manually
    # NOTE: scale specifications should be selected based on radius
    # but be careful not have both large scale (>16) and large radius (>1000),
    #  it is forbidden under [OSM policies](https://operations.osmfoundation.org/policies/tiles/)
    # -- 2     = coarse image, select for worldwide or continental scales
    # -- 4-6   = medium coarseness, select for countries and larger states
    # -- 6-10  = medium fineness, select for smaller states, regions, and cities
    # -- 10-12 = fine image, select for city boundaries and zip codes
    # -- 14+   = extremely fine image, select for roads, blocks, buildings

    is_single_coordinate_pair = isinstance(lon, (int, float))
    if is_single_coordinate_pair:
        extent = compute_single_coordinate_extent(
            lon=lon, lat=lat, dist=single_coordinate_radius
        )
        radius = single_coordinate_radius
    else:
        extent, radius = compute_multiple_coordinate_extent(lons=lon, lats=lat)

    # auto-calculate scale
    scale = int(120 / np.log(radius))
    scale = (scale < 20) and scale or 19

    ax.set_extent(extent)  # set extents
    ax.add_image(img, int(scale))  # add OSM with zoom specification

    # add site
    ax.scatter(lon, lat, transform=data_crs, **kwargs)

    gl = ax.gridlines(
        draw_labels=True, crs=data_crs, color="k", lw=0.5, auto_update=True
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER

    return ax


def compute_single_coordinate_extent(lon, lat, dist):
    """This function calculates extent of map
    Inputs:
        lat,lon: location in degrees
        dist: dist to edge from centre
    """

    dist_cnr = np.sqrt(2 * dist**2)
    top_left = cgeo.Geodesic().direct(
        points=(lon, lat), azimuths=-45, distances=dist_cnr
    )[:, 0:2][0]
    bot_right = cgeo.Geodesic().direct(
        points=(lon, lat), azimuths=135, distances=dist_cnr
    )[:, 0:2][0]

    extent = [top_left[0], bot_right[0], bot_right[1], top_left[1]]

    return extent


def compute_multiple_coordinate_extent(lons, lats):
    pairs = [(lon, lat) for lon, lat in zip(lons, lats)]
    bounding_box = BoundingBox(pairs)

    buffer = 0.15 * bounding_box.height  # add 15% buffer

    min_y = bounding_box.min_point.y - buffer
    max_y = bounding_box.max_point.y + buffer

    height = max_y - min_y
    geodetic_radius = height / 2
    width = height

    points = np.array(
        [
            [bounding_box.center.x, bounding_box.center.y],
            [bounding_box.center.x, bounding_box.center.y + geodetic_radius],
        ],
    )
    radius_geometry = shapely.geometry.LineString(points)
    radius = cgeo.Geodesic().geometry_length(geometry=radius_geometry)

    min_x = bounding_box.center.x - width
    max_x = bounding_box.center.x + width

    extent = np.round(
        [
            min_x,
            max_x,
            min_y,
            max_y,
        ],
        decimals=8,
    )

    return extent, radius


def image_spoof(self, tile):
    """this function reformats web requests from OSM for cartopy
    Heavily based on code by Joshua Hrisko at:
        https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
    """

    url = self._image_url(tile)  # get the url of the street map API
    req = Request(url)  # start request
    req.add_header("User-agent", "Anaconda 3")  # add user agent to request
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read())  # get image
    fh.close()  # close url
    img = Image.open(im_data)  # open image with PIL
    img = img.convert(self.desired_tile_form)  # set image format
    return img, self.tileextent(tile), "lower"  # reformat for cartopy


def igeoplot(
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
    is_single_coordinate_pair = isinstance(df.lon, (int, float))
    if is_single_coordinate_pair:
        extent = compute_single_coordinate_extent(
            lon=df.lon, lat=df.lat, dist=single_coordinate_radius
        )
    else:
        extent, _ = compute_multiple_coordinate_extent(lons=df.lon, lats=df.lat)

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
            # family="Courier New, monospace",
            size=18,
        ),
    )
    fig.update_layout(
        mapbox=dict(
            bounds={
                "north": extent[3],
                "south": extent[2],
                "east": extent[1],
                "west": extent[0],
            },
        )
    )

    return fig


def fft(
    fft,
    frequency_range=np.empty(0),
    xlabel: str = "Frequency Index",
    title: str = "FFT",
    context: str = "talk",
    label=None,
) -> plt.axes:
    sns.set_context(context)

    if frequency_range.size != 0:
        xlabel = "Frequency [Hz]"

    _, ax = plt.subplots()
    ax.plot(frequency_range, np.abs(fft), label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("FFT Magnitude")

    return ax
