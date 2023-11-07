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
    if style == "map":
        ## MAP STYLE
        cimgt.OSM.get_image = __image_spoof
        img = cimgt.OSM()  # spoofed, downloaded street map
    elif style == "satellite":
        # SATELLITE STYLE
        cimgt.QuadtreeTiles.get_image = __image_spoof
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
        extent = __compute_single_coordinate_extent(
            lon=lon, lat=lat, distance=single_coordinate_radius
        )
        radius = single_coordinate_radius
    else:
        extent, radius = __compute_multiple_coordinate_extent(lons=lon, lats=lat)

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


def igeoplot(
    df: pd.DataFrame,
    source: str = None,
    hover_data: dict = None,
    labels: dict = None,
    title: str = "Interactive Geoplot",
    size: int = 20,
    single_coordinate_radius: float = 500.0,
    **kwargs
) -> go.Figure:
    # append size of points
    size *= np.ones(df.shape[0])

    # compute bounds
    is_single_coordinate_pair = isinstance(df.lon, (int, float))
    if is_single_coordinate_pair:
        extent = __compute_single_coordinate_extent(
            lon=df.lon, lat=df.lat, distance=single_coordinate_radius
        )
    else:
        extent, _ = __compute_multiple_coordinate_extent(lons=df.lon, lats=df.lat)

    fig = px.scatter_mapbox(
        data_frame=df,
        lat="lat",
        lon="lon",
        color=source,
        hover_data=hover_data,
        labels=labels,
        template="seaborn",
        **kwargs,
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


def __compute_single_coordinate_extent(lon, lat, distance):
    """This function calculates extent of map
    Inputs:
        lat,lon: location in degrees
        dist: dist to edge from centre
    """

    dist_cnr = np.sqrt(2 * distance**2)
    top_left = cgeo.Geodesic().direct(
        points=(lon, lat), azimuths=-45, distances=dist_cnr
    )[:, 0:2][0]
    bot_right = cgeo.Geodesic().direct(
        points=(lon, lat), azimuths=135, distances=dist_cnr
    )[:, 0:2][0]

    extent = [top_left[0], bot_right[0], bot_right[1], top_left[1]]

    return extent


def __compute_multiple_coordinate_extent(lons, lats):
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


def __image_spoof(self, tile):
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
