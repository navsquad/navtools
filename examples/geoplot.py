import navtools as nt
import numpy as np
import pandas as pd

ublox_lat = 32 + 0.005 * np.random.randn(100)
ublox_lon = -85 + 0.005 * np.random.randn(100)
ublox_alt = 201 + 40 * np.random.randn(100)
ublox_df = pd.DataFrame({"lat": ublox_lat, "lon": ublox_lon, "alt": ublox_alt})
ublox_df["source"] = "ublox"

novatel_lat = 32 + 0.005 * np.random.randn(100)
novatel_lon = -85 + 0.005 * np.random.randn(100)
novatel_alt = 201 + 40 * np.random.randn(100)
novatel_df = pd.DataFrame({"lat": novatel_lat, "lon": novatel_lon, "alt": novatel_alt})
novatel_df["source"] = "NovAtel"

gps_df = pd.concat([ublox_df, novatel_df])

gps_fig = nt.plot.geoplot(
    gps_df,
    source="source",
    hover_data={"alt": True},
    labels={
        "lat": "Latitude [degs]",
        "lon": "Longitude [degs]",
        "alt": "Altitude [m]",
        "source": "Source",
    },
)
gps_fig.show()
