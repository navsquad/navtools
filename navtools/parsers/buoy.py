__all__ = ["ceilTime", "Buoy", "BuoyParser"]

import os
import io
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

from navtools.io import ensure_exist
from navtools.conversions import lla2ecef
from scipy.interpolate import CubicSpline

D2R = np.pi / 180
LLA_D2R = np.array([D2R, D2R, 1.0])


def ceilTime(dt: datetime, delta: timedelta) -> datetime:
    """rounds time to next interval of 'delta'

    Parameters
    ----------
    dt : datetime
        datetime object
    delta : timedelta
        timedelta to ceiling datetime to

    Returns
    -------
    datetime
        datetime with ceiling rounding
    """
    return datetime.min + np.ceil((dt - datetime.min) / delta) * delta


@dataclass(frozen=True)
class Buoy:
    id: str
    spline: CubicSpline
    start_time: datetime

    def at(self, times: list | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """generates xyz positions for each buoy at the specified times

        Parameters
        ----------
        times : list | np.ndarray
            list of datetimes

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ECEF position and velocities
        """
        if hasattr(times, "__len__"):
            relative_times = np.array([(t - self.start_time).total_seconds() for t in times])
        else:
            relative_times = (times - self.start_time).total_seconds()
        tx_pos = self.spline(relative_times)
        tx_vel = self.spline(relative_times, 1)
        return tx_pos, tx_vel


class BuoyParser:
    def __init__(self, start_time: datetime, stop_time: datetime, filename: Path | str = None):
        ensure_exist("/tmp/buoy/")
        self.__start_time = start_time
        self.__stop_time = stop_time
        self.__start_time_prev_hr = start_time.replace(second=0, minute=0) - timedelta(hours=1)
        self.__stop_time_next_day = ceilTime(stop_time, timedelta(days=1))
        self.__timedelta = stop_time - start_time
        self.__url_format = "%Y-%m-%dT%H%%3A%M%%3A%SZ"
        self.__str_format = "%Y-%m-%dT%H:%M:%SZ"
        if filename is None:
            self.__out_filename = f"/tmp/buoy/BUOY_PARSER_{start_time.strftime('%Y%m%d-%H%M%S')}"
        else:
            self.__out_filename = filename

    def grab_data(self, list_of_keys: list = ["drifting buoy"]):
        # grab data and put into dataframe
        if os.path.isfile(f"{self.__out_filename}.npz"):
            # data already parsed into spline
            self.__buoy_splines = np.load(f"{self.__out_filename}.npz", allow_pickle=True)["data"]
        else:
            if os.path.isfile(f"{self.__out_filename}.csv"):
                # data already downloaded
                self.__read_file(self, f"{self.__out_filename}.csv")
            else:
                # data need to be downloaded
                self.__download_data()

            # generate buoy spline trajectories
            self.__remove_unnecessary_data(list_of_keys)
            self.__interpolate_trajectories()

            # save the downloaded csv and cubic spline interpolations
            self.__buoy_df.to_csv(f"{self.__out_filename}.csv", index=False)
            np.savez_compressed(f"{self.__out_filename}", data=self.__buoy_splines)

    def grab_emitters(self):
        return self.__buoy_splines

    def __interpolate_trajectories(self):
        # interpolate the path to generate position and velocity
        start = np.datetime64(self.__start_time_prev_hr)
        stop = self.__timedelta.total_seconds()
        second = np.timedelta64(1, "s")

        unique_buoys = np.unique(self.__buoy_df["ID"])
        self.__buoy_splines = np.empty(unique_buoys.size, dtype=Buoy)
        for i, buoy_id in enumerate(unique_buoys):
            buoys = self.__buoy_df[self.__buoy_df["ID"] == buoy_id]
            buoy_time = np.unique(buoys["Time [UTC]"])
            idx = [buoys[buoys["Time [UTC]"] == t].index[0] for t in buoy_time]

            buoys = buoys.loc[idx]
            buoy_time = (buoy_time - start) / second

            # grab ecef position
            buoy_ecef_pos = np.array([buoys["X [m]"].values, buoys["Y [m]"].values, buoys["Z [m]"].values]).T

            # make sure there are the same number of points as in 'relative_time'
            # TODO: figure out other conditions of failure
            if buoy_ecef_pos.shape[0] == 1:
                buoy_ecef_pos = np.vstack((buoy_ecef_pos, buoy_ecef_pos))
                if buoys["Time [UTC]"].values[0] > start:
                    buoy_time = np.array([0, buoy_time[0]])
                else:
                    buoy_time = np.array([buoy_time[0], stop])
            if np.all(buoy_time == 0):
                buoy_time = np.array(
                    (
                        pd.date_range(
                            start=self.__start_time_prev_hr,
                            end=self.__stop_time_next_day,
                            freq=timedelta(seconds=3600 * 24 / (buoy_time.size - 1)),
                        )
                        - self.__start_time_prev_hr
                    ).total_seconds()
                )
            self.__buoy_splines[i] = Buoy(
                id=buoy_id,
                spline=CubicSpline(x=buoy_time, y=buoy_ecef_pos),
                start_time=self.__start_time_prev_hr,
            )

    def __download_data(self):
        start_time = self.__start_time_prev_hr.strftime(self.__url_format)
        stop_time = self.__stop_time_next_day.strftime(self.__url_format)

        # currently defaults to only retrieving "generic" buoy data
        # TODO: fix this for modularity
        archive_url = (
            "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/OSMC_RealTime.csv?"
            + "platform_code%2Cplatform_type%2Ccountry%2Ctime%2Clatitude%2Clongitude%2Cobservation_depth&"
            + f"time%3E={start_time}&time%3C={stop_time}&orderBy(%22platform_code%2Ctime%22)"
        )

        r = requests.get(archive_url)
        if r.status_code == 200:
            self.__buoy_df = pd.read_csv(
                io.StringIO(r.text),
                delimiter=",",
                skiprows=2,
                names=[
                    "ID",
                    "Type",
                    "Country",
                    "Time [UTC]",
                    "Latitude [deg]",
                    "Longitude [deg]",
                    "Depth [m]",
                ],
            )
        else:
            raise FileNotFoundError()

    def __read_file(self, filename: str):
        # read directly from file
        self.__buoy_df = pd.read_csv(
            filename,
            delimiter=",",
            skiprows=2,
            names=[
                "ID",
                "Type",
                "Country",
                "Time [UTC]",
                "Latitude [deg]",
                "Longitude [deg]",
                "Depth [m]",
            ],
        )

    def __remove_unnecessary_data(self, list_of_keys: list):
        # remove unnecessary time points
        self.__buoy_df["Time [UTC]"] = pd.to_datetime(self.__buoy_df["Time [UTC]"], format=self.__str_format)

        idx = np.logical_and(
            self.__buoy_df["Time [UTC]"] >= self.__start_time_prev_hr,
            self.__buoy_df["Time [UTC]"] <= self.__stop_time_next_day,
        )
        self.__buoy_df = self.__buoy_df[idx]
        self.__buoy_df.reset_index(drop=True, inplace=True)

        # only grab desired portion of downloaded data
        idx = [
            i
            for i in self.__buoy_df.index
            if any(k.lower() in self.__buoy_df["Type"][i].casefold() for k in list_of_keys)
        ]
        self.__buoy_df = self.__buoy_df.iloc[idx]
        self.__buoy_df.reset_index(drop=True, inplace=True)

        # convert lla to ecef
        L = self.__buoy_df["Type"].size
        x, y, z = np.zeros(L), np.zeros(L), np.zeros(L)
        for lat, lon, alt, i in zip(
            self.__buoy_df["Latitude [deg]"],
            self.__buoy_df["Longitude [deg]"],
            self.__buoy_df["Depth [m]"],
            self.__buoy_df.index,
        ):
            x[i], y[i], z[i] = lla2ecef(np.array([lat, lon, alt]) * LLA_D2R)

        # overwrite lla data with ecef data
        self.__buoy_df.rename(
            columns={
                "Latitude [deg]": "X [m]",
                "Longitude [deg]": "Y [m]",
                "Depth [m]": "Z [m]",
            },
            inplace=True,
        )
        self.__buoy_df["X [m]"] = x
        self.__buoy_df["Y [m]"] = y
        self.__buoy_df["Z [m]"] = z


# start = datetime(2024, 1, 1, 12, 0, 0)
# stop = start + timedelta(seconds=1)

# bp = BuoyParser(start, stop)
# bp.grab_data()
# buoys = bp.grab_emitters()

# for b in buoys:
#     print(f"{b.id}, Start = {b.at(start)[0]}, Stop = {b.at(stop)[0]}")
