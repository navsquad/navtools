import numpy as np
from scipy.spatial.transform import Rotation as Rot


def enu2ecefv(east, north, up, lat0, lon0, degrees=True):
    enu = np.column_stack([east, north, up])
    if degrees:
        lat0 = np.radians(lat0)
        lon0 = np.radians(lon0)

    R = np.matrix(
        [
            [-np.sin(lon0), -np.cos(lon0) * np.sin(lat0), np.cos(lon0) * np.cos(lat0)],
            [np.cos(lon0), -np.sin(lon0) * np.sin(lat0), np.sin(lon0) * np.cos(lat0)],
            [0, np.cos(lat0), np.sin(lat0)],
        ],
    )
    R = Rot.from_matrix(R)
    ecef = R.apply(enu)

    return ecef
