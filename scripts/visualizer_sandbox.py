import navtools.emitters.satellites as ntem
from navtools.constants import WGS84_RADIUS
import numpy as np
import pyvista
from pyvista import examples
from datetime import datetime, timedelta
from tqdm import tqdm
from skyfield.framelib import itrs, ICRS
import astropy.coordinates as coord
import astropy.units as u
import pymap3d as pm
from navtools.plot.simulation import SatelliteEmitterVisualizer
from collections import defaultdict

LEAP_SECONDS = 18
INITIAL_TIME = datetime(2023, 10, 1, 18, 40, LEAP_SECONDS)
DURATION = 24 * 3600  # [s]
FSIM = 1 / 350
RX_POS = np.array([422.9989540813995, -5361.713343705153, 3416.8771615653777]) * 1e3

CONSTELLATIONS = ["STARLINK"]

datetimes = [
    INITIAL_TIME + timedelta(0, step * (1 / FSIM))
    for step in range(int(DURATION * FSIM))
]

constellations = ntem.SatelliteEmitters(constellations=CONSTELLATIONS)

# sl_ecef_pos = np.array(
#     [
#         state.pos
#         for state in multiple_emitters.values()
#         if state.id.startswith("STARLINK")
#     ]
# )


vis = SatelliteEmitterVisualizer(
    is_point_light=False, point_size=10, off_screen=True, window_scale=1 / 2
)
vis.open_gif(
    filename="20231003_SatelliteEmitters_Starlink_Auburn_24HR.gif",
    fps=20,
    subrectangles=True,
)

for time in tqdm(datetimes):
    emitter_states = constellations.from_datetime(
        datetime=time, rx_pos=RX_POS, is_only_visible_emitters=False
    )
    # gal_ecef_pos = np.array(
    #     [state.pos for state in emitter_states.values() if state.id.startswith("E")]
    # )
    gps_ecef_pos = np.array(
        [
            state.pos
            for state in emitter_states.values()
            if state.id.startswith("STARLINK")
        ]
    )
    # ir_ecef_pos = np.array(
    #     [state.pos for state in emitter_states.values() if state.id.startswith("I")]
    # )

    vis.update_constellation(
        datetime=time,
        x=gps_ecef_pos[:, 0],
        y=gps_ecef_pos[:, 1],
        z=gps_ecef_pos[:, 2],
        name="STARLINK",
        color="blue",
    )
    # vis.update_constellation(
    #     datetime=time,
    #     x=gal_ecef_pos[:, 0],
    #     y=gal_ecef_pos[:, 1],
    #     z=gal_ecef_pos[:, 2],
    #     name="GALILEO",
    #     color="lightblue",
    # )

    # vis.update_constellation(
    #     datetime=time,
    #     x=ir_ecef_pos[:, 0],
    #     y=ir_ecef_pos[:, 1],
    #     z=ir_ecef_pos[:, 2],
    #     name="IRIDIUM-NEXT",
    #     color="red",
    # )
    vis.update_receiver_position(
        datetime=time,
        x=RX_POS[0],
        y=RX_POS[1],
        z=RX_POS[2],
        name="Auburn",
        color="#ed07ed",
    )
    vis.add_text(
        text=f"Datetime: {time}", color="#ab5fed", font_size=14, name="datetime"
    )
    vis.render()
    vis.write_frame()


# vis.show(
#     screenshot="20231003_SatelliteEmitters_GEI_Tokyo.png", transparent_background=True
# )
# # vis.save_graphic(filename="20231003_SatelliteEmitters_GEI_Tokyo_Realistic.svg")
vis.close()
# light = pyvista.Light()
# light.set_direction_angle(210, -20)

# earth = examples.planets.load_earth(radius=WGS84_RADIUS)
# earth_texture = examples.load_globe_texture()
# earth.translate((0.0, 0.0, 0.0), inplace=True)
# axes = pyvista.Axes(show_actor=True, line_width=5)
# earth.rotate_z(90, point=axes.origin, inplace=True)

# pl = pyvista.Plotter(lighting="none", off_screen=False)
# cubemap = examples.download_cubemap_space_16k()
# _ = pl.add_actor(cubemap.to_skybox())
# pl.set_environment_texture(cubemap, True)
# pl.add_light(light)
# pl.add_mesh(earth, texture=earth_texture, smooth_shading=True)


# orbits = emitters.from_datetimes(
#     datetimes=datetimes, rx_pos=RX_POS, is_only_visible_emitters=False
# )
# pos = [
#     np.asarray(
#         pm.ecef2eci(state.pos[0], state.pos[1], state.pos[2], time=state.datetime)
#     )
#     for epoch in orbits
#     for state in epoch.values()
# ]

# # pos = [p.transform_to(coord.ICRS) for p in pos]

# spline = pyvista.Spline(pos, n_points=10000)
# # spline.plot(
# #     render_lines_as_tubes=True,
# #     line_width=10,
# #     show_scalar_bar=False,
# # )
# pl.add_mesh(spline, color="red")


# gps_pos = [
#     np.asarray(
#         pm.ecef2eci(state.pos[0], state.pos[1], state.pos[2], time=state.datetime)
#     )
#     for state in all_states.values()
# ]
# gps = pyvista.PolyData(gps_pos)
# pl.add_mesh(
#     gps, color="#39FF14", point_size=7.5, render_points_as_spheres=True, label="GPS"
# )
# au_pos = np.asarray(
#     pm.ecef2eci(x=423756.0, y=-5361363.0, z=3417705.0, time=INITIAL_TIME)
# )
# au = pyvista.PolyData(au_pos)
# pl.add_mesh(
#     au, color="red", point_size=7.5, render_points_as_spheres=True, label="Auburn"
# )
# pl.add_legend(bcolor=None, face=None, size=(0.1, 0.1))

# pl.camera.azimuth = 25
# # plt.camera.elevation = 35
# pl.camera.zoom(1)
# pl.open_gif("simmysimingtonII.gif", fps=30)

# pl.show()

# for date in tqdm(datetimes):
#     all_states = emitters.from_datetime(
#         datetime=date, rx_pos=RX_POS, is_only_visible_emitters=False
#     )
#     gps_pos = np.array(
#         [state.pos for state in all_states.values()]
#     )

#     pl.update_coordinates(gps_pos, mesh=gps, render=False)
#     pl.add_lines(points, color='purple', width=3)
#     # pl.update_coordinates(RX_POS, mesh=rx, render=False)

#     text = pl.add_text(f"Datetime: {date}", color="#ab5fed")

#     pl.write_frame()
#     pl.remove_actor(text)

# pl.close()
