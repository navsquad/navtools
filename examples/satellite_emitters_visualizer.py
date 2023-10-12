import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

from navtools.emitters.satellite import SatelliteEmitters
from navtools.plot.satellite import SatelliteEmitterVisualizer

LEAP_SECONDS = 18
INITIAL_TIME = datetime(2023, 10, 1, 18, 40, LEAP_SECONDS)
DURATION = 24 * 3600  # [s]
FSIM = 1 / 350  # [Hz]
RX_POS = np.array([422.9989540813995, -5361.713343705153, 3416.8771615653777]) * 1e3

CONSTELLATIONS = ["GPS", "GALILEO", "IRIDIUM-NEXT"]

constellations = SatelliteEmitters(constellations=CONSTELLATIONS)

### create image ###
vis = SatelliteEmitterVisualizer(
    is_point_light=True, point_size=10, off_screen=False, window_scale=1 / 2
)
emitter_states = constellations.from_datetime(
    datetime=INITIAL_TIME, rx_pos=RX_POS, is_only_visible_emitters=False
)
gps_ecef_pos = np.array(
    [state.pos for state in emitter_states.values() if state.id.startswith("G")]
)
vis.update_constellation(
    datetime=INITIAL_TIME,
    x=gps_ecef_pos[:, 0],
    y=gps_ecef_pos[:, 1],
    z=gps_ecef_pos[:, 2],
    name="GPS",
    color="#19e612",
)
vis.update_receiver_position(
    datetime=INITIAL_TIME,
    x=RX_POS[0],
    y=RX_POS[1],
    z=RX_POS[2],
    name="Auburn",
    color="#ed07ed",
)
### save images ###
# vis.show(
#     screenshot="file.png", transparent_background=True
# )
# # vis.save_graphic(filename="file.svg")
vis.show()


### create animation ###
vis = SatelliteEmitterVisualizer(
    is_point_light=False, point_size=10, off_screen=True, window_scale=1 / 2
)
vis.open_gif(
    filename="SatelliteEmitters_Auburn_24HR.gif",
    fps=20,
    subrectangles=True,
)

datetimes = [
    INITIAL_TIME + timedelta(0, step * (1 / FSIM))
    for step in range(int(DURATION * FSIM))
]

for time in tqdm(datetimes, desc="rendering animation"):
    emitter_states = constellations.from_datetime(
        datetime=time, rx_pos=RX_POS, is_only_visible_emitters=False
    )
    gal_ecef_pos = np.array(
        [state.pos for state in emitter_states.values() if state.id.startswith("E")]
    )
    gps_ecef_pos = np.array(
        [state.pos for state in emitter_states.values() if state.id.startswith("G")]
    )
    ir_ecef_pos = np.array(
        [state.pos for state in emitter_states.values() if state.id.startswith("I")]
    )

    vis.update_constellation(
        datetime=time,
        x=gps_ecef_pos[:, 0],
        y=gps_ecef_pos[:, 1],
        z=gps_ecef_pos[:, 2],
        name="GPS",
        color="#19e612",
    )
    vis.update_constellation(
        datetime=time,
        x=gal_ecef_pos[:, 0],
        y=gal_ecef_pos[:, 1],
        z=gal_ecef_pos[:, 2],
        name="GALILEO",
        color="lightblue",
    )

    vis.update_constellation(
        datetime=time,
        x=ir_ecef_pos[:, 0],
        y=ir_ecef_pos[:, 1],
        z=ir_ecef_pos[:, 2],
        name="IRIDIUM-NEXT",
        color="red",
    )
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


vis.close()
