import numpy as np

from tools.core import *
from tools.instances import *
from tools.tables import make_table
from sim_runner import trajectories_table


EARTH_ORBIT_RADIUS = EARTH.radius + 400e3

ESCAPE_DATES = date_linspace(TIMESCALE, TIMESCALE.utc(2026, 10, 10, 0, 0), TIMESCALE.utc(2026, 11, 0, 0, 0), 5)
ESCAPE_ANTISOLAR_OFFSETS = np.linspace(-34, -70, 10)
ESCAPE_DELTA_VS = np.linspace(4303, 4600, 10)

INSERT_DELTA_DAYS = 400


def main():
    results = trajectories_table(
        earth_to_mars_trajectory, ESCAPE_DATES, ESCAPE_ANTISOLAR_OFFSETS, ESCAPE_DELTA_VS
    )

    dist = np.zeros((ESCAPE_DATES.shape[0], ESCAPE_ANTISOLAR_OFFSETS.size, ESCAPE_DELTA_VS.size), dtype=float)

    for i, date in enumerate(ESCAPE_DATES):
        for j, offset in enumerate(ESCAPE_ANTISOLAR_OFFSETS):
            for k, delta_v in enumerate(ESCAPE_DELTA_VS):
                res = results[i, j, k]
                _, dist[i, j, k] = res.find_closest_point(TIMESCALE, MARS, SUN)

    for i, date in enumerate(ESCAPE_DATES):
        make_table(f'custom_dist_{date.utc_iso()}.csv', ESCAPE_ANTISOLAR_OFFSETS, ESCAPE_DELTA_VS, dist[i])


def earth_to_mars_trajectory(escape_date, antisolar_offset, escape_delta_v):
    antisolar_dir = EARTH.get_relative_position(SUN, escape_date)
    antisolar_dir /= np.linalg.norm(antisolar_dir)

    launch_pos_dir = rotate_vector(antisolar_dir, np.deg2rad(antisolar_offset))
    launch_pos_vector = launch_pos_dir * EARTH_ORBIT_RADIUS

    launch_v = EARTH.get_circular_orbit_velocity(EARTH_ORBIT_RADIUS)
    launch_v_dir = rotate_vector(launch_pos_dir, np.pi / 2)

    earth_pos_at_escape = EARTH.get_relative_position(SUN, escape_date)
    escape_pos_vector = earth_pos_at_escape + launch_pos_vector

    earth_v_at_escape = EARTH.get_relative_velocity(SUN, escape_date)
    escape_v_vector = earth_v_at_escape + launch_v_dir * (launch_v + escape_delta_v)

    insert_date = escape_date + INSERT_DELTA_DAYS

    trans_res = SOLVER.solve(
        [SUN, EARTH, MARS],
        x0=escape_pos_vector[0],
        y0=escape_pos_vector[1],
        vx0=escape_v_vector[0],
        vy0=escape_v_vector[1],
        date_span=(escape_date, insert_date)
    )

    return trans_res


if __name__ == '__main__':
    main()