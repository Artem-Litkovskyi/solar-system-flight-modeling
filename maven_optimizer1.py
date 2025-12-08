import numpy as np

from tools.core import *
from tools.instances import *
from tools.tables import make_table
from sim_runner import run_simulations


# Earth Parking Orbit
EARTH_ORBIT_RADIUS = EARTH.radius + 400e3

ESCAPE_DATE = TIMESCALE.utc(2013, 10, 18, 18, 28)

ESCAPE_ANTISOLAR_OFFSET_SPAN = (-10, -55)  # degrees
ESCAPE_ANTISOLAR_OFFSET_NUM = 10

ESCAPE_DELTA_V_SPAN = (4000, 4099)
ESCAPE_DELTA_V_NUM = 10


# Mars Orbit Insertion
INSERT_DELTA_DAYS = TIMESCALE.utc(2014, 9, 22, 2, 24) - TIMESCALE.utc(2013, 10, 18, 18, 28)


def main():
    dates, offsets, delta_vs, results = run_simulations(
        earth_to_mars_transfer,
        (ESCAPE_DATE, ESCAPE_DATE), 1,
        ESCAPE_ANTISOLAR_OFFSET_SPAN, ESCAPE_ANTISOLAR_OFFSET_NUM,
        ESCAPE_DELTA_V_SPAN, ESCAPE_DELTA_V_NUM
    )

    dist = np.zeros((ESCAPE_ANTISOLAR_OFFSET_NUM, ESCAPE_DELTA_V_NUM), dtype=float)

    for i, date in enumerate(dates):
        for j, offset in enumerate(offsets):
            for k, delta_v in enumerate(delta_vs):
                res = results[i, j, k]
                dist[j, k] = res.find_last_point_distance(TIMESCALE, MARS, SUN) - MARS.radius

    make_table('MAVEN_dist.csv', offsets, delta_vs, dist)


# Simulate a single flight with given parameters
def earth_to_mars_transfer(escape_date, antisolar_offset, escape_delta_v):
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
