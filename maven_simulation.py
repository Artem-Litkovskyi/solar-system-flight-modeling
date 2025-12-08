import numpy as np
from tools.core import *
from tools.instances import *
from tools.plots import *


PREFIX = 'MAVEN'

# Earth Parking Orbit
EARTH_ORBIT_RADIUS = EARTH.radius + 400e3
ESCAPE_DATE = TIMESCALE.utc(2013, 10, 18, 18, 28)

# Parameters from optimization
ESCAPE_ANTISOLAR_OFFSET = -31.9732  # degrees
ESCAPE_DELTA_V = 4059.2919

# Mars Orbit Insertion
INSERT_DATE = TIMESCALE.utc(2014, 9, 23, 2, 24)  # Note: adjusted by +24 hours to match expected results
MARS_SEMIMAJOR = (MARS.radius + 382e3 + MARS.radius + 44500e3) / 2

# Expected values
EXPECTED_ESCAPE_DELTA_V = 3770
EXPECTED_INSERT_DELTA_V = 1230.5


def main():
    antisolar_dir = EARTH.get_relative_position(SUN, ESCAPE_DATE)
    antisolar_dir /= np.linalg.norm(antisolar_dir)

    launch_pos_dir = rotate_vector(antisolar_dir, np.deg2rad(ESCAPE_ANTISOLAR_OFFSET))
    launch_pos_vector = launch_pos_dir * EARTH_ORBIT_RADIUS

    launch_v = EARTH.get_circular_orbit_velocity(EARTH_ORBIT_RADIUS)
    launch_v_dir = rotate_vector(launch_pos_dir, np.pi / 2)  # Rotate 90 degrees counter-clockwise
    launch_v_vector = launch_v_dir * launch_v

    # For visualization purposes only (does not affect the rest of the simulation)
    earth_orbit_period = EARTH.get_orbital_period(EARTH_ORBIT_RADIUS)
    launch_date = date_plus_seconds(ESCAPE_DATE, -earth_orbit_period)
    earth_res = SOLVER.solve(
        [EARTH],
        x0=launch_pos_vector[0],
        y0=launch_pos_vector[1],
        vx0=launch_v_vector[0],
        vy0=launch_v_vector[1],
        date_span=(launch_date, ESCAPE_DATE)
    )

    # Earth to Mars transfer
    earth_pos_at_escape = EARTH.get_relative_position(SUN, ESCAPE_DATE)
    escape_pos_vector = earth_pos_at_escape + launch_pos_vector
    
    earth_v_at_escape = EARTH.get_relative_velocity(SUN, ESCAPE_DATE)
    escape_v_vector = earth_v_at_escape + launch_v_dir * (launch_v + ESCAPE_DELTA_V)

    trans_res = SOLVER.solve(
        [SUN, EARTH, MARS],
        x0=escape_pos_vector[0],
        y0=escape_pos_vector[1],
        vx0=escape_v_vector[0],
        vy0=escape_v_vector[1],
        date_span=(ESCAPE_DATE, INSERT_DATE)
    )

    # Mars orbit insertion
    mars_pos_at_insert = MARS.get_relative_position(SUN, INSERT_DATE)
    insert_pos_vector = np.array((trans_res.x[-1], trans_res.y[-1])) - mars_pos_at_insert

    mars_v_at_insert = MARS.get_relative_velocity(SUN, INSERT_DATE)
    insert_v_vector = np.array((trans_res.vx[-1], trans_res.vy[-1])) - mars_v_at_insert
    insert_v_dir = insert_v_vector / np.linalg.norm(insert_v_vector)

    mars_encounter_dist = np.linalg.norm(insert_pos_vector)
    insert_delta_v = MARS.get_elliptical_orbit_velocity(MARS_SEMIMAJOR, mars_encounter_dist) - np.linalg.norm(insert_v_vector)

    mars_orbit_period = MARS.get_orbital_period(MARS_SEMIMAJOR)
    end_date = date_plus_seconds(INSERT_DATE, mars_orbit_period)

    mars_res = SOLVER.solve(
        [MARS],
        x0=insert_pos_vector[0],
        y0=insert_pos_vector[1],
        vx0=insert_v_vector[0] + insert_v_dir[0] * insert_delta_v,
        vy0=insert_v_vector[1] + insert_v_dir[1] * insert_delta_v,
        date_span=(INSERT_DATE, end_date)
    )

    # Output near-Earth results
    trans_dates = seconds_to_date(TIMESCALE, trans_res.t)
    near_earth_res = earth_res + trans_res.with_pos_offset(-EARTH.get_relative_position(SUN, trans_dates))
    plot_trajectory_obj_and_soi(
        TIMESCALE, EARTH, SUN, ESCAPE_DATE,
        near_earth_res.x, near_earth_res.y, earth_res.t.size-1,
        margin_obj=-0.495,
        file_prefix=PREFIX
    )
    print('\n--- Earth orbit ---')
    print(f'Period: {earth_orbit_period / DAYS_TO_SECONDS:.4f} days')
    print(f'Velocity at periapsis (Earth): {launch_v:.4f} m/s')
    print(f'Velocity at periapsis (Sun): {np.linalg.norm(EARTH.rel_v_to_abs(SUN, ESCAPE_DATE, launch_v_vector)):.4f} m/s')

    # Output transfer results
    plot_trajectory_system(
        TIMESCALE, [SUN, EARTH, MARS],
        ESCAPE_DATE, INSERT_DATE,
        trans_res.x, trans_res.y,
        file_prefix=PREFIX
    )

    print('\n--- Earth orbit escape ---')
    print('Escape date:', ESCAPE_DATE.utc_iso())
    print(f'Escape delta v: {ESCAPE_DELTA_V:.4f} m/s')
    print(f'Expected escape delta v: {EXPECTED_ESCAPE_DELTA_V:.4f} m/s')
    print(f'Velocity (Earth): {np.linalg.norm(EARTH.abs_v_to_rel(SUN, ESCAPE_DATE, escape_v_vector)):.4f} m/s')
    print(f'Velocity (Sun): {np.linalg.norm(escape_v_vector):.4f} m/s')


    # Output near-Mars results
    near_mars_res = trans_res.with_pos_offset(-MARS.get_relative_position(SUN, trans_dates)) + mars_res
    plot_trajectory_obj_and_soi(
        TIMESCALE, MARS, SUN, INSERT_DATE,
        near_mars_res.x, near_mars_res.y, -1,
        file_prefix=PREFIX
    )

    print('\n--- Mars encounter ---')
    print(f'Velocity (Mars): {np.linalg.norm(insert_v_vector):.4f} m/s')
    print(f'Velocity (Sun): {np.hypot(trans_res.vx[-1], trans_res.vy[-1]):.4f} m/s')

    print('\n--- Mars orbit insert ---')
    print(f'Delta days: {INSERT_DATE - ESCAPE_DATE:.4f} days')
    print('Insert date:', INSERT_DATE.utc_iso())
    print(f'Mars encounter distance: {mars_encounter_dist:.4f} m')
    print(f'Mars semimajor: {MARS_SEMIMAJOR:.4f} m')
    print(f'Delta v: {insert_delta_v:.4f} m/s')
    print(f'Expected delta v: {EXPECTED_INSERT_DELTA_V:.4f} m/s')

    print('\n--- Relative errors ---')
    print(f'Escape delta v: {relative_error(EXPECTED_ESCAPE_DELTA_V, ESCAPE_DELTA_V):.4f}')
    print(f'Insert delta v: {relative_error(EXPECTED_INSERT_DELTA_V, insert_delta_v):.4f}')


if __name__ == '__main__':
    main()
