import numpy as np

from tools.core import *
from tools.instances import *
from tools.plots import *


def full_earth_to_mars_mission(
        earth_orbit_radius,
        escape_date,
        escape_antisolar_offset,
        escape_delta_v,
        insert_date,
        mars_orbit_semimajor,
        file_prefix
):
    antisolar_dir = EARTH.get_relative_position(SUN, escape_date)
    antisolar_dir /= np.linalg.norm(antisolar_dir)

    launch_pos_dir = rotate_vector(antisolar_dir, np.deg2rad(escape_antisolar_offset))
    launch_pos_vector = launch_pos_dir * earth_orbit_radius

    launch_v = EARTH.get_circular_orbit_velocity(earth_orbit_radius)
    launch_v_dir = rotate_vector(launch_pos_dir, np.pi / 2)  # Rotate 90 degrees counter-clockwise
    launch_v_vector = launch_v_dir * launch_v

    # For visualization purposes only (does not affect the rest of the simulation)
    earth_orbit_period = EARTH.get_orbital_period(earth_orbit_radius)
    launch_date = date_plus_seconds(escape_date, -earth_orbit_period)
    earth_res = SOLVER.solve(
        [EARTH],
        x0=launch_pos_vector[0],
        y0=launch_pos_vector[1],
        vx0=launch_v_vector[0],
        vy0=launch_v_vector[1],
        date_span=(launch_date, escape_date)
    )

    # Earth to Mars transfer
    earth_pos_at_escape = EARTH.get_relative_position(SUN, escape_date)
    escape_pos_vector = earth_pos_at_escape + launch_pos_vector

    earth_v_at_escape = EARTH.get_relative_velocity(SUN, escape_date)
    escape_v_vector = earth_v_at_escape + launch_v_dir * (launch_v + escape_delta_v)

    trans_res = SOLVER.solve(
        [SUN, EARTH, MARS],
        x0=escape_pos_vector[0],
        y0=escape_pos_vector[1],
        vx0=escape_v_vector[0],
        vy0=escape_v_vector[1],
        date_span=(escape_date, insert_date)
    )

    # Mars orbit insertion
    mars_pos_at_insert = MARS.get_relative_position(SUN, insert_date)
    insert_pos_vector = np.array((trans_res.x[-1], trans_res.y[-1])) - mars_pos_at_insert

    mars_v_at_insert = MARS.get_relative_velocity(SUN, insert_date)
    insert_v_vector = np.array((trans_res.vx[-1], trans_res.vy[-1])) - mars_v_at_insert
    insert_v = np.linalg.norm(insert_v_vector)
    insert_v_dir = insert_v_vector / insert_v

    mars_encounter_dist = np.linalg.norm(insert_pos_vector)
    insert_delta_v = MARS.get_elliptical_orbit_velocity(mars_orbit_semimajor, mars_encounter_dist) - insert_v

    mars_orbit_period = MARS.get_orbital_period(mars_orbit_semimajor)
    end_date = date_plus_seconds(insert_date, mars_orbit_period)

    mars_res = SOLVER.solve(
        [MARS],
        x0=insert_pos_vector[0],
        y0=insert_pos_vector[1],
        vx0=insert_v_vector[0] + insert_v_dir[0] * insert_delta_v,
        vy0=insert_v_vector[1] + insert_v_dir[1] * insert_delta_v,
        date_span=(insert_date, end_date)
    )

    # Output near-Earth results
    trans_dates = seconds_to_date(TIMESCALE, trans_res.t)
    near_earth_res = earth_res + trans_res.with_pos_offset(-EARTH.get_relative_position(SUN, trans_dates))
    plot_trajectory_obj_and_soi(
        TIMESCALE, EARTH, SUN, escape_date,
        near_earth_res.x, near_earth_res.y, earth_res.t.size - 1,
        margin_obj=-0.495,
        file_prefix=file_prefix
    )
    print('\n--- Earth orbit ---')
    print(f'Period: {earth_orbit_period / DAYS_TO_SECONDS:.4f} days')
    print(f'Velocity at periapsis (Earth): {launch_v:.4f} m/s')
    print(
        f'Velocity at periapsis (Sun): {np.linalg.norm(EARTH.rel_v_to_abs(SUN, escape_date, launch_v_vector)):.4f} m/s')

    # Output transfer results
    plot_trajectory_system(
        TIMESCALE, [SUN, EARTH, MARS],
        escape_date, insert_date,
        trans_res.x, trans_res.y,
        file_prefix=file_prefix
    )

    print('\n--- Earth orbit escape ---')
    print('Escape date:', escape_date.utc_iso())
    print(f'Escape delta v: {escape_delta_v:.4f} m/s')
    print(f'Velocity (Earth): {np.linalg.norm(EARTH.abs_v_to_rel(SUN, escape_date, escape_v_vector)):.4f} m/s')
    print(f'Velocity (Sun): {np.linalg.norm(escape_v_vector):.4f} m/s')

    # Output near-Mars results
    near_mars_res = trans_res.with_pos_offset(-MARS.get_relative_position(SUN, trans_dates)) + mars_res
    plot_trajectory_obj_and_soi(
        TIMESCALE, MARS, SUN, insert_date,
        near_mars_res.x, near_mars_res.y, -1,
        file_prefix=file_prefix
    )

    print('\n--- Mars encounter ---')
    print(f'Velocity (Mars): {insert_v:.4f} m/s')
    print(f'Velocity (Sun): {np.hypot(trans_res.vx[-1], trans_res.vy[-1]):.4f} m/s')

    print('\n--- Mars orbit insert ---')
    print(f'Delta days: {insert_date - escape_date:.4f} days')
    print('Insert date:', insert_date.utc_iso())
    print(f'Mars encounter distance: {mars_encounter_dist:.4f} m')
    print(f'Mars semimajor: {mars_orbit_semimajor:.4f} m')
    print(f'Delta v: {insert_delta_v:.4f} m/s')

    return insert_delta_v