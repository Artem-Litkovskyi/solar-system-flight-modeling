import numpy as np
from tools.core import *
from tools.instances import *
from tools.plots import *


PREFIX = 'MAVEN'

# Earth Parking Orbit
EARTH_ORBIT_RADIUS = EARTH.radius + 200e3
ESCAPE_DATE = TIMESCALE.utc(2013, 11, 18, 18, 28+27)
ESCAPE_ANTISOLAR_OFFSET = -52  # degrees
ESCAPE_DELTA_V = 4000

# Mars Orbit Insertion
INSERT_DATE = TIMESCALE.utc(2014, 9, 22, 2, 24)
INSERT_DELTA_V = 1230.5

# Expected values
MARS_ORBIT_PERIAPSIS = MARS.radius + 382e3
MARS_ORBIT_APOAPSIS = MARS.radius + 44500e3


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

    # Output near-Earth results
    trans_dates = seconds_to_date(TIMESCALE, trans_res.t)
    near_earth_res = earth_res + trans_res.with_pos_offset(-EARTH.get_relative_position(SUN, trans_dates))
    plot_trajectory_obj_and_soi(
        TIMESCALE, EARTH, SUN, ESCAPE_DATE,
        near_earth_res.x, near_earth_res.y, earth_res.t.size,
        margin_obj=-0.495,
        file_prefix=PREFIX
    )
    print('--- Earth orbit ---')
    print('Escape date:', ESCAPE_DATE.utc_iso())
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
    print('--- Earth orbit escape ---')
    print(f'Velocity (Earth): {np.linalg.norm(EARTH.abs_v_to_rel(SUN, ESCAPE_DATE, escape_v_vector)):.4f} m/s')
    print(f'Velocity (Sun): {np.linalg.norm(escape_v_vector):.4f} m/s')

if __name__ == '__main__':
    main()
