from classes import *
from plots import *


START_TIME = 284.8*24*3600  # seconds


def main():
    solver = FlightSolver()

    # Data
    sun = AstronomicalObject(
        name='Sun', color='tab:orange',
        mass=1.9885e30, radius=695700000)

    earth_orbit = Orbit(
        central_body=sun,
        semimajor_axis=149598023000, eccentricity=0.0167086,
        periapsis_argument=np.deg2rad(114.20783),
        orbital_period=365.256363004*24*3600)
    earth = AstronomicalObject(
        name='Earth', color='tab:blue',
        mass=5.972168e24, radius=6371000, orbit=earth_orbit)

    mars_orbit = Orbit(
        central_body=sun,
        semimajor_axis=227939366000, eccentricity=0.0934,
        periapsis_argument=np.deg2rad(286.5),
        orbital_period=686.980*24*3600)
    mars = AstronomicalObject(
        name='Mars', color='tab:red',
        mass=6.419e23, radius=3389500, orbit=mars_orbit)

    earth_soi = earth.get_sphere_of_influence()
    mars_soi = mars.get_sphere_of_influence()

    print(f'Earth SOI: {earth_soi/1000:0.3f} km')
    print(f'Mars SOI: {mars_soi/1000:0.3f} km')

    # Flight scenarios
    earth_to_mars_transfer(solver, sun, earth, earth_soi, mars, mars_soi)
    earth_fall_and_ellipse(solver, earth)
    mars_undershoot_and_overshoot(solver, sun, earth, mars, mars_soi)


def earth_to_mars_transfer(solver, sun, earth, earth_soi, mars, mars_soi):
    # The flight will start at a geostationary orbit (GSO)
    gso_r = 42164000
    gso_v = earth.get_orbital_velocity(gso_r)
    gso_period = 2 * np.pi * gso_r / gso_v
    transfer_start_t = START_TIME + gso_period

    sun_to_earth = earth.get_position(transfer_start_t)
    sun_to_earth /= np.linalg.norm(sun_to_earth)

    start_dir = np.array([-sun_to_earth[1], sun_to_earth[0]])  # 90° CCW rotation
    gso_v_vect = start_dir * gso_v

    print(f'GSO velocity (rel): {gso_v:0.3f} m/s')
    print(f'GSO velocity (abs): {np.linalg.norm(earth.rel_v_to_abs(START_TIME, gso_v_vect)):0.3f} m/s')
    print(f'GSO period: {gso_period / 3600:0.3f} hours')

    gso_res = solver.solve(
        astro_objects=earth,  # Only Earth gravity. The coordinates are relative to Earth
        x0=sun_to_earth[0] * gso_r,
        y0=sun_to_earth[1] * gso_r,
        vx0=gso_v_vect[0],
        vy0=gso_v_vect[1],
        t_span=(START_TIME, transfer_start_t)  # make one full turn
    )

    # Transfer from Earth to Mars
    escape_v = 35085.5 + 100
    transfer_end_t = transfer_start_t + 234.7*24*3600

    escape_v_vect = start_dir * escape_v

    print(f'GSO escape velocity (rel): {np.linalg.norm(earth.abs_v_to_rel(transfer_start_t, escape_v_vect)):0.3f} m/s')
    print(f'GSO escape velocity (abs): {escape_v:0.3f} m/s')

    transfer_res = solver.solve(
        astro_objects=[sun, earth, mars],  # The coordinates are relative to Sun (0, 0)
        x0=earth.get_position(transfer_start_t)[0] + gso_res.x[-1],
        y0=earth.get_position(transfer_start_t)[1] + gso_res.y[-1],
        vx0=escape_v_vect[0],
        vy0=escape_v_vect[1],
        t_span=(transfer_start_t, transfer_end_t)
    )

    earth_periapsis_i, earth_periapsis, _ = transfer_res.find_periapsis(earth, 10)
    # mars_periapsis_i, mars_periapsis, _ = transfer_res.find_periapsis(mars, 10)  # Was somewhat useful for tuning
    # transfer_res = transfer_res[:mars_periapsis_i+1]
    # transfer_end_t = transfer_res.t[-1]
    # days_to_periapsis = (transfer_res.t[-1] - transfer_start_t) / (24 * 3600)
    # print(f'Transfer start -> the closest approach to Mars: {days_to_periapsis:0.3f} days')

    enc_v_vect = np.array([transfer_res.vx[-1], transfer_res.vy[-1]])

    print(f'Mars encounter velocity (rel): {np.linalg.norm(mars.abs_v_to_rel(transfer_end_t, enc_v_vect)):0.3f} m/s')
    print(f'Mars encounter velocity (abs): {np.linalg.norm(enc_v_vect):0.3f} m/s')

    # Get to Mars orbit
    mars_rel_x = transfer_res.x[-1] - mars.get_position(transfer_end_t)[0]
    mars_rel_y = transfer_res.y[-1] - mars.get_position(transfer_end_t)[1]

    mars_orbit_r = np.hypot(mars_rel_x, mars_rel_y)
    mars_orbit_v = mars.get_orbital_velocity(mars_orbit_r)
    mars_orbit_period = 2 * np.pi * mars_orbit_r / mars_orbit_v

    mars_dir = enc_v_vect / np.linalg.norm(enc_v_vect)
    mars_orbit_v_vect = mars_dir * mars_orbit_v

    print(f'Mars orbit velocity (rel): {mars_orbit_v:0.3f} m/s')
    print(f'Mars orbit velocity (abs): {np.linalg.norm(mars.rel_v_to_abs(transfer_end_t, mars_orbit_v_vect)):0.3f} m/s')
    print(f'Mars orbit period: {mars_orbit_period / 3600:0.3f} hours')

    mars_res = solver.solve(
        astro_objects=mars,
        x0=mars_rel_x,
        y0=mars_rel_y,
        vx0=mars_orbit_v_vect[0],
        vy0=mars_orbit_v_vect[1],
        t_span=(transfer_end_t, transfer_end_t + mars_orbit_period)
    )

    # Plot Earth orbit and escape
    near_earth_result = gso_res + transfer_res.with_offset(-earth.get_position(transfer_start_t))
    plot_trajectory_planet_and_soi(
        earth, earth_soi, near_earth_result.x, near_earth_result.y,
        earth_periapsis_i, earth_periapsis, transfer_end_t)

    # Plot transfer trajectory in a Solar System scale
    plot_trajectory_system(sun, earth, mars, transfer_start_t, transfer_end_t, transfer_res.x, transfer_res.y)

    # Plot Mars encounter and orbit
    near_mars_result = transfer_res.with_offset(-mars.get_position(transfer_end_t)) + mars_res
    transfer_end_i = transfer_res.t.shape[0]
    transfer_end_dist = np.hypot(near_mars_result.x[transfer_end_i], near_mars_result.y[transfer_end_i])
    plot_trajectory_planet_and_soi(
        mars, mars_soi, near_mars_result.x, near_mars_result.y,
        transfer_end_i, transfer_end_dist, transfer_end_t)  # mars_periapsis_i, mars_periapsis)


def earth_fall_and_ellipse(solver, earth):
    # The flight will start at a geostationary orbit (GSO)
    earth_orbit_r = 20000000
    earth_orbit_v_a = 2000
    earth_orbit_v_b = 5500
    earth_orbit_period_a = 1.5*3600
    earth_orbit_period_b = 24*3600

    sun_to_earth = earth.get_position(START_TIME)
    sun_to_earth /= np.linalg.norm(sun_to_earth)

    start_dir = np.array([-sun_to_earth[1], sun_to_earth[0]])

    earth_res_a = solver.solve(
        astro_objects=earth,
        x0=sun_to_earth[0] * earth_orbit_r,
        y0=sun_to_earth[1] * earth_orbit_r,
        vx0=start_dir[0] * earth_orbit_v_a,
        vy0=start_dir[1] * earth_orbit_v_a,
        t_span=(START_TIME, START_TIME + earth_orbit_period_a)
    )

    earth_res_b = solver.solve(
        astro_objects=earth,
        x0=sun_to_earth[0] * earth_orbit_r,
        y0=sun_to_earth[1] * earth_orbit_r,
        vx0=start_dir[0] * earth_orbit_v_b,
        vy0=start_dir[1] * earth_orbit_v_b,
        t_span=(START_TIME, START_TIME + earth_orbit_period_b)
    )

    # Plot falling trajectory and elliptical trajectory
    plot_trajectories_planet(
        earth,
        earth_res_a.x, earth_res_a.y, earth_res_b.x, earth_res_b.y,
        0, earth_orbit_r, START_TIME)


def mars_undershoot_and_overshoot(solver, sun, earth, mars, mars_soi):
    # The flight will start at a geostationary orbit (GSO)
    gso_r = 42164000
    gso_v = earth.get_orbital_velocity(gso_r)
    gso_period = 2 * np.pi * gso_r / gso_v
    transfer_start_t = START_TIME + gso_period

    sun_to_earth = earth.get_position(transfer_start_t)
    sun_to_earth /= np.linalg.norm(sun_to_earth)

    start_dir = np.array([-sun_to_earth[1], sun_to_earth[0]])

    gso_res = solver.solve(
        astro_objects=earth,
        x0=sun_to_earth[0] * gso_r,
        y0=sun_to_earth[1] * gso_r,
        vx0=start_dir[0] * gso_v,
        vy0=start_dir[1] * gso_v,
        t_span=(START_TIME, transfer_start_t)
    )

    # Transfer from Earth to Mars
    escape_v_a = 35081.7
    escape_v_b = 35089.3
    transfer_end_t = transfer_start_t + 300*24*3600

    escape_v_vect_a = start_dir * escape_v_a
    escape_v_vect_b = start_dir * escape_v_b

    print(f'GSO escape velocity A (rel): {np.linalg.norm(earth.abs_v_to_rel(transfer_start_t, escape_v_vect_a)):0.3f} m/s')
    print(f'GSO escape velocity A (abs): {escape_v_a:0.3f} m/s')
    print(f'GSO escape velocity B (rel): {np.linalg.norm(earth.abs_v_to_rel(transfer_start_t, escape_v_vect_b)):0.3f} m/s')
    print(f'GSO escape velocity B (abs): {escape_v_b:0.3f} m/s')

    transfer_res_a = solver.solve(
        astro_objects=[sun, earth, mars],
        x0=earth.get_position(transfer_start_t)[0] + gso_res.x[-1],
        y0=earth.get_position(transfer_start_t)[1] + gso_res.y[-1],
        vx0=escape_v_vect_a[0],
        vy0=escape_v_vect_a[1],
        t_span=(transfer_start_t, transfer_end_t)
    )

    transfer_res_b = solver.solve(
        astro_objects=[sun, earth, mars],
        x0=earth.get_position(transfer_start_t)[0] + gso_res.x[-1],
        y0=earth.get_position(transfer_start_t)[1] + gso_res.y[-1],
        vx0=escape_v_vect_b[0],
        vy0=escape_v_vect_b[1],
        t_span=(transfer_start_t, transfer_end_t)
    )

    mars_periapsis_i_a, mars_periapsis_a, _ = transfer_res_a.find_periapsis(mars, 5)
    mars_periapsis_i_b, mars_periapsis_b, _ = transfer_res_b.find_periapsis(mars, 5)
    mars_periapsis_t_a = transfer_res_a.t[mars_periapsis_i_a]
    mars_periapsis_t_b = transfer_res_b.t[mars_periapsis_i_b]

    # Plot Mars undershoot and overshoot
    near_mars_result_a = transfer_res_a.with_offset(-mars.get_position(mars_periapsis_t_a))
    near_mars_result_b = transfer_res_b.with_offset(-mars.get_position(mars_periapsis_t_b))
    plot_trajectories_soi(
        mars, mars_soi,
        near_mars_result_a.x, near_mars_result_a.y, mars_periapsis_i_a, mars_periapsis_a, mars_periapsis_t_a,
        near_mars_result_b.x, near_mars_result_b.y, mars_periapsis_i_b, mars_periapsis_b, mars_periapsis_t_b)


if __name__ == '__main__':
    main()
