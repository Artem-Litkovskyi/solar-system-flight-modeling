import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


DPI = 150

FULL_ORBIT_RESOLUTION=100
ORBIT_PART_RESOLUTION=10
ORBIT_PART_INTERVAL=10*3600

BIG_SUN_SCALE=15
BIG_PLANET_SCALE=750


def plot_trajectory_system(sun, earth, mars, t0, t1, trajectory_xs, trajectory_ys):
    earth_orbit = earth.get_position(
        np.linspace(0, earth.orbit.orbital_period, FULL_ORBIT_RESOLUTION))
    mars_orbit = mars.get_position(
        np.linspace(0, mars.orbit.orbital_period, FULL_ORBIT_RESOLUTION))

    earth_pos0 = earth.get_position(t0)
    mars_pos0 = mars.get_position(t0)
    earth_pos1 = earth.get_position(t1)
    mars_pos1 = mars.get_position(t1)

    fig, ax = plt.subplots(figsize=(8,8))

    # Orbits
    ax.plot(earth_orbit[:,0], earth_orbit[:,1], color=earth.color, alpha=0.5)
    ax.plot(mars_orbit[:,0], mars_orbit[:,1], color=mars.color, alpha=0.5)

    # Bodies
    ax.add_patch(Circle(
        (0, 0), sun.radius*BIG_SUN_SCALE,
        fill=True, color=sun.color, label=sun.name))
    ax.add_patch(Circle(
        (earth_pos0[0], earth_pos0[1]), earth.radius*BIG_PLANET_SCALE,
        fill=True, color=earth.color, label=f'{earth.name} at Launch'))
    ax.add_patch(Circle(
        (mars_pos0[0], mars_pos0[1]), mars.radius*BIG_PLANET_SCALE,
        fill=True, color=mars.color, label=f'{mars.name} at Launch'))
    ax.add_patch(Circle(
        (earth_pos1[0], earth_pos1[1]), earth.radius * BIG_PLANET_SCALE,
        fill=False, color=earth.color, linestyle='--', label=f'{earth.name} at Arrival'))
    ax.add_patch(Circle(
        (mars_pos1[0], mars_pos1[1]), mars.radius * BIG_PLANET_SCALE,
        fill=False, color=mars.color, linestyle='--', label=f'{mars.name} at Arrival'))

    # Trajectory
    ax.plot(trajectory_xs, trajectory_ys, color='black', label='Trajectory')
    # ax.plot(
    #     (trajectory_xs[0], -2*trajectory_xs[0]), (trajectory_ys[0], -2*trajectory_ys[0]),
    #     color='black', linestyle=':', label='Start')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend()

    # Autoscale: include orbits and SOI
    all_x = np.concatenate([earth_orbit[:,0], mars_orbit[:,0]])
    all_y = np.concatenate([earth_orbit[:,1], mars_orbit[:,1]])
    margin = 0.05 * max(np.ptp(all_x), np.ptp(all_y))
    ax.set_xlim(all_x.min()-margin, all_x.max()+margin)
    ax.set_ylim(all_y.min()-margin, all_y.max()+margin)

    _save('plots/Solar_system.png')


def plot_trajectory_planet_and_soi(planet, planet_soi, trajectory_xs, trajectory_ys, radius_i, radius_length, orbit_t):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

    lim1 = (-planet.radius * 8, planet.radius * 8)
    lim2 = (-planet_soi * 1.05, planet_soi * 1.05)

    _draw(axs[0], planet, planet_soi, orbit_t,
          trajectory_xs, trajectory_ys, lim1, lim1,
          radius_i=radius_i, radius_length=radius_length)
    _draw(axs[1], planet, planet_soi, orbit_t,
          trajectory_xs, trajectory_ys, lim2, lim2,
          radius_i=radius_i, radius_length=radius_length)

    axs[1].legend(loc='upper right')

    _save(f'plots/{planet.name}_and_SOI.png')


def plot_trajectories_planet(
        planet,
        trajectory1_xs, trajectory1_ys,
        trajectory2_xs, trajectory2_ys,
        radius_i, radius_length, orbit_t):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    lim = (-planet.radius * 8, planet.radius * 8)

    _draw(axs[0], planet, -1, orbit_t,
          trajectory1_xs, trajectory1_ys, lim, lim,
          radius_i=radius_i, radius_length=radius_length)
    _draw(axs[1], planet, -1, orbit_t,
          trajectory2_xs, trajectory2_ys, lim, lim,
          radius_i=radius_i, radius_length=radius_length)

    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')

    _save(f'plots/{planet.name}_trajectories.png')


def plot_trajectories_soi(
        planet, planet_soi,
        trajectory1_xs, trajectory1_ys, radius1_i, radius1_length, orbit1_t,
        trajectory2_xs, trajectory2_ys, radius2_i, radius2_length, orbit2_t):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    lim = (-planet_soi * 1.5, planet_soi * 1.5)

    _draw(axs[0], planet, planet_soi, orbit1_t,
          trajectory1_xs, trajectory1_ys, lim, lim,
          radius_i=radius1_i, radius_length=radius1_length)
    _draw(axs[1], planet, planet_soi, orbit2_t,
          trajectory2_xs, trajectory2_ys, lim, lim,
          radius_i=radius2_i, radius_length=radius2_length)

    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')

    _save(f'plots/{planet.name}_SOI_trajectories.png')


def _draw(
        ax, planet, planet_soi, orbit_t,
        trajectory_xs, trajectory_ys, xlim, ylim,
        radius_i=None, radius_length=None):
    ax.add_patch(Circle(
        (0, 0), planet.radius,
        fill=True, color=planet.color, zorder=5, label=planet.name))

    planet_orbit = planet.get_position(  # Relative to (0, 0)
        np.linspace(orbit_t - ORBIT_PART_INTERVAL, orbit_t + ORBIT_PART_INTERVAL, ORBIT_PART_RESOLUTION))
    planet_orbit -= planet.get_position(orbit_t)  # Make relative to planet at time t
    ax.plot(planet_orbit[:, 0], planet_orbit[:, 1], color=planet.color, alpha=0.5, label=f'Orbit')

    if planet_soi > 0:
        ax.add_patch(Circle(
            (0, 0), planet_soi,
            fill=False, color=planet.color, linestyle='--', label='Sphere of Influence'))

    ax.plot(trajectory_xs, trajectory_ys, color='black', label='Trajectory')

    if radius_i is not None:
        ax.plot((0, trajectory_xs[radius_i]), (0, trajectory_ys[radius_i]),
                color='black', linestyle=':', label=f'Radius ({radius_length/1000:.1f} km)')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _save(out_path):
    if not os.path.exists(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    print(f'Plot saved to {out_path}')