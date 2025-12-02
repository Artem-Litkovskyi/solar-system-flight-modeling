import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import skyfield.timelib
from matplotlib.patches import Circle
from classes import *


PLOTS_DIR = 'plots'

DPI = 150
ORBIT_RESOLUTION = 100
ORBIT_PART_DAYS = 15

SUN_SCALE = 15
PLANET_SCALE = 750

COLORS = {
    'sun': 'tab:orange',
    'earth': 'tab:blue',
    'mars': 'tab:red',
}


def get_color(astro_name):
    return COLORS.get(astro_name.strip().lower(), 'tab:gray')


def get_orbit(timescale, obj, central_obj, date0, date1=None):
    if date1 is None:
        period = obj.get_orbital_period(central_obj, date0)
        date1 = date_plus_seconds(date0, period)
    orbit_dates = date_linspace(timescale, date0, date1, ORBIT_RESOLUTION)
    orbit = obj.get_relative_position(central_obj, orbit_dates)
    return orbit


# Full plots
def plot_trajectory_system(
        timescale, astro_objects, date0, date1,
        trajectory_xs, trajectory_ys,
        margin=0.05, file_prefix=''):
    fig, ax = plt.subplots(figsize=(8,8))

    xlim, ylim = _draw_astro_objects(timescale, ax, astro_objects, date0, date1)
    _draw_trajectory_with_radius(ax, trajectory_xs, trajectory_ys)
    # ax.plot(
    #     (trajectory_xs[0], -2*trajectory_xs[0]), (trajectory_ys[0], -2*trajectory_ys[0]),
    #     color='black', linestyle=':', label='Start')
    _prettify_axes(ax, xlim, ylim, margin=margin, legend_loc='best')

    _save('Solar_system.png', prefix=file_prefix)


def plot_trajectory_obj_and_soi(
        timescale, obj, central_obj, orbit_date,
        trajectory_xs, trajectory_ys,
        radius_i=None,
        margin_obj=-0.48, margin_soi=0.05, file_prefix=''
):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

    xlim1, ylim1 = _draw_obj_with_soi(timescale, axs[0], obj, central_obj, orbit_date)
    _draw_trajectory_with_radius(axs[0], trajectory_xs, trajectory_ys, radius_i=radius_i)
    _prettify_axes(axs[0], xlim1, ylim1, margin=margin_obj)

    xlim2, ylim2 = _draw_obj_with_soi(timescale, axs[1], obj, central_obj, orbit_date)
    _draw_trajectory_with_radius(axs[1], trajectory_xs, trajectory_ys, radius_i=radius_i)
    _prettify_axes(axs[1], xlim2, ylim2, margin=margin_soi, legend_loc='upper right')

    _save(f'{obj.name}_and_SOI.png', prefix=file_prefix)


# Drawing helpers
def _draw_astro_objects(
        timescale: skyfield.timelib.Timescale,
        ax: matplotlib.axes._axes.Axes,
        astro_objects: list[AstronomicalObject],
        date0: skyfield.timelib.Time,
        date1: skyfield.timelib.Time,
        central_obj_scale=SUN_SCALE,
        astro_obj_scale=PLANET_SCALE
):
    central_obj = astro_objects[0]

    xlim, ylim = [0, 0], [0, 0]

    # Plot orbits
    for obj in astro_objects[1:]:
        orbit = get_orbit(timescale, obj, central_obj, date0)
        xlim[0] = min(xlim[0], orbit[0].min())
        xlim[1] = max(xlim[1], orbit[0].max())
        ylim[0] = min(ylim[0], orbit[1].min())
        ylim[1] = max(ylim[1], orbit[1].max())
        ax.plot(orbit[0], orbit[1], color=get_color(obj.name), alpha=0.5)

    # Plot positions
    ax.add_patch(Circle(
        (0, 0), central_obj.radius * central_obj_scale,
        fill=True, color=get_color(central_obj.name), label=central_obj.name))

    for obj in astro_objects[1:]:
        lbl = obj.name
        clr = get_color(obj.name)
        ax.add_patch(Circle(
            obj.get_relative_position(central_obj, date0), obj.radius * astro_obj_scale,
            fill=True, color=clr, label=f'{lbl} at Launch'))
        ax.add_patch(Circle(
            obj.get_relative_position(central_obj, date1), obj.radius * astro_obj_scale,
            fill=False, color=clr, linestyle='--', label=f'{lbl} at Arrival'))

    return xlim, ylim


def _draw_obj_with_soi(
        timescale: skyfield.timelib.Timescale,
        ax: matplotlib.axes._axes.Axes,
        obj: AstronomicalObject,
        central_obj: AstronomicalObject,
        orbit_date: skyfield.timelib.Time
):
    clr = get_color(obj.name)

    # Plot object
    ax.add_patch(Circle(
        (0, 0), obj.radius,
        fill=True, color=clr, zorder=5, label=obj.name))

    # Plot object's SOI
    soi_radius = obj.get_sphere_of_influence(central_obj, orbit_date)
    ax.add_patch(Circle(
        (0, 0), soi_radius,
        fill=False, color=clr, linestyle='--', label='Sphere of Influence'))

    # Plot orbit
    orbit = get_orbit(timescale, obj, central_obj, orbit_date - ORBIT_PART_DAYS, orbit_date + ORBIT_PART_DAYS)
    pos = obj.get_relative_position(central_obj, orbit_date)
    print(orbit[:, 0:3])
    print(pos)
    orbit -= pos[:, None]  # Make relative to obj
    print(orbit[:, 0:3])
    ax.plot(orbit[0], orbit[1], color=clr, alpha=0.5, label='Orbit')

    return (-soi_radius, soi_radius), (-soi_radius, soi_radius)


def _draw_trajectory_with_radius(
        ax: matplotlib.axes._axes.Axes,
        trajectory_xs: np.ndarray, trajectory_ys: np.ndarray,
        radius_i: int = None
):
    ax.plot(trajectory_xs, trajectory_ys, color='black', label='Trajectory')

    if radius_i is not None:
        r_x = trajectory_xs[radius_i]
        r_y = trajectory_ys[radius_i]
        r = np.hypot(r_x, r_y)
        ax.plot((0, r_x), (0, r_y),
                color='black', linestyle=':', label=f'Radius ({r / 1000:.1f} km)')


def _prettify_axes(
        ax: matplotlib.axes._axes.Axes,
        xlim: list[float],
        ylim: list[float],
        margin=0.1,
        legend_loc=None
):
    ax.set_aspect('equal', 'box')

    m = margin * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    ax.set_xlim(xlim[0] - m, xlim[1] + m)
    ax.set_ylim(ylim[0] - m, ylim[1] + m)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    if legend_loc is not None:
        ax.legend(loc=legend_loc)


def _save(filename, prefix=''):
    full_filename = filename
    if prefix:
        full_filename = prefix + full_filename
    full_path = os.path.join(PLOTS_DIR, full_filename)

    if not os.path.exists(full_path):
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(full_path, dpi=DPI)

    print(f'Plot saved to {full_path}')