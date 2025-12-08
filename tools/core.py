from typing import Self, Sequence
import numpy as np
from scipy.integrate import solve_ivp
from skyfield.elementslib import osculating_elements_of
import skyfield.positionlib
import skyfield.timelib


__all__ = [
    'G', 'DAYS_TO_SECONDS', 'HOURS_TO_SECONDS',
    'date_linspace', 'date_plus_seconds', 'seconds_to_date',
    'rotate_vector', 'relative_error',
    'AstronomicalObject',
    'FlightSolver', 'FlightSolverResult'
]


G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
DAYS_TO_SECONDS = 86400.0
HOURS_TO_SECONDS = 3600.0


def date_linspace(timescale, date0, date1, num_points):
    jd_array = np.linspace(date0.tt, date1.tt, num_points)
    return timescale.tt_jd(jd_array)


def date_plus_seconds(date, seconds):
    return date + seconds / DAYS_TO_SECONDS


def seconds_to_date(timescale, t):
    return timescale.tt_jd(t / DAYS_TO_SECONDS)


def rotate_vector(v, theta):
    r = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return r @ v


def relative_error(a, b):
    return np.abs(a - b) / np.abs(b)


def _zeros_2d(date):
    if date.shape:
        return np.zeros((2, date.shape))
    return np.zeros(2)


class AstronomicalObject:
    def __init__(self, ephemeris, timescale, name, mass, radius):
        self.ephemeris = ephemeris
        self.timescale = timescale
        self.name = name
        self.mass = mass
        self.radius = radius

    # Mass dependent
    def get_gravity_acceleration(self, relative_position: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(relative_position)
        multiplier = G * self.mass / r ** 3
        return -relative_position * multiplier

    def get_sphere_of_influence(self, observer_object: Self, date) -> float | np.ndarray:
        if observer_object is self:
            raise ValueError('Sphere of influence is not defined if observer_object is the same object.')
        r = np.linalg.norm(self.get_relative_position(observer_object, date))
        return r * (self.mass / observer_object.mass) ** (2 / 5)

    def get_circular_orbit_velocity(self, radius) -> float | np.ndarray:
        return np.sqrt(G * self.mass / radius)

    def get_elliptical_orbit_velocity(self, semimajor, current_radius) -> float | np.ndarray:
        return np.sqrt(G * self.mass * (2 / current_radius - 1 / semimajor))

    def get_orbital_period(self, semimajor) -> float | np.ndarray:
        return 2 * np.pi * np.sqrt(semimajor ** 3 / (G * self.mass))

    # Mass independent
    def get_object_orbital_period(self, observer_object: Self, date) -> float | np.ndarray:
        if observer_object is self:
            raise ValueError('Orbital period is not defined if observer_object is the same object.')
        rel_pos = self._get_pos_data(observer_object, date)
        elements = osculating_elements_of(rel_pos)
        return elements.period_in_days * DAYS_TO_SECONDS

    def get_relative_position(self, observer_object: Self, date) -> np.ndarray:
        if observer_object is self:
            return _zeros_2d(date)
        pos_m = self._get_pos_data_2d(observer_object, date).xyz.m
        return pos_m[:2]

    def get_relative_velocity(self, observer_object: Self, date) -> np.ndarray:
        if observer_object is self:
            return _zeros_2d(date)
        vel_mps = self._get_pos_data_2d(observer_object, date).velocity.m_per_s
        return vel_mps[:2]

    def _get_pos_data(self, observer_object: Self, date) -> skyfield.positionlib.Barycentric:
        observer = self.ephemeris[observer_object.name]
        target = self.ephemeris[self.name]
        pos_data = (target - observer).at(date)
        return pos_data

    def _get_pos_data_2d(self, observer_object: Self, date) -> skyfield.positionlib.Barycentric:
        pos_data = self._get_pos_data(observer_object, date)

        # Force to 2D: zero out z coordinate
        pos_data.xyz.au[2] *= 0
        pos_data.velocity.au_per_d[2] *= 0

        return pos_data

    # Utils
    def rel_v_to_abs(self, observer_object: Self, date, rel_velocity: np.ndarray) -> np.ndarray:
        return rel_velocity + self.get_relative_velocity(observer_object, date)

    def abs_v_to_rel(self, observer_object: Self, date, abs_velocity: np.ndarray) -> np.ndarray:
        return abs_velocity - self.get_relative_velocity(observer_object, date)


class FlightSolverResult:
    def __init__(self, t, x, y, vx, vy):
        self.t = t
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def __add__(self, other):
        t = np.concatenate((self.t, other.t))
        x = np.concatenate((self.x, other.x))
        y = np.concatenate((self.y, other.y))
        vx = np.concatenate((self.vx, other.vx))
        vy = np.concatenate((self.vy, other.vy))

        return FlightSolverResult(t, x, y, vx, vy)

    def __getitem__(self, key):
        return FlightSolverResult(self.t[key], self.x[key], self.y[key], self.vx[key], self.vy[key])

    def with_pos_offset(self, offset: np.ndarray):
        return FlightSolverResult(self.t, self.x + offset[0], self.y + offset[1], self.vx, self.vy)

    def find_closest_point(
            self, timescale: skyfield.timelib.Timescale,
            astro_obj: AstronomicalObject | None = None,
            central_obj: AstronomicalObject | None = None
    ):
        if astro_obj is None:
            rel_pos = np.array((self.x, self.y))
        else:
            spacecraft_trajectory = np.array((self.x, self.y))
            astro_obj_trajectory = astro_obj.get_relative_position(central_obj, seconds_to_date(timescale, self.t))
            rel_pos = spacecraft_trajectory - astro_obj_trajectory

        sqr_dist = rel_pos[0] ** 2 + rel_pos[1] ** 2
        best_i = np.argmin(sqr_dist)

        return best_i, np.sqrt(sqr_dist[best_i])

    def find_last_point_distance(
            self, timescale: skyfield.timelib.Timescale,
            astro_obj: AstronomicalObject | None = None,
            central_obj: AstronomicalObject | None = None
    ):
        spacecraft_pos = np.array((self.x[-1], self.y[-1]))
        astro_obj_pos = astro_obj.get_relative_position(central_obj, seconds_to_date(timescale, self.t[-1]))
        rel_pos = spacecraft_pos - astro_obj_pos
        dist = np.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2)
        return dist


class FlightSolver:
    def __init__(self, timescale: skyfield.timelib.Timescale, method='RK45', rtol=1e-11, atol=1e-8):
        self.timescale = timescale
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def solve(
            self, astro_objects: Sequence[AstronomicalObject],
            x0: float, y0: float, vx0: float, vy0: float,
            date_span: Sequence[skyfield.timelib.Time],
            points_per_day: int | None = None
    ) -> FlightSolverResult:
        """
        Calculates spacecraft trajectory. Coordinates are relative to the first AstronomicalObject.

        :param astro_objects: target AstronomicalObjects
        :param x0: spacecraft x position
        :param y0: spacecraft y position
        :param vx0: spacecraft x velocity
        :param vy0: spacecraft y velocity
        :param date_span: interval of integration (date0, date1)
        :param points_per_day: number of trajectory points per day
        """

        t_start = float(date_span[0].tt * DAYS_TO_SECONDS)
        t_end = float(date_span[1].tt * DAYS_TO_SECONDS)

        t_eval = None
        if points_per_day:
            delta_seconds = t_end - t_start
            t_eval = np.linspace(t_start, t_end, int(delta_seconds / DAYS_TO_SECONDS * points_per_day))

        sol = solve_ivp(  # type: ignore
            fun=self._flight_ode,
            t_span=(t_start, t_end),
            y0=[x0, y0, vx0, vy0],
            method=self.method,
            t_eval=t_eval,
            args=astro_objects,
            rtol=self.rtol,
            atol=self.atol
        )

        return FlightSolverResult(sol.t, *sol.y)

    def _flight_ode(self, t, state, *astro_objects):
        date = self.timescale.tt_jd(t / DAYS_TO_SECONDS)
        x, y, vx, vy = state
        central_obj = astro_objects[0]
        g = sum(
            obj.get_gravity_acceleration(np.array([x, y]) - obj.get_relative_position(central_obj, date))
            for obj in astro_objects
        )
        return [vx, vy, g[0], g[1]]
