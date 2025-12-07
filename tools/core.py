from typing import Self, Sequence
import numpy as np
from scipy.integrate import solve_ivp
from skyfield.elementslib import osculating_elements_of
import skyfield.positionlib
import skyfield.timelib


__all__ = [
    'G', 'DAYS_TO_SECONDS', 'HOURS_TO_SECONDS',
    'date_linspace', 'date_plus_seconds',
    'AstronomicalObject',
    'FlightSolver', 'FlightSolverResult'
]


G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
DAYS_TO_SECONDS = 86400.0
HOURS_TO_SECONDS = 3600.0


def date_linspace(timescale, date0, date1, num_points):
    jd_array = np.linspace(date0.tt, date1.tt, num_points)
    return timescale.tt(jd=jd_array)


def date_plus_seconds(date, seconds):
    return date + seconds / DAYS_TO_SECONDS


def zeros_2d(date):
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

    def get_orbital_velocity(self, radius) -> float | np.ndarray:
        return np.sqrt(G * self.mass / radius)

    def get_orbital_period(self, observer_object: Self, date) -> float | np.ndarray:
        if observer_object is self:
            raise ValueError('Orbital period is not defined if observer_object is the same object.')
        a = self.get_semimajor_axis(observer_object, date)
        return 2 * np.pi * np.sqrt(a ** 3 / (G * observer_object.mass))

    # Mass independent
    def get_semimajor_axis(self, observer_object: Self, date) -> float | np.ndarray:
        if observer_object is self:
            raise ValueError('Semimajor axis is not defined if observer_object is the same object.')
        rel_pos = self._get_pos_data(observer_object, date)
        elements = osculating_elements_of(rel_pos)
        return elements.semi_major_axis.m

    def get_relative_position(self, observer_object: Self, date) -> np.ndarray:
        if observer_object is self:
            return zeros_2d(date)
        pos_m = self._get_pos_data(observer_object, date).xyz.m
        return pos_m[:2]

    def get_relative_velocity(self, observer_object: Self, date) -> np.ndarray:
        if observer_object is self:
            return zeros_2d(date)
        vel_mps = self._get_pos_data(observer_object, date).velocity.m_per_s
        return vel_mps[:2]

    def _get_pos_data(self, observer_object: Self, date) -> skyfield.positionlib.Barycentric:
        observer = self.ephemeris[observer_object.name]
        target = self.ephemeris[self.name]
        pos_data = (target - observer).at(date)

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

    def with_offset(self, offset: np.ndarray):
        return FlightSolverResult(self.t, self.x + offset[0], self.y + offset[1], self.vx, self.vy)

    def find_closest_point(self, astro_object=None):
        """
        If astro_object is not None, finds a trajectory point that is the closest one to the astro_object.
        Otherwise, finds the closest point to (0, 0).

        :param astro_object: target AstronomicalObject
        :return: point index, distance
        """

        if astro_object is None:
            rel_x = self.x
            rel_y = self.y
        else:
            astro_object_pos = astro_object.get_position(self.t)
            rel_x = self.x - astro_object_pos[:, 0]
            rel_y = self.y - astro_object_pos[:, 1]

        sqr_dist = rel_x * rel_x + rel_y * rel_y
        best_i = np.argmin(sqr_dist)

        return best_i, np.sqrt(sqr_dist[best_i])

    def find_periapsis(self, astro_object, max_angle_deg):
        """
        Finds a trajectory point that can be used to enter astro_object's orbit:
        1. Filters out points whose angle between velocity and a radius vector
        to the astro_object deviates from 90° more than for max_angle_deg degrees.
        2. Finds the closest one among them.

        :param astro_object: target AstronomicalObject
        :param max_angle_deg: smaller angle means the more circular orbit
        :return: point index, periapsis radius, deviation in degrees
        """

        obj_pos = astro_object.get_position(self.t)
        rel_x = self.x - obj_pos[:, 0]
        rel_y = self.y - obj_pos[:, 1]

        sqr_dist = rel_x ** 2 + rel_y ** 2
        r_norm = np.sqrt(sqr_dist)

        dot = self.vx * rel_x + self.vy * rel_y
        v_norm = np.sqrt(self.vx ** 2 + self.vy ** 2)

        cos_angle = dot / (v_norm * r_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp numerical noise
        deviation_deg = np.abs(np.degrees(np.arccos(cos_angle)) - 90)

        max_deviation = max(max_angle_deg, np.min(deviation_deg))  # Ensure at least one match

        # Find indices where angle deviates from 90° by less than max_angle
        candidate_indices = np.where(deviation_deg < max_deviation)[0]

        # Among candidates, choose the one with the smallest distance
        best_i = candidate_indices[np.argmin(r_norm[candidate_indices])]

        return best_i, r_norm[best_i], deviation_deg[best_i]


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

        sol = solve_ivp(
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
