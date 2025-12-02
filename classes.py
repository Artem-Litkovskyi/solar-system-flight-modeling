import numpy as np
from scipy.integrate import solve_ivp
from skyfield.elementslib import osculating_elements_of


G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2


def date_linspace(timescale, date0, date1, num_points):
    jd_array = np.linspace(date0.tt, date1.tt, num_points)
    return timescale.tt(jd=jd_array)


def date_plus_seconds(date, seconds):
    return date + seconds / 86400.0  # Convert seconds to days


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

    def get_sphere_of_influence(self, observer_object, date):
        r = np.linalg.norm(self.get_relative_position(observer_object, date))
        return r * (self.mass / observer_object.mass) ** (2 / 5)

    def get_orbital_velocity(self, radius):
        return np.sqrt(G * self.mass / radius)

    def get_orbital_period(self, observer_object, date):
        a = self.get_semimajor_axis(observer_object, date)
        return 2 * np.pi * np.sqrt(a ** 3 / (G * observer_object.mass))

    # Mass independent
    def get_semimajor_axis(self, observer_object, date):
        rel_pos = self._get_relative_position_data(observer_object, date)
        elements = osculating_elements_of(rel_pos)
        return elements.semi_major_axis.m

    def get_relative_position(self, observer_object, date) -> np.ndarray:
        pos_m = self._get_relative_position_data(observer_object, date).xyz.m
        return np.array(pos_m)

    def get_relative_velocity(self, observer_object, date) -> np.ndarray:
        vel_mps = self._get_relative_position_data(observer_object, date).velocity.m_per_s
        return np.array(vel_mps)

    def _get_relative_position_data(self, observer_object, date):
        observer = self.ephemeris[observer_object.name]
        target = self.ephemeris[self.name]
        return (target - observer).at(date)

    # Utils
    def rel_v_to_abs(self, observer_object, time, rel_velocity: np.ndarray) -> np.ndarray:
        return rel_velocity + self.get_velocity(observer_object, time)

    def abs_v_to_rel(self, observer_object, time, abs_velocity: np.ndarray) -> np.ndarray:
        return abs_velocity - self.get_velocity(observer_object, time)


class FlightSolver:
    def __init__(self, rtol=1e-12, atol=1e-12):
        self.rtol = rtol
        self.atol = atol

    @staticmethod
    def _flight_ode_simple(t, state, astro_object):
        x, y, vx, vy = state
        g = astro_object.get_gravity_acceleration(np.array([x, y]))
        return [vx, vy, g[0], g[1]]

    @staticmethod
    def _flight_ode_multi(t, state, astro_objects):
        x, y, vx, vy = state
        g = sum(
            obj.get_gravity_acceleration(np.array([x, y]) - obj.get_position(t))
            for obj in astro_objects
        )
        return [vx, vy, g[0], g[1]]

    def solve(self, astro_objects, x0, y0, vx0, vy0, t_span, points_per_hour=5):
        """
        If astro_objects is a single AstronomicalObject, coordinates are considered to be relative.
        If astro_objects is a list of AstronomicalObjects, coordinates are considered to be absolute.

        :param astro_objects: target AstronomicalObjects
        :param x0: starting x coordinate
        :param y0: starting y coordinate
        :param vx0: starting x velocity
        :param vy0: starting y velocity
        :param t_span: interval of integration (t0, tf)
        :param points_per_hour: number of trajectory points per hour
        """

        delta_time = t_span[1] - t_span[0]
        t_eval = np.linspace(*t_span, int(delta_time / 3600 * points_per_hour))

        fun = self._flight_ode_simple if isinstance(astro_objects, AstronomicalObject) else self._flight_ode_multi

        sol = solve_ivp(
            lambda t, state: fun(t, state, astro_objects),
            t_span, [x0, y0, vx0, vy0], t_eval=t_eval,
            rtol=self.rtol, atol=self.atol,
        )

        return FlightSolverResult(sol.t, *sol.y)


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
