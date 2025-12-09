from skyfield.api import Loader
from tools.core import AstronomicalObject, FlightSolver

__all__ = ['EPHEMERIS', 'TIMESCALE', 'SOLVER', 'SUN', 'EARTH', 'MARS']

load = Loader('./skyfield_data')

EPHEMERIS = load('de421.bsp')
TIMESCALE = load.timescale()

SOLVER = FlightSolver(TIMESCALE, method='RK45', rtol=1e-11, atol=1e-8)

SUN = AstronomicalObject(
    ephemeris=EPHEMERIS,
    timescale=TIMESCALE,
    name='Sun',
    mass=1.98842e30,
    radius=695700e3
)

EARTH = AstronomicalObject(
    ephemeris=EPHEMERIS,
    timescale=TIMESCALE,
    name='Earth',
    mass=5.97219e24,
    radius=6371.0084e3
)

MARS = AstronomicalObject(
    ephemeris=EPHEMERIS,
    timescale=TIMESCALE,
    name='Mars',
    mass=6.41693e23,
    radius=3389.50e3
)