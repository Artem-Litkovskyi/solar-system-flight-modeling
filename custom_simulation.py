from tools.instances import *
from earth_to_mars_mission import full_earth_to_mars_mission


PREFIX = 'custom'

# Earth Parking Orbit
EARTH_ORBIT_RADIUS = EARTH.radius + 200e3
ESCAPE_DATE = TIMESCALE.utc(2026, 10, 15, 6, 0)

# Parameters from optimization
ESCAPE_ANTISOLAR_OFFSET = -44  # degrees
ESCAPE_DELTA_V = 4399.621

# Mars Orbit Insertion
INSERT_DATE = TIMESCALE.utc(2027, 11, 21, 19, 50)
MARS_SEMIMAJOR = (MARS.radius + 382e3 + MARS.radius + 44500e3) / 2


def main():
    full_earth_to_mars_mission(
        EARTH_ORBIT_RADIUS,
        ESCAPE_DATE,
        ESCAPE_ANTISOLAR_OFFSET,
        ESCAPE_DELTA_V,
        INSERT_DATE,
        MARS_SEMIMAJOR,
        PREFIX
    )


if __name__ == '__main__':
    main()
