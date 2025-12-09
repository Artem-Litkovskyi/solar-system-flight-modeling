from tools.core import *
from tools.instances import *
from earth_to_mars_mission import full_earth_to_mars_mission


PREFIX = 'MAVEN'

# Earth Parking Orbit
EARTH_ORBIT_RADIUS = EARTH.radius + 400e3
ESCAPE_DATE = TIMESCALE.utc(2013, 10, 18, 18, 28)

# Parameters from optimization
ESCAPE_ANTISOLAR_OFFSET = -31.9732  # degrees
ESCAPE_DELTA_V = 4059.173

# Mars Orbit Insertion
INSERT_DATE = TIMESCALE.utc(2014, 9, 23, 2, 36)  # Note: adjusted by +1:00:22
MARS_SEMIMAJOR = MARS.radius + (382e3 + 44500e3) / 2

# Expected values
EXPECTED_ESCAPE_DELTA_V = 3770
EXPECTED_INSERT_DELTA_V = 1230.5


def main():
    insert_delta_v = full_earth_to_mars_mission(
        EARTH_ORBIT_RADIUS,
        ESCAPE_DATE,
        ESCAPE_ANTISOLAR_OFFSET,
        ESCAPE_DELTA_V,
        INSERT_DATE,
        MARS_SEMIMAJOR,
        PREFIX
    )

    print('\n--- Relative errors ---')
    print(f'Escape delta v: {relative_error(EXPECTED_ESCAPE_DELTA_V, ESCAPE_DELTA_V):.4f}')
    print(f'Insert delta v: {relative_error(EXPECTED_INSERT_DELTA_V, abs(insert_delta_v)):.4f}')
    print(f'Total delta v: {relative_error(
        EXPECTED_ESCAPE_DELTA_V + EXPECTED_INSERT_DELTA_V,
        ESCAPE_DELTA_V + abs(insert_delta_v)
    ):.4f}')


if __name__ == '__main__':
    main()
