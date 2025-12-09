import numpy as np
from scipy.optimize import minimize

from tools.instances import *
from maven_optimizer1 import ESCAPE_DATES, earth_to_mars_trajectory


INIT_ANTISOLAR_OFFSET = -32
INIT_ESCAPE_DELTA_V = 4058

ANTISOLAR_OFFSET_SCALE = 100
DELTA_V_SCALE = 1

MARS_ORBIT_PERIAPSIS = MARS.radius + 382e3


def main():
    options = {
        'ftol': 1e-4,
        'gtol': 1e-4,
        'eps': 1e-5,
        'maxiter': 25
    }

    init_scaled = [INIT_ANTISOLAR_OFFSET * ANTISOLAR_OFFSET_SCALE, INIT_ESCAPE_DELTA_V * DELTA_V_SCALE]
    res = minimize(distance_to_mars, init_scaled, method='L-BFGS-B', options=options)

    best_offset = res.x[0] / ANTISOLAR_OFFSET_SCALE
    best_delta_v = res.x[1] / DELTA_V_SCALE

    print(f'Result: offset={best_offset}, delta_v={best_delta_v}')


def distance_to_mars(args):
    antisolar_offset_scaled, escape_delta_v_scaled = args

    res = earth_to_mars_trajectory(
        ESCAPE_DATES[0],
        antisolar_offset_scaled / ANTISOLAR_OFFSET_SCALE,
        escape_delta_v_scaled / DELTA_V_SCALE
    )

    return np.abs(res.find_last_point_distance(TIMESCALE, MARS, SUN) - MARS_ORBIT_PERIAPSIS)


if __name__ == '__main__':
    main()