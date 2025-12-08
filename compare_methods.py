import numpy as np
from tools.core import *
from tools.instances import *
from tools.tables import *


orbit_r = 8000e3
orbit_v = EARTH.get_circular_orbit_velocity(orbit_r)
orbit_period = EARTH.get_orbital_period(orbit_r)

date0 = TIMESCALE.utc(2025, 12, 1, 12)
date1 = date_plus_seconds(date0, orbit_period)

methods = ['RK23', 'RK45', 'DOP853']
rtols = [1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
atols = [1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]


def main():
    # Find the best method and rtol combination
    m_and_r_err = []
    best_rtol_i_per_method = []
    best_rtol_per_method = []

    for method in methods:
        row = []
        for rtol in rtols:
            row.append(get_error(method, rtol, 1e-10))
        best_rtol_i_per_method.append(np.argmin(row))
        best_rtol_per_method.append(np.min(row))
        m_and_r_err.append(row)

    best_method_i = np.argmin(best_rtol_per_method)
    best_rtol_i = best_rtol_i_per_method[best_method_i]

    # Find the best atol
    a_err = []
    for atol in atols:
        a_err.append(get_error(methods[best_method_i], rtols[best_rtol_i], atol))

    best_atol_i = np.argmin(a_err)

    # Save results
    print('Orbital period (hours):', orbit_period / HOURS_TO_SECONDS)
    print('Best method:', methods[best_method_i])
    print('Best rtol:', rtols[best_rtol_i])
    print('Best atol:', atols[best_atol_i])

    make_table('methods_and_rtols.csv', methods, rtols, m_and_r_err)
    make_table('atols.csv', ['error'], atols, [a_err])


def get_error(method, rtol, atol):
    solver = FlightSolver(TIMESCALE, method=method, rtol=rtol, atol=atol)

    res = solver.solve(
        [EARTH],
        x0=orbit_r, y0=0,
        vx0=0, vy0=EARTH.get_circular_orbit_velocity(orbit_r),
        date_span=(date0, date1)
    )

    delta_x = res.x[0] - res.x[-1]
    delta_y = res.y[0] - res.y[-1]

    return np.hypot(delta_x, delta_y)


if __name__ == '__main__':
    main()