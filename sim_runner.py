import itertools

import concurrent.futures
import numpy as np
from tqdm import tqdm

from tools.core import *


def trajectories_table_wrapper(args):
    flight_func, d, o, v = args
    return flight_func(d, o, v)


def trajectories_table(trajectory_function, escape_dates, antisolar_offsets, escape_delta_vs):
    param_generator = (
        (trajectory_function, d, o, v)
        for d, o, v in itertools.product(escape_dates, antisolar_offsets, escape_delta_vs)
    )

    total_sims = escape_dates.shape[0] * antisolar_offsets.size * escape_delta_vs.size

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_iterator = executor.map(trajectories_table_wrapper, param_generator)
        results_flat = list(tqdm(results_iterator, total=total_sims, desc='Simulating Trajectories'))

    results = np.array(results_flat, dtype=FlightSolverResult).reshape(escape_dates.shape[0], antisolar_offsets.size, escape_delta_vs.size)

    return results