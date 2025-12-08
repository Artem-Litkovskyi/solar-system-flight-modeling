import numpy as np
import concurrent.futures
import itertools
from tqdm import tqdm  # Requires: pip install tqdm

from tools.core import *
from tools.instances import *


def simulation_wrapper(args):
    flight_func, d, o, v = args
    return flight_func(d, o, v)


def run_simulations(
        flight_function,
        date_span, date_num,
        antisolar_offset_span, antisolar_offset_num,
        escape_delta_v_span, escape_delta_v_num
):
    dates = date_linspace(TIMESCALE, date_span[0], date_span[1], date_num)
    offsets = np.linspace(antisolar_offset_span[0], antisolar_offset_span[1], antisolar_offset_num)
    delta_vs = np.linspace(escape_delta_v_span[0], escape_delta_v_span[1], escape_delta_v_num)

    param_generator = (
        (flight_function, d, o, v)
        for d, o, v in itertools.product(dates, offsets, delta_vs)
    )

    total_sims = date_num * antisolar_offset_num * escape_delta_v_num

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # chunk size=1 is crucial for SLOW functions.
        # It allows dynamic load balancing: if one simulation takes longer,
        # other cores continue picking up the slack.
        results_iterator = executor.map(simulation_wrapper, param_generator, chunksize=1)
        results_flat = list(tqdm(results_iterator, total=total_sims, desc='Simulating Trajectories'))

    results = np.array(results_flat, dtype=FlightSolverResult).reshape(date_num, antisolar_offset_num, escape_delta_v_num)

    return dates, offsets, delta_vs, results