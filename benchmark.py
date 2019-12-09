import argparse
import random
import timeit


SEED = 42
random.seed(SEED)
OBSERVATION = "THTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTH"


def benchmark(number, stmt, setup=None):
    setup = setup if setup is not None else []
    duration = timeit.timeit(setup="\n".join(setup), stmt=stmt, number=number)
    report = f"Ran {number} times. Took {duration} secs."

    print(f"{report.ljust(75)} {stmt}")


def benchmark_numpy_base(number, observation, values):
    benchmark(
        number,
        f'hmm.viterbi("{observation}")',
        ["from hmm.hmm_jhu import HMM", "hmm = HMM(", values, ")"],
    )


def benchmark_numpy_numba(number, observation, values):
    benchmark(
        number,
        f'hmm_numba.viterbi("{observation}")',
        [
            "import warnings",
            "from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning",
            "warnings.simplefilter('ignore', category=NumbaDeprecationWarning)",
            "warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)",
            "from hmm.hmm_jhu_numba import HMMNumba",
            "HMMNumba.warm_up()",
            "hmm_numba = HMMNumba(",
            values,
            ")",
        ],
    )


def benchmark_py_base(number, observation, values):
    benchmark(
        number,
        f'hmm.viterbi("{observation}")',
        ["from hmm.hmm_py import HMM", "hmm = HMM(", values, ")"],
    )


def benchmark_py_numba(number, observation, values):
    benchmark(
        number,
        f'hmm_numba.viterbi("{observation}")',
        [
            "import warnings",
            "from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning",
            "warnings.simplefilter('ignore', category=NumbaDeprecationWarning)",
            "warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)",
            "from hmm.hmm_py_numba import HMMNumba",
            "HMMNumba.warm_up()",
            "hmm_numba = HMMNumba(",
            values,
            ")",
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmarking HMM for Dishonest Casino"
    )
    group = parser.add_argument_group("Parameters")
    group.add_argument(
        "-e",
        "--executions",
        required=False,
        default=1,
        type=int,
        help="The number of times to run the program.",
    )
    group.add_argument(
        "-o",
        "--observation",
        required=False,
        default=OBSERVATION,
        type=str,
        help="The observed sequence.",
    )
    arguments = parser.parse_args()
    return arguments


def main():
    pass


if __name__ == "__main__":
    num = 1
    obs = "THTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTH"
    val = """
        {"F-F": 0.6, "F-L": 0.4, "L-F": 0.4, "L-L": 0.6},
        {"F-H": 0.5, "F-T": 0.5, "L-H": 0.8, "L-T": 0.2},
        {"F": 0.5, "L": 0.5}"""

    benchmark_numpy_base(num, obs, val)
    # benchmark_numpy_numba(num, obs, val)
    benchmark_py_base(num, obs, val)
    # benchmark_py_numba(num, obs, val)
