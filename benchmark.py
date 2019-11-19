import timeit


def benchmark(number, stmt, setup=None):
    setup = setup if setup is not None else []
    duration = timeit.timeit(setup="\n".join(setup), stmt=stmt, number=number)
    report = f"Ran {number} times. Took {duration} secs."

    print(f"{report.ljust(75)} {stmt}")


def benchmark_base(number, observation, values):
    benchmark(number, f'hmm.viterbi("{observation}")', [
        "from hmm.hmm_jhu import HMM",
        "hmm = HMM(",
        values,
        ")"
    ])


def benchmark_numba(number, observation, values):
    benchmark(number, f'hmm_numba.viterbi("{observation}")', [
        "import warnings",
        "from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning",
        "warnings.simplefilter('ignore', category=NumbaDeprecationWarning)",
        "warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)",
        "from hmm.hmm_jhu_numba import HMMNumba",
        "HMMNumba.warm_up()",
        "hmm_numba = HMMNumba(",
        values,
        ")"
    ])


if __name__ == '__main__':
    num = 100000
    obs = "THTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTH"
    val = '''
        {"FF": 0.6, "FL": 0.4, "LF": 0.4, "LL": 0.6},
        {"FH": 0.5, "FT": 0.5, "LH": 0.8, "LT": 0.2},
        {"F": 0.5, "L": 0.5}'''

    benchmark_base(num, obs, val)
    # benchmark_numba(num, obs, val)
