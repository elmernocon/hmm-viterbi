import argparse
import random
import timeit

import matplotlib.pyplot as plt
import seaborn as sns


SEED = 42
random.seed(SEED)
OBSERVATION = "THTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTH"


def benchmark(number, stmt, setup=None):
    setup = setup if setup is not None else []
    duration = timeit.timeit(setup="\n".join(setup), stmt=stmt, number=number)
    report = f"Ran {number} times. Took {duration} secs."

    truncate_len = 80
    print(f"\t{report}\t|\t{(stmt if len(stmt) < truncate_len else stmt[:truncate_len] + '..')}")
    return duration


def benchmark_numpy_base(number, observation, values):
    return benchmark(
        number,
        f'hmm.viterbi("{observation}")',
        ["from hmm.hmm_jhu import HMM", "hmm = HMM(", values, ")"],
    )


def benchmark_numpy_numba(number, observation, values):
    return benchmark(
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
    return benchmark(
        number,
        f'hmm.viterbi("{observation}")',
        ["from hmm.hmm_py import HMM", "hmm = HMM(", values, ")"],
    )


def benchmark_py_numba(number, observation, values):
    return benchmark(
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


def generate_sequence(length):
    seq = [random.choice(["H", "T"]) for _ in range(length)]
    return "".join([element for element in seq])


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
    group.add_argument(
        "-r",
        "--generate_random",
        required=False,
        type=int,
        help="The length of generated observed sequence.",
    )
    group.add_argument(
        "-i",
        "--interval",
        required=False,
        type=int,
        nargs="+",
        help="Range of sequence lengths, e.g. (10, 100, 10) would produce sequence lengths of 10 to 90 with delta=10.",
    )
    group.add_argument(
        "-x",
        "--export",
        required=False,
        type=bool,
        help="Export the results to a file."
    )
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    num = arguments.executions

    val = """
        {"F-F": 0.6, "F-L": 0.4, "L-F": 0.4, "L-L": 0.6},
        {"F-H": 0.5, "F-T": 0.5, "L-H": 0.8, "L-T": 0.2},
        {"F": 0.5, "L": 0.5}"""
    numpy_durations = []
    numpy_numba_durations = []
    python_durations = []
    python_numba_durations = []
    if not arguments.generate_random and not arguments.interval:
        obs = arguments.observation
        print(obs)
    elif arguments.generate_random:
        seq_len = arguments.generate_random
        obs = generate_sequence(seq_len)
        print(obs)
    elif arguments.interval is not None:
        start = arguments.interval[0]
        end = arguments.interval[1]
        step = arguments.interval[2]
        for seq_len in range(start, end, step):
            obs = generate_sequence(seq_len)

            print("Base NumPy")
            numpy_duration = benchmark_numpy_base(num, obs, val)
            numpy_durations.append(numpy_duration)

            print("Numba NumPy")
            numpy_numba_duration = benchmark_numpy_numba(num, obs, val)
            numpy_numba_durations.append(numpy_numba_duration)

            print("Base Python")
            python_duration = benchmark_py_base(num, obs, val)
            python_durations.append(python_duration)

            print("Numba Python")
            python_numba_duration = benchmark_py_numba(num, obs, val)
            python_numba_durations.append(python_numba_duration)

        sns.set_style("darkgrid")
        plt.plot(numpy_durations, label="Base NumPy")
        plt.plot(numpy_numba_durations, label="Numba NumPy")
        plt.plot(python_durations, label="Base Python")
        plt.plot(python_numba_durations, label="Numba Python")
        plt.xlabel("Sequence Lengths")
        plt.ylabel("Duration (in seconds)")
        plt.legend(loc="upper left")
        plt.title("HMM Benchmark (Sequence Length vs. Time)")
        plt.show()
    print("Base NumPy")
    numpy_duration = benchmark_numpy_base(num, obs, val)
    numpy_durations.append(numpy_duration)

    print("Numba NumPy")
    numpy_numba_duration = benchmark_numpy_numba(num, obs, val)
    numpy_numba_durations.append(numpy_numba_duration)

    print("Base Python")
    python_duration = benchmark_py_base(num, obs, val)
    python_durations.append(python_duration)

    print("Numba Python")
    python_numba_duration = benchmark_py_numba(num, obs, val)
    python_numba_durations.append(python_numba_duration)

    if arguments.export:
        with open("numpy_durations.txt", "w") as file:
            for index, numpy_duration in enumerate(numpy_durations):
                if index != len(numpy_durations) - 1:
                    file.write("{}\n".format(numpy_duration))
                else:
                    file.write("{}".format(numpy_duration))
        with open("python_durations.txt", "w") as file:
            for index, python_duration in enumerate(python_durations):
                if index != len(python_durations) - 1:
                    file.write("{}\n".format(python_duration))
                else:
                    file.write("{}".format(python_duration))


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
