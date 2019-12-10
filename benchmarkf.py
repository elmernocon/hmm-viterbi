import argparse
import random as rd
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hmm.hmm_py_profile import HMM as phmm_python  # base python
from hmm.hmm_py_profile_numba import HMMNumba as phmm_python_nb  # numba python
from hmm.hmm_jhu_profile import HMM as phmm_numpy  # base numpy
from hmm.hmm_jhu_profile_numba import HMMNumba as phmm_numpy_nb  # numba numpy


SEED = 42
rd.seed(42)


def init():
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)


def pad(i: int, size: int = 3, fill: str = "0") -> str:
    return str(i).rjust(size, fill)


def create_observation(size: int = 1) -> str:
    s = size * 5
    t = float(s) * 0.2
    u = rd.randint(-t, t)
    return "".join([rd.choice("acgt") for _ in range(s + u)])


def create_profile_hmm(HMM: object, size: int = 1) -> object:
    transition_matrix = dict()

    transition_matrix[f"Begin-I{pad(0)}"] = 0.1
    transition_matrix[f"Begin-M{pad(1)}"] = 0.9

    transition_matrix[f"I{pad(0)}-I{pad(0)}"] = 0.1
    transition_matrix[f"I{pad(0)}-D{pad(1)}"] = 0.01
    transition_matrix[f"I{pad(0)}-M{pad(1)}"] = 0.89

    for s in range(1, (size * 5) + 1):
        t = s + 1
        transition_matrix[f"I{pad(s)}-I{pad(s)}"] = 0.1
        if t <= (size * 5):
            transition_matrix[f"I{pad(s)}-D{pad(t)}"] = 0.01
            transition_matrix[f"I{pad(s)}-M{pad(t)}"] = 0.89
        else:
            transition_matrix[f"I{pad(s)}-End"] = 0.9

    for s in range(1, (size * 5) + 1):
        t = s + 1
        transition_matrix[f"D{pad(s)}-I{pad(s)}"] = 0.01
        if t <= (size * 5):
            transition_matrix[f"D{pad(s)}-D{pad(t)}"] = 0.1
            transition_matrix[f"D{pad(s)}-M{pad(t)}"] = 0.89
        else:
            transition_matrix[f"D{pad(s)}-End"] = 0.99

    for s in range(1, (size * 5) + 1):
        t = s + 1
        transition_matrix[f"M{pad(s)}-I{pad(s)}"] = 0.05
        if t <= (size * 5):
            transition_matrix[f"M{pad(s)}-M{pad(t)}"] = 0.9
            transition_matrix[f"M{pad(s)}-D{pad(t)}"] = 0.05
        else:
            transition_matrix[f"M{pad(s)}-End"] = 0.95

    emission_matrix = dict()

    emission_matrix[f"I{pad(0)}-a"] = 0.2
    emission_matrix[f"I{pad(0)}-c"] = 0.3
    emission_matrix[f"I{pad(0)}-g"] = 0.3
    emission_matrix[f"I{pad(0)}-t"] = 0.2

    for s in range(1, (size * 5) + 1):
        emission_matrix[f"I{pad(s)}-a"] = 0.2
        emission_matrix[f"I{pad(s)}-c"] = 0.3
        emission_matrix[f"I{pad(s)}-g"] = 0.3
        emission_matrix[f"I{pad(s)}-t"] = 0.2

    for s in range(size):
        t = s * 5 + 1
        emission_matrix[f"M{pad(t)}-a"] = 0.8
        emission_matrix[f"M{pad(t)}-c"] = 0.1
        emission_matrix[f"M{pad(t)}-g"] = 0.05
        emission_matrix[f"M{pad(t)}-t"] = 0.05
        u = s * 5 + 2
        emission_matrix[f"M{pad(u)}-a"] = 0.1
        emission_matrix[f"M{pad(u)}-c"] = 0.1
        emission_matrix[f"M{pad(u)}-g"] = 0.75
        emission_matrix[f"M{pad(u)}-t"] = 0.05
        v = s * 5 + 3
        emission_matrix[f"M{pad(v)}-a"] = 0.1
        emission_matrix[f"M{pad(v)}-c"] = 0.1
        emission_matrix[f"M{pad(v)}-g"] = 0.7
        emission_matrix[f"M{pad(v)}-t"] = 0.1
        x = s * 5 + 4
        emission_matrix[f"M{pad(x)}-a"] = 0.1
        emission_matrix[f"M{pad(x)}-c"] = 0.1
        emission_matrix[f"M{pad(x)}-g"] = 0.2
        emission_matrix[f"M{pad(x)}-t"] = 0.6
        y = s * 5 + 5
        emission_matrix[f"M{pad(y)}-a"] = 0.1
        emission_matrix[f"M{pad(y)}-c"] = 0.8
        emission_matrix[f"M{pad(y)}-g"] = 0.05
        emission_matrix[f"M{pad(y)}-t"] = 0.05

    initial_probabilities = dict()

    initial_probabilities["Begin"] = 1.0
    initial_probabilities[f"I{pad(0)}"] = 0.0
    initial_probabilities["End"] = 0.0

    for s in range(1, (size * 5) + 1):
        initial_probabilities[f"I{pad(s)}"] = 0.0
        initial_probabilities[f"D{pad(s)}"] = 0.0
        initial_probabilities[f"M{pad(s)}"] = 0.0

    return HMM(transition_matrix, emission_matrix, initial_probabilities)


def get_duration(hmm, size, observation):
    start_time = time.time()
    hmm = create_profile_hmm(hmm, size)
    result = hmm.viterbi_log(observation)
    end_time = time.time()
    duration = end_time - start_time
    return duration, result


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking Profile HMM")
    group = parser.add_argument_group("Parameters")
    group.add_argument(
        "-l",
        "--length",
        required=False,
        default=5,
        type=int,
        help="The length of the random observation sequence to generate.",
    )
    group.add_argument(
        "-x",
        "--export",
        required=False,
        default=True,
        type=bool,
        help="Export the results to a file."
    )
    arguments = parser.parse_args()
    return arguments


def main(arguments):

    init()
    phmm_python_nb.warm_up()
    phmm_numpy_nb.warm_up()
    python_durations = []
    numpy_durations = []
    python_numba_durations = []
    numpy_numba_durations = []

    for size in range(arguments.length, 10, 1):
        observation = create_observation(size)
        print(observation, len(observation))

        duration, result = get_duration(phmm_python, size, observation)
        python_durations.append(duration)
        print("Base Python")
        print(f"\tTook {duration} secs.\t|\t{result}")

        duration, result = get_duration(phmm_numpy, size, observation)
        numpy_durations.append(duration)
        print("Base NumPy")
        print(f"\tTook {duration} secs.\t|\t{result}")

        duration, result = get_duration(phmm_python_nb, size, observation)
        python_numba_durations.append(duration)
        print("Numba Python")
        print(f"\tTook {duration} secs.\t|\t{result}")

        duration, result = get_duration(phmm_numpy_nb, size, observation)
        numpy_numba_durations.append(duration)
        print("Numba NumPy")
        print(f"\tTook {duration} secs.\t|\t{result}")

    sns.set_style("darkgrid")
    plt.plot(python_durations, label="Base Python")
    plt.plot(numpy_durations, label="Base NumPy")
    plt.plot(python_numba_durations, label="Numba Python")
    plt.plot(numpy_numba_durations, label="Numba NumPy")
    plt.xlabel("Sequence Lengths")
    plt.ylabel("Duration (in seconds)")
    plt.title("Profile HMM (Sequence Length vs. Time)")
    plt.legend(loc="upper left")
    plt.show()

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
