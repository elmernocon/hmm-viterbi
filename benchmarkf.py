import argparse
import pandas as pd
import random as rd

from hmm.hmm_py_profile import HMM


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


def create_profile_hmm(size: int = 1) -> HMM:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking Profile HMM")
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
        "-l",
        "--length",
        required=False,
        default=5,
        type=int,
        help="The length of the random observation sequence to generate.",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    init()

    sz = 5  # sz is automatically multiplied by 5.

    observation = create_observation(sz)
    print(observation, len(observation))

    hmm = create_profile_hmm(sz)
    # print(hmm)
    print(hmm.viterbi_log(observation))
