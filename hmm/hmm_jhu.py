import numba as nb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class HMM(object):

    def __init__(self,
                 transition_matrix: Dict[str, float],
                 emission_matrix: Dict[str, float],
                 initial_probabilities: Dict[str, float]):

        # Initialize states and symbols
        self.Q, self.S = set(), set()

        # Add state labels to the set self.Q
        for transition, probability in transition_matrix.items():
            source, destination = transition[0], transition[1]
            self.Q.add(source)
            self.Q.add(destination)

        # Add symbols to the set self.S
        for emission, probability in emission_matrix.items():
            state, symbol = emission[0], emission[1]
            self.Q.add(state)
            self.S.add(symbol)

        # Sort the sets self.Q and self.S
        self.Q = sorted(list(self.Q))
        self.S = sorted(list(self.S))
        self.q_len, self.s_len = len(self.Q), len(self.S)

        # Create maps from states and symbols to integers that functions as unique identifiers
        self.q_map, self.s_map = {}, {}
        for i in range(len(self.Q)):
            self.q_map[self.Q[i]] = i
        for i in range(len(self.S)):
            self.s_map[self.S[i]] = i

        # Create and populate transition probability matrix
        self.A = np.zeros(shape=(self.q_len, self.q_len), dtype=np.float64)
        for transition, probability in transition_matrix.items():
            source, destination = transition[0], transition[1]
            self.A[self.q_map[source], self.q_map[destination]] = probability

        # Make self.A stochastic (i.e. make rows add to 1)
        # self.A /= self.A.sum(axis=1)[:, np.newaxis]
        a_sums = self.A.sum(axis=1, keepdims=1)
        a_sums[a_sums == 0] = 1
        self.A /= a_sums

        # Create and populate emission probability matrix
        self.E = np.zeros(shape=(self.q_len, self.s_len), dtype=np.float64)
        for emission, probability in emission_matrix.items():
            state, symbol = emission[0], emission[1]
            self.E[self.q_map[state], self.s_map[symbol]] = probability

        # Make self.E stochastic (i.e. make rows add to 1)
        # self.E /= self.E.sum(axis=1)[:, np.newaxis]
        e_sums = self.E.sum(axis=1, keepdims=1)
        e_sums[e_sums == 0] = 1
        self.E /= e_sums

        # Initial probability
        self.I = [0.0] * self.q_len
        for state, probability in initial_probabilities.items():
            self.I[self.q_map[state]] = probability

        # Make self.I stochastic (i.e. adds to 1)
        self.I = np.divide(self.I, sum(self.I))

        # Create log-base-2 versions for log-space functions
        self.A_log = np.log2(self.A)
        self.E_log = np.log2(self.E)
        self.I_log = np.log2(self.I)

    def __repr__(self):

        transition_data_frame = pd.DataFrame(self.A).rename(
            columns=lambda s: self.Q.__getitem__(int(s)),
            index=lambda s: self.Q.__getitem__(int(s)),
        )

        emission_data_frame = pd.DataFrame(self.E).rename(
            columns=lambda s: self.S.__getitem__(int(s)),
            index=lambda s: self.Q.__getitem__(int(s)),
        )

        initial_probability_data_frame = pd.DataFrame(self.I).rename(
            columns=lambda _: "%",
            index=lambda s: self.Q.__getitem__(int(s))
        )

        transition_log_data_frame = pd.DataFrame(self.A_log).rename(
            columns=lambda s: self.Q.__getitem__(int(s)),
            index=lambda s: self.Q.__getitem__(int(s)),
        )

        emission_log_data_frame = pd.DataFrame(self.E_log).rename(
            columns=lambda s: self.S.__getitem__(int(s)),
            index=lambda s: self.Q.__getitem__(int(s)),
        )

        initial_log_probability_data_frame = pd.DataFrame(self.I_log).rename(
            columns=lambda _: "%",
            index=lambda s: self.Q.__getitem__(int(s))
        )

        representation = [
            f"States: {self.Q}",
            f"Symbols: {self.S}",
            "",
            "Transition Matrix:", str(transition_data_frame),
            "",
            "Emission Matrix:", str(emission_data_frame),
            "",
            "Initial Probability:", str(initial_probability_data_frame),
            "",
            "Transition Matrix (log2):", str(transition_log_data_frame),
            "",
            "Emission Matrix (log2):", str(emission_log_data_frame),
            "",
            "Initial Probability (log2):", str(initial_log_probability_data_frame)
        ]

        return "\n".join(representation)

    @staticmethod
    def calculate_viterbi(
            states: List[str],
            transition_matrix,
            emission_matrix,
            initial_probabilities,
            x: List[int]) -> Tuple[float, str]:

        n_row, n_col = len(states), len(x)

        # Probability information
        # S(k, i), score of the most likely path up to step i with p(i) = k
        mat = np.zeros(shape=(n_row, n_col), dtype=np.float64)

        # Traceback information
        mat_tb = np.zeros(shape=(n_row, n_col), dtype=np.int32)

        # Fill in first column
        for i in range(0, n_row):
            mat[i, 0] = emission_matrix[i, x[0]] * initial_probabilities[i]

        # Fill in the rest of the mat and mat_tb tables
        for j in range(1, n_col):
            for i in range(0, n_row):
                ep = emission_matrix[i, x[j]]
                mx, mxi = mat[0, j - 1] * transition_matrix[0, i] * ep, 0
                for i2 in range(1, n_row):
                    pr = mat[i2, j - 1] * transition_matrix[i2, i] * ep
                    if pr > mx:
                        mx, mxi = pr, i2
                mat[i, j], mat_tb[i, j] = mx, mxi

        # Find the final state with maximal probability
        omx, omxi = mat[0, n_col - 1], 0
        for i in range(1, n_row):
            if mat[i, n_col - 1] > omx:
                omx, omxi = mat[i, n_col - 1], i

        # Backtrace
        i, p = omxi, [omxi]
        for j in range(n_col - 1, 0, -1):
            i = mat_tb[i, j]
            p.insert(0, i)

        # Build path
        path = "".join([states[q] for q in p])

        return omx, path

    @staticmethod
    def calculate_viterbi_log(
            states: List[str],
            transition_matrix,
            emission_matrix,
            initial_probabilities,
            x: List[int]) -> Tuple[float, str]:

        n_row, n_col = len(states), len(x)

        # Probability information
        # S(k, i), score of the most likely path up to step i with p(i) = k
        mat = np.zeros(shape=(n_row, n_col), dtype=np.float64)

        # Traceback information
        mat_tb = np.zeros(shape=(n_row, n_col), dtype=np.int32)

        # Fill in first column
        for i in range(0, n_row):
            mat[i, 0] = emission_matrix[i, x[0]] + initial_probabilities[i]

        # Fill in the rest of the mat and mat_tb tables
        for j in range(1, n_col):
            for i in range(0, n_row):
                ep = emission_matrix[i, x[j]]
                mx, mxi = mat[0, j - 1] + transition_matrix[0, i] + ep, 0
                for i2 in range(1, n_row):
                    pr = mat[i2, j - 1] + transition_matrix[i2, i] + ep
                    if pr > mx:
                        mx, mxi = pr, i2
                mat[i, j], mat_tb[i, j] = mx, mxi

        # Find the final state with maximal probability
        omx, omxi = mat[0, n_col - 1], 0
        for i in range(1, n_row):
            if mat[i, n_col - 1] > omx:
                omx, omxi = mat[i, n_col - 1], i

        # Backtrace
        i, p = omxi, [omxi]
        for j in range(n_col - 1, 0, -1):
            i = mat_tb[i, j]
            p.insert(0, i)

        # Build path
        path = "".join([states[q] for q in p])

        return omx, path

    @staticmethod
    @nb.jit(nopython=True)
    def calculate_viterbi_numba(
            states: List[str],
            transition_matrix,
            emission_matrix,
            initial_probabilities,
            x: List[int]) -> Tuple[float, str]:

        n_row, n_col = len(states), len(x)

        # Probability information
        # S(k, i), score of the most likely path up to step i with p(i) = k
        mat = np.zeros(shape=(n_row, n_col), dtype=np.float64)

        # Traceback information
        mat_tb = np.zeros(shape=(n_row, n_col), dtype=np.int32)

        # Fill in first column
        for i in range(0, n_row):
            mat[i, 0] = emission_matrix[i, x[0]] * initial_probabilities[i]

        # Fill in the rest of the mat and mat_tb tables
        for j in range(1, n_col):
            for i in range(0, n_row):
                ep = emission_matrix[i, x[j]]
                mx, mxi = mat[0, j - 1] * transition_matrix[0, i] * ep, 0
                for i2 in range(1, n_row):
                    pr = mat[i2, j - 1] * transition_matrix[i2, i] * ep
                    if pr > mx:
                        mx, mxi = pr, i2
                mat[i, j], mat_tb[i, j] = mx, mxi

        # Find the final state with maximal probability
        omx, omxi = mat[0, n_col - 1], 0
        for i in range(1, n_row):
            if mat[i, n_col - 1] > omx:
                omx, omxi = mat[i, n_col - 1], i

        # Backtrace
        i, p = omxi, [omxi]
        for j in range(n_col - 1, 0, -1):
            i = mat_tb[i, j]
            p.insert(0, i)

        # Build path
        path = "".join([states[q] for q in p])

        return omx, path

    @staticmethod
    @nb.jit(nopython=True)
    def calculate_viterbi_log_numba(
            states: List[str],
            transition_matrix,
            emission_matrix,
            initial_probabilities,
            x: List[int]) -> Tuple[float, str]:

        n_row, n_col = len(states), len(x)

        # Probability information
        # S(k, i), score of the most likely path up to step i with p(i) = k
        mat = np.zeros(shape=(n_row, n_col), dtype=np.float64)

        # Traceback information
        mat_tb = np.zeros(shape=(n_row, n_col), dtype=np.int32)

        # Fill in first column
        for i in range(0, n_row):
            mat[i, 0] = emission_matrix[i, x[0]] + initial_probabilities[i]

        # Fill in the rest of the mat and mat_tb tables
        for j in range(1, n_col):
            for i in range(0, n_row):
                ep = emission_matrix[i, x[j]]
                mx, mxi = mat[0, j - 1] + transition_matrix[0, i] + ep, 0
                for i2 in range(1, n_row):
                    pr = mat[i2, j - 1] + transition_matrix[i2, i] + ep
                    if pr > mx:
                        mx, mxi = pr, i2
                mat[i, j], mat_tb[i, j] = mx, mxi

        # Find the final state with maximal probability
        omx, omxi = mat[0, n_col - 1], 0
        for i in range(1, n_row):
            if mat[i, n_col - 1] > omx:
                omx, omxi = mat[i, n_col - 1], i

        # Backtrace
        i, p = omxi, [omxi]
        for j in range(n_col - 1, 0, -1):
            i = mat_tb[i, j]
            p.insert(0, i)

        # Build path
        path = "".join([states[q] for q in p])

        return omx, path

    def convert_symbols(self, x: str) -> List[int]:
        return list(map(self.s_map.get, x))
        
    def joint_probability(self, p: str, x: str) -> float:

        # Convert state characters to identifiers
        p = list(map(self.q_map.get, p))

        # Convert symbol characters to identifiers
        x = list(map(self.s_map.get, x))

        # Initial probability
        probability = self.I[p[0]]

        # Transition probability
        for i in range(1, len(p)):
            probability *= self.A[p[i - 1], p[i]]

        # Emission probability
        for i in range(0, len(p)):
            probability *= self.E[p[i], x[i]]

        return probability

    def joint_probability_log(self, p: str, x: str) -> float:

        # Convert state characters to identifiers
        p = list(map(self.q_map.get, p))

        # Convert symbol characters to identifiers
        x = list(map(self.s_map.get, x))

        # Initial probability
        probability = self.I_log[p[0]]

        # Transition probability
        for i in range(1, len(p)):
            probability += self.A_log[p[i - 1], p[i]]

        # Emission probability
        for i in range(0, len(p)):
            probability += self.E_log[p[i], x[i]]

        return probability

    def viterbi(self, x: str) -> Tuple[float, str]:
        return HMM.calculate_viterbi(self.Q, self.A, self.E, self.I, self.convert_symbols(x))

    def viterbi_log(self, x: str) -> Tuple[float, str]:
        return HMM.calculate_viterbi_log(self.Q, self.A_log, self.E_log, self.I_log, self.convert_symbols(x))

    def viterbi_numba(self, x: str) -> Tuple[float, str]:
        return HMM.calculate_viterbi_numba(self.Q, self.A, self.E, self.I, self.convert_symbols(x))

    def viterbi_log_numba(self, x: str) -> Tuple[float, str]:
        return HMM.calculate_viterbi_log_numba(self.Q, self.A_log, self.E_log, self.I_log, self.convert_symbols(x))

    pass
