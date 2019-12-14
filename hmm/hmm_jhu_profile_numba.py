# MIT License
#
# Copyright (c) 2019 Elmer Nocon, Abien Fred Agarap
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Profile Hidden Markov Model using NumPy with Numba"""
import numba as nb
import numpy as np

from typing import List, Tuple

from hmm.hmm_jhu_profile import HMM


class HMMNumba(HMM):
    @staticmethod
    def warm_up():
        m = HMMNumba({"A-A": 1.0}, {"A-a": 1.0}, {"A": 1.0})

        # HMMNumba.calculate_viterbi(m.Q, m.A, m.E, m.I, m.convert_symbols("a"))
        m.viterbi("a")

        # HMMNumba.calculate_viterbi_log(m.Q, m.A_log, m.E_log, m.I_log, m.convert_symbols("a"))
        m.viterbi("a")

    @staticmethod
    @nb.jit(nopython=True)
    def calculate_viterbi(
        states: List[str],
        transition_matrix,
        emission_matrix,
        initial_probabilities,
        x: List[int],
        deletion_states=None,
    ) -> Tuple[float, str]:

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
                offset = -1 if i not in deletion_states else 0
                ep = emission_matrix[i, x[j]]
                mx, mxi = mat[0, j + offset] * transition_matrix[0, i] * ep, 0
                for i2 in range(1, n_row):
                    offset = -1 if i not in deletion_states else 0
                    pr = mat[i2, j + offset] * transition_matrix[i2, i] * ep
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
    def calculate_viterbi_log(
        states: List[str],
        transition_matrix,
        emission_matrix,
        initial_probabilities,
        x: List[int],
        deletion_states=None,
    ) -> Tuple[float, str]:

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
                offset = -1 if i not in deletion_states else 0
                ep = emission_matrix[i, x[j]]
                mx, mxi = mat[0, j + offset] + transition_matrix[0, i] + ep, 0
                for i2 in range(1, n_row):
                    offset = -1 if i not in deletion_states else 0
                    pr = mat[i2, j + offset] + transition_matrix[i2, i] + ep
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

    def viterbi(self, x: str) -> Tuple[float, str]:
        return HMM.calculate_viterbi(
            self.Q,
            self.A,
            self.E,
            self.I,
            self.convert_symbols(x),
            deletion_states=self.deletion_states_mapped,
        )

    def viterbi_log(self, x: str) -> Tuple[float, str]:
        return HMM.calculate_viterbi_log(
            self.Q,
            self.A_log,
            self.E_log,
            self.I_log,
            self.convert_symbols(x),
            deletion_states=self.deletion_states_mapped,
        )
