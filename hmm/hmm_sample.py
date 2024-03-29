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
"""Hidden Markov Model for CPG Islands"""
import numpy as np
from hmm.hmm_jhu import HMM


def create_hmm_cpg_islands():

    hmm_cpg_islands = HMM(
        # States
        # 0 : Begin/End
        # A : A+
        # C : C+
        # G : G+
        # T : T+
        # B : A-
        # D : C-
        # H : G-
        # U : T-
        # Symbols
        # a
        # c
        # g
        # t
        # Transition Matrix
        {
            "0-0": 0.0000000,
            "0-A": 0.0725193,
            "0-C": 0.1637630,
            "0-G": 0.1788242,
            "0-T": 0.0754545,
            "0-B": 0.1322050,
            "0-D": 0.1267006,
            "0-H": 0.1226380,
            "0-U": 0.1278950,
            "A-0": 0.0010000,
            "A-A": 0.1762237,
            "A-C": 0.2682517,
            "A-G": 0.4170629,
            "A-T": 0.1174825,
            "A-B": 0.0035964,
            "A-D": 0.0054745,
            "A-H": 0.0085104,
            "A-U": 0.0023976,
            "C-0": 0.0010000,
            "C-A": 0.1672435,
            "C-C": 0.3599201,
            "C-G": 0.2679840,
            "C-T": 0.1838722,
            "C-B": 0.0034131,
            "C-D": 0.0073453,
            "C-H": 0.0054690,
            "C-U": 0.0037524,
            "G-0": 0.0010000,
            "G-A": 0.1576223,
            "G-C": 0.3318881,
            "G-G": 0.3671328,
            "G-T": 0.1223776,
            "G-B": 0.0032167,
            "G-D": 0.0067732,
            "G-H": 0.0074915,
            "G-U": 0.0024975,
            "T-0": 0.0010000,
            "T-A": 0.0773426,
            "T-C": 0.3475514,
            "T-G": 0.3759440,
            "T-T": 0.1781818,
            "T-B": 0.0015784,
            "T-D": 0.0070929,
            "T-H": 0.0076723,
            "T-U": 0.0036363,
            "B-0": 0.0010000,
            "B-A": 0.0002997,
            "B-C": 0.0002047,
            "B-G": 0.0002837,
            "B-T": 0.0002097,
            "B-B": 0.2994005,
            "B-D": 0.2045904,
            "B-H": 0.2844305,
            "B-U": 0.2095804,
            "D-0": 0.0010000,
            "D-A": 0.0003216,
            "D-C": 0.0002977,
            "D-G": 0.0000769,
            "D-T": 0.0003016,
            "D-B": 0.3213566,
            "D-D": 0.2974045,
            "D-H": 0.0778441,
            "D-U": 0.3013966,
            "H-0": 0.0010000,
            "H-A": 0.0001768,
            "H-C": 0.0002387,
            "H-G": 0.0002917,
            "H-T": 0.0002917,
            "H-B": 0.1766463,
            "H-D": 0.2385224,
            "H-H": 0.2914165,
            "H-U": 0.2914155,
            "U-0": 0.0010000,
            "U-A": 0.0002477,
            "U-C": 0.0002457,
            "U-G": 0.0002977,
            "U-T": 0.0002077,
            "U-B": 0.2475044,
            "U-D": 0.2455084,
            "U-H": 0.2974035,
            "U-U": 0.2075844,
        },
        # Emission Matrix
        {
            "A-a": 1.0,
            "A-c": 0.0,
            "A-g": 0.0,
            "A-t": 0.0,
            "C-a": 0.0,
            "C-c": 1.0,
            "C-g": 0.0,
            "C-t": 0.0,
            "G-a": 0.0,
            "G-c": 0.0,
            "G-g": 1.0,
            "G-t": 0.0,
            "T-a": 0.0,
            "T-c": 0.0,
            "T-g": 0.0,
            "T-t": 1.0,
            "B-a": 1.0,
            "B-c": 0.0,
            "B-g": 0.0,
            "B-t": 0.0,
            "D-a": 0.0,
            "D-c": 1.0,
            "D-g": 0.0,
            "D-t": 0.0,
            "H-a": 0.0,
            "H-c": 0.0,
            "H-g": 1.0,
            "H-t": 0.0,
            "U-a": 0.0,
            "U-c": 0.0,
            "U-g": 0.0,
            "U-t": 1.0,
        },
        # Initial Probabilities
        {
            "0": 1.0,
            "A": 0.0,
            "C": 0.0,
            "G": 0.0,
            "T": 0.0,
            "B": 0.0,
            "D": 0.0,
            "H": 0.0,
            "U": 0.0,
        },
    )
    return hmm_cpg_islands


if __name__ == "__main__":

    np.seterr(divide="ignore", invalid="ignore")

    hmm = create_hmm_cpg_islands()

    observation = "ccg"

    score, path = hmm.viterbi_log(observation)

    print(f"Score: {score}\n" f"Path: {path}\n" f"Observation: {observation}")
