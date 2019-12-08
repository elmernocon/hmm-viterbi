import unittest


class HMMTestCase(unittest.TestCase):
    def test_hmm_jhu_repr(self):

        from hmm.hmm_jhu import HMM

        hmm = HMM(
            {"F-F": 0.85, "F-L": 0.15, "L-F": 0.1, "L-L": 0.9},  # Transition matrix
            {"F-H": 0.65, "F-T": 0.35, "L-H": 0.75, "L-T": 0.25},  # Emission matrix
            {"F": 0.5, "L": 0.5},
        )  # Initial probabilities

        r_e = "States: ['F', 'L']\n\
Symbols: ['H', 'T']\n\
\n\
Transition Matrix:\n\
      F     L\n\
F  0.85  0.15\n\
L  0.10  0.90\n\
\n\
Emission Matrix:\n\
      H     T\n\
F  0.65  0.35\n\
L  0.75  0.25\n\
\n\
Initial Probability:\n\
     %\n\
F  0.5\n\
L  0.5\n\
\n\
Transition Matrix (log2):\n\
          F         L\n\
F -0.234465 -2.736966\n\
L -3.321928 -0.152003\n\
\n\
Emission Matrix (log2):\n\
          H         T\n\
F -0.621488 -1.514573\n\
L -0.415037 -2.000000\n\
\n\
Initial Probability (log2):\n\
     %\n\
F -1.0\n\
L -1.0"
        r_a = str(hmm)

        self.assertEqual(r_e, r_a)

    def test_hmm_jhu_joint_probability(self):

        from hmm.hmm_jhu import HMM

        hmm = HMM(
            {"F-F": 0.9, "F-L": 0.1, "L-F": 0.1, "L-L": 0.9},  # Transition matrix
            {"F-H": 0.5, "F-T": 0.5, "L-H": 0.75, "L-T": 0.25},  # Emission matrix
            {"F": 0.5, "L": 0.5},
        )  # Initial probabilities

        jp_e = (0.5 ** 9) * (0.75 ** 3) * (0.9 ** 8) * (0.1 ** 2)
        jp_a = hmm.joint_probability("FFFLLLFFFFF", "THTHHHTHTTH")

        self.assertAlmostEqual(jp_e, jp_a)

    def test_hmm_jhu_joint_probability_log(self):

        import math
        from hmm.hmm_jhu import HMM

        hmm = HMM(
            {"F-F": 0.9, "F-L": 0.1, "L-F": 0.1, "L-L": 0.9},  # Transition matrix
            {"F-H": 0.5, "F-T": 0.5, "L-H": 0.75, "L-T": 0.25},  # Emission matrix
            {"F": 0.5, "L": 0.5},
        )  # Initial probabilities

        jp_e = math.log2((0.5 ** 9) * (0.75 ** 3) * (0.9 ** 8) * (0.1 ** 2))
        jp_a = hmm.joint_probability_log("FFFLLLFFFFF", "THTHHHTHTTH")

        self.assertAlmostEqual(jp_e, jp_a)

    def test_hmm_jhu_hmm(self):

        from hmm.hmm_jhu import HMM

        hmm = HMM(
            {"F-F": 0.6, "F-L": 0.4, "L-F": 0.4, "L-L": 0.6},  # Transition matrix
            {"F-H": 0.5, "F-T": 0.5, "L-H": 0.8, "L-T": 0.2},  # Emission matrix
            {"F": 0.5, "L": 0.5},
        )  # Initial probabilities

        score_e, path_e = 2.8665446400000001e-06, "FFFLLLFFFFL"
        score_a, path_a = hmm.viterbi("THTHHHTHTTH")

        self.assertAlmostEqual(score_e, score_a)
        self.assertEqual(path_e, path_a)

    def test_hmm_jhu_hmm_log(self):

        import math
        from hmm.hmm_jhu import HMM

        hmm = HMM(
            {"F-F": 0.6, "F-L": 0.4, "L-F": 0.4, "L-L": 0.6},  # Transition matrix
            {"F-H": 0.5, "F-T": 0.5, "L-H": 0.8, "L-T": 0.2},  # Emission matrix
            {"F": 0.5, "L": 0.5},
        )  # Initial probabilities

        score_e, path_e = math.log2(2.8665446400000001e-06), "FFFLLLFFFFL"
        score_a, path_a = hmm.viterbi_log("THTHHHTHTTH")

        self.assertAlmostEqual(score_e, score_a)
        self.assertEqual(path_e, path_a)

    def test_hmm_jhu_hmm_underflow(self):

        from hmm.hmm_jhu import HMM

        hmm = HMM(
            {"F-F": 0.6, "F-L": 0.4, "L-F": 0.4, "L-L": 0.6},  # Transition matrix
            {"F-H": 0.5, "F-T": 0.5, "L-H": 0.8, "L-T": 0.2},  # Emission matrix
            {"F": 0.5, "L": 0.5},
        )  # Initial probabilities

        score_e, path_e = (
            0.0,
            "FFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        )
        score_a, path_a = hmm.viterbi("THTHHHTHTTH" * 100)

        self.assertAlmostEqual(score_e, score_a)
        self.assertEqual(path_e, path_a)

        score_e, path_e = (
            -1824.4030071946879,
            "FFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFL",
        )
        score_a, path_a = hmm.viterbi_log("THTHHHTHTTH" * 100)

        self.assertAlmostEqual(score_e, score_a)
        self.assertEqual(path_e, path_a)


if __name__ == "__main__":
    unittest.main()
