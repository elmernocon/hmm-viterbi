import unittest


class HMMTestCase(unittest.TestCase):

    def test_hmm_jhu_repr(self):

        from hmm.hmm_jhu import HMM

        hmm = HMM(
            {"FF": 0.85, "FL": 0.15, "LF": 0.1, "LL": 0.9},         # Transition matrix
            {"FH": 0.65, "FT": 0.35, "LH": 0.75, "LT": 0.25},       # Emission matrix
            {"F": 0.5, "L": 0.5})                                   # Initial probabilities

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
L  0.5"
        r_a = str(hmm)

        self.assertEqual(r_e, r_a)

    def test_hmm_jhu_joint_probability(self):

        from hmm.hmm_jhu import HMM

        hmm = HMM(
            {"FF": 0.9, "FL": 0.1, "LF": 0.1, "LL": 0.9},           # Transition matrix
            {"FH": 0.5, "FT": 0.5, "LH": 0.75, "LT": 0.25},         # Emission matrix
            {"F": 0.5, "L": 0.5})                                   # Initial probabilities

        jp_e = (0.5 ** 9) * (0.75 ** 3) * (0.9 ** 8) * (0.1 ** 2)
        jp_a = hmm.joint_probability("FFFLLLFFFFF", "THTHHHTHTTH")

        self.assertAlmostEqual(jp_e, jp_a)

    def test_hmm_jhu_joint_probability_log(self):

        import math
        from hmm.hmm_jhu import HMM

        hmm = HMM(
            {"FF": 0.9, "FL": 0.1, "LF": 0.1, "LL": 0.9},           # Transition matrix
            {"FH": 0.5, "FT": 0.5, "LH": 0.75, "LT": 0.25},         # Emission matrix
            {"F": 0.5, "L": 0.5})                                   # Initial probabilities

        jp_e = math.log2((0.5 ** 9) * (0.75 ** 3) * (0.9 ** 8) * (0.1 ** 2))
        jp_a = hmm.joint_probability_log("FFFLLLFFFFF", "THTHHHTHTTH")

        self.assertAlmostEqual(jp_e, jp_a)

    def test_hmm_jhu_hmm(self):

        from hmm.hmm_jhu import HMM

        hmm = HMM(
            {"FF": 0.6, "FL": 0.4, "LF": 0.4, "LL": 0.6},           # Transition matrix
            {"FH": 0.5, "FT": 0.5, "LH": 0.8, "LT": 0.2},           # Emission matrix
            {"F": 0.5, "L": 0.5})                                   # Initial probabilities

        score_e, path_e = 2.8665446400000001e-06, "FFFLLLFFFFL"
        score_a, path_a = hmm.viterbi("THTHHHTHTTH")

        self.assertAlmostEqual(score_e, score_a)
        self.assertEqual(path_e, path_a)

    def test_hmm_jhu_hmm_log(self):

        import math
        from hmm.hmm_jhu import HMM

        hmm = HMM(
            {"FF": 0.6, "FL": 0.4, "LF": 0.4, "LL": 0.6},           # Transition matrix
            {"FH": 0.5, "FT": 0.5, "LH": 0.8, "LT": 0.2},           # Emission matrix
            {"F": 0.5, "L": 0.5})                                   # Initial probabilities

        score_e, path_e = math.log2(2.8665446400000001e-06), "FFFLLLFFFFL"
        score_a, path_a = hmm.viterbi_log("THTHHHTHTTH")

        self.assertAlmostEqual(score_e, score_a)
        self.assertEqual(path_e, path_a)

    def test_hmm_jhu_hmm_underflow(self):

        from hmm.hmm_jhu import HMM

        hmm = HMM(
            {"FF": 0.6, "FL": 0.4, "LF": 0.4, "LL": 0.6},           # Transition matrix
            {"FH": 0.5, "FT": 0.5, "LH": 0.8, "LT": 0.2},           # Emission matrix
            {"F": 0.5, "L": 0.5})                                   # Initial probabilities

        score_e, path_e = 0.0,\
                          "FFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
        score_a, path_a = hmm.viterbi("THTHHHTHTTH" * 100)

        self.assertAlmostEqual(score_e, score_a)
        self.assertEqual(path_e, path_a)

        score_e, path_e = -1824.4030071946879,\
                          "FFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFFFFFLLLFFFFL"
        score_a, path_a = hmm.viterbi_log("THTHHHTHTTH" * 100)

        self.assertAlmostEqual(score_e, score_a)
        self.assertEqual(path_e, path_a)


if __name__ == '__main__':
    unittest.main()
