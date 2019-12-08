from hmm.hmm_jhu import HMM as HMMJHU
from hmm.hmm_py import HMM as HMMPy
from hmm.hmm_jhu_profile import HMM as HMMJHUProfile
from hmm.hmm_py_profile import HMM as HMMPyProfile


def main():
    t = {
        "A-A": 0.5,
        "A-B": 0.33,
        "A-D": 0.17,
        "B-B": 0.5,
        "B-A": 0.17,
        "B-D": 0.33,
        "D-D": 0.5,
        "D-A": 0.33,
        "D-B": 0.17,
    }

    e = {"A-0": 0.5, "A-1": 0.5, "B-0": 0.75, "B-1": 0.25, "D-0": 0.25, "D-1": 0.75}

    i = {"A": 1.0, "B": 0.0, "D": 0.0}

    seq = "0110110111"
    hmm_jhu = HMMJHU(t, e, i)
    hmm_py = HMMPy(t, e, i)
    hmm_jhu_profile = HMMJHUProfile(t, e, i)
    hmm_py_profile = HMMPyProfile(t, e, i)

    if not (str(hmm_jhu) == str(hmm_py) == str(hmm_jhu_profile) == str(hmm_py_profile)):
        raise Exception

    if hmm_jhu.viterbi(seq) != hmm_py.viterbi(seq):
        raise Exception

    if hmm_jhu.viterbi_log(seq) != hmm_py.viterbi_log(seq):
        raise Exception

    if hmm_jhu_profile.viterbi(seq) != hmm_py_profile.viterbi(seq):
        raise Exception

    if hmm_jhu_profile.viterbi_log(seq) != hmm_py_profile.viterbi_log(seq):
        raise Exception


if __name__ == "__main__":
    main()
