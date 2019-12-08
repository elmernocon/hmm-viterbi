from hmm.hmm_jhu import HMM as HMMJHU
from hmm.hmm_profile import HMM as HMMProfile
from hmm.hmm_py import HMM as HMMPy

if __name__ == "__main__":

    t = {
        "A-A": 0.5,
        "A-B": 0.33,
        "A-D": 0.17,
        "B-B": 0.5,
        "B-A": 0.17,
        "B-D": 0.33,
        "D-D": 0.5,
        "D-A": 0.33,
        "D-B": 0.17
    }

    e = {
        "A-0": 0.5,
        "A-1": 0.5,
        "B-0": 0.75,
        "B-1": 0.25,
        "D-0": 0.25,
        "D-1": 0.75
    }

    i = {
        "A": 1.0,
        "B": 0.0,
        "D": 0.0
    }

    seq = "0110110111"
    hmm_jhu = HMMJHU(t, e, i)
    hmm_profile = HMMProfile(t, e, i)
    hmm_py = HMMPy(t, e, i)

    str_hmm_jhu = str(hmm_jhu)
    str_hmm_profile = str(hmm_profile)
    str_hmm_py = str(hmm_py)

    if not (str_hmm_jhu == str_hmm_profile == str_hmm_py):
        raise Exception

    pb_hmm_jhu, pt_hmm_jhu = hmm_jhu.viterbi(seq)
    pb_hmm_profile, pt_hmm_profile = hmm_profile.viterbi(seq)
    pb_hmm_py, pt_hmm_py = hmm_py.viterbi(seq)

    if not ((pb_hmm_jhu, pt_hmm_jhu) == (pb_hmm_py, pt_hmm_py)):
        raise Exception

    print(pb_hmm_jhu, pt_hmm_jhu)
    print(pb_hmm_profile, pt_hmm_profile)
    print(pb_hmm_py, pt_hmm_py)
