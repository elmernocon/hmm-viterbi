import timeit


def benchmark_base():

    setup = '''
from hmm.hmm_jhu import HMM

hmm = HMM(
    {"FF": 0.6, "FL": 0.4, "LF": 0.4, "LL": 0.6},           # Transition matrix
    {"FH": 0.5, "FT": 0.5, "LH": 0.8, "LT": 0.2},           # Emission matrix
    {"F": 0.5, "L": 0.5})                                   # Initial probabilities
'''

    stmt = '''
hmm.viterbi("THTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTH")
'''

    print(timeit.timeit(
        setup=setup,
        stmt=stmt,
        number=100))


def benchmark_numba():

    setup = '''
import warnings
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from hmm.hmm_jhu import HMM

hmm = HMM(
    {"FF": 0.6, "FL": 0.4, "LF": 0.4, "LL": 0.6},           # Transition matrix
    {"FH": 0.5, "FT": 0.5, "LH": 0.8, "LT": 0.2},           # Emission matrix
    {"F": 0.5, "L": 0.5})                                   # Initial probabilities

hmm.viterbi_numba("T")
    '''

    stmt = '''
hmm.viterbi_numba("THTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTH")
    '''

    print(timeit.timeit(
        setup=setup,
        stmt=stmt,
        number=100))


if __name__ == '__main__':
    benchmark_base()
    benchmark_numba()
