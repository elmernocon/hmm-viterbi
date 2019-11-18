import timeit


def benchmark_base(observation: str, number: int):

    setup = '''
from hmm.hmm_jhu import HMM

hmm = HMM(
    {"FF": 0.6, "FL": 0.4, "LF": 0.4, "LL": 0.6},           # Transition matrix
    {"FH": 0.5, "FT": 0.5, "LH": 0.8, "LT": 0.2},           # Emission matrix
    {"F": 0.5, "L": 0.5})                                   # Initial probabilities
'''

    stmt = f'hmm.viterbi("{observation}")'

    print(timeit.timeit(
        setup=setup,
        stmt=stmt,
        number=number))


def benchmark_numba(observation: str, number: int):

    setup = '''
import warnings
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from hmm.hmm_jhu import HMM

HMM.numba_warm_up()

hmm = HMM(
    {"FF": 0.6, "FL": 0.4, "LF": 0.4, "LL": 0.6},           # Transition matrix
    {"FH": 0.5, "FT": 0.5, "LH": 0.8, "LT": 0.2},           # Emission matrix
    {"F": 0.5, "L": 0.5})                                   # Initial probabilities
    '''

    stmt = f'hmm.viterbi_numba("{observation}")'

    print(timeit.timeit(
        setup=setup,
        stmt=stmt,
        number=number))


if __name__ == '__main__':

    obs, num = "THTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTHTHTHHHTHTTH", 100000

    benchmark_base(obs, num)
    benchmark_numba(obs, num)
