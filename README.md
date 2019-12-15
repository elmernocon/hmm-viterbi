Hidden Markov Model and Viterbi Algorithm: An optimization using Numba and PyPy
====

## Usage

To run the HMM on Dishonest Casino, 

```buildoutcfg
usage: benchmark.py [-h] [-e EXECUTIONS] [-o OBSERVATION] [-r GENERATE_RANDOM]
[-i INTERVAL [INTERVAL ...]] [-x EXPORT]

Benchmarking HMM for Dishonest Casino

optional arguments:
-h, --help            show this help message and exit

Parameters:
-e EXECUTIONS, --executions EXECUTIONS
The number of times to run the program.
-o OBSERVATION, --observation OBSERVATION
The observed sequence.
-r GENERATE_RANDOM, --generate_random GENERATE_RANDOM
The length of generated observed sequence.
-i INTERVAL [INTERVAL ...], --interval INTERVAL [INTERVAL ...]
Range of sequence lengths, e.g. (10, 100, 10) would
produce sequence lengths of 10 to 90 with delta=10.
-x EXPORT, --export EXPORT
Export the results to a file.
```

```buildoutcfg
usage: benchmarkf.py [-h] [-i INTERVAL [INTERVAL ...]] [-x EXPORT]

Benchmarking Profile HMM

optional arguments:
  -h, --help            show this help message and exit

Parameters:
  -i INTERVAL [INTERVAL ...], --interval INTERVAL [INTERVAL ...]
                        Range of sequence lengths, e.g. (10, 100, 10) would
                        produce sequence lengths of 10 to 90 with delta=10.
  -x EXPORT, --export EXPORT
                        Export the results to a file.
```

