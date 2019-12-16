Hidden Markov Model and Viterbi Algorithm: An optimization using Numba and PyPy
====

## Overview

What if we are given a sequence of observations over time, and we are interested in discovering the reason behind such a sequence, or predicting what the next state will be? In this article, we provide simple notes on the concept of Hidden Markov Model (HMM) -- a formalism to reason about states over time that is specifically aimed at decode a series of \textit{hidden} states from a series of observations. We also go through a fast decoding algorithm for HMM, called the \textit{Viterbi} algorithm, and implement in Python with the aid of \textit{just-in-time} (JIT) compilers to further improve the speed of the said algorithm.

## Usage

It is recommended to create a virtual environment for using this repository.

```buildoutcfg
$ virtualenv venv --python=python3 
$ source venv/bin/activate
```

Then, install the dependencies inside the virtual environment,

```buildoutcfg
$ pip install -r requirements.txt
```

To run the HMM on Dishonest Casino, the following are the program parameters,

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

For the Profile HMM, the following are the parameters

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

In our study, we used the following arguments for benchmarking,

For HMM,

```buildoutcfg
$ python3 benchmark.py --interval 10 100 10
```

For Profile HMM,

```buildoutcfg
$ python3 benchmarkf.py --interval 1 20 1
```

The seed values are already set in both benchmarking modules for reproducibility of pseudorandomly generated sequences for testing.

To use our HMM implementations in a different Python program,

```python3
>>> from hmm.hmm_jhu import HMM
>>> hmm = HMM(transition_matrix: dict, emmision_matrx: dict, initial_probabilities: dict)
>>> hmm.viterbi(observation: str)
```
