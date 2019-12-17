import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

title = "Profile HMM (Sequence Length vs Time)"
numba_python_filename = "numba_python_durations.txt"
numba_numpy_filename = "numba_numpy_durations.txt"
pypy_python_filename = "pypy_python_durations.txt"
pypy_numpy_filename = "pypy_numpy_durations.txt"


def load_results(filename):
    with open(filename, "r") as file:
        durations = file.readlines()
    durations = np.array([float(duration) for duration in durations], np.float64)
    return durations


numba_python_durations = load_results(numba_python_filename)
numba_numpy_durations = load_results(numba_numpy_filename)
pypy_python_durations = load_results(pypy_python_filename)
pypy_numpy_durations = load_results(pypy_numpy_filename)

sns.set_style("darkgrid")
plt.plot(numba_python_durations, label="Numba Python")
plt.plot(numba_numpy_durations, label="Numba NumPy")
plt.plot(pypy_python_durations, label="PyPy Python")
plt.plot(pypy_numpy_durations, label="PyPy NumPy")
plt.xlabel("Sequence Lengths")
plt.ylabel("Duration (in seconds)")
plt.legend(loc="upper left")
plt.title(title)
plt.savefig("mbp_nb_pypy_phmm.png", dpi=300)
plt.show()
