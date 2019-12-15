import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

title = "Profile HMM (Sequence Length vs Time)"
python_filename = "benchmarkf_python_durations.txt"
numpy_filename = "benchmarkf_numpy_durations.txt"
py_nb_filename = "benchmarkf_python_numba_durations.txt"
np_nb_filename = "benchmarkf_numpy_numba_durations.txt"


def load_results(filename):
    with open(filename, "r") as file:
        durations = file.readlines()
    durations = np.array(
            [float(duration) for duration in durations], np.float64
            )
    return durations


python_durations = load_results(python_filename)
numpy_durations = load_results(numpy_filename)
python_numba_durations = load_results(py_nb_filename)
numpy_numba_durations = load_results(np_nb_filename)

sns.set_style("darkgrid")
plt.plot(python_durations, label="Base Python")
plt.plot(numpy_durations, label="Base NumPy")
plt.plot(python_numba_durations, label="Numba Python")
plt.plot(numpy_numba_durations, label="Numba NumPy")
plt.xlabel("Sequence Lengths")
plt.ylabel("Duration (in seconds)")
plt.legend(loc="upper left")
plt.title(title)
plt.savefig("dell_cpython_phmm.png", dpi=300)
plt.show()
