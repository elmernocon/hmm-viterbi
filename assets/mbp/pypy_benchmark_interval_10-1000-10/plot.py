import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

title = "HMM (Sequence Length vs Time)"
python_filename = "benchmark_python_durations.txt"
numpy_filename = "benchmark_numpy_durations.txt"


def load_results(filename):
    with open(filename, "r") as file:
        durations = file.readlines()
    durations = np.array([float(duration) for duration in durations], np.float64)
    return durations


python_durations = load_results(python_filename)
numpy_durations = load_results(numpy_filename)

sns.set_style("darkgrid")
plt.plot(python_durations, label="Base Python")
plt.plot(numpy_durations, label="Base NumPy")
plt.xlabel("Sequence Lengths")
plt.ylabel("Duration (in seconds)")
plt.legend(loc="upper left")
plt.title(title)
plt.savefig("mbp_pypy_hmm.png", dpi=300)
plt.show()
