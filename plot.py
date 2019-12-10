import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open("numpy_durations.txt", "r") as file:
    numpy_durations = file.readlines()

with open("python_durations.txt", "r") as file:
    python_durations = file.readlines()

numpy_durations = np.array(
    [float(duration) for duration in numpy_durations], np.float64
)
python_durations = np.array(
    [float(duration) for duration in python_durations], np.float64
)

sns.set_style("darkgrid")
plt.plot(numpy_durations, label="Base NumPy")
plt.plot(python_durations, label="Base Python")
plt.xlabel("Sequence Lengths")
plt.ylabel("Duration (in seconds)")
plt.legend(loc="upper left")
plt.show()
