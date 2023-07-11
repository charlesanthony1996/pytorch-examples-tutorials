import numpy as np
import matplotlib.pyplot as plt


# parameters
frequency = 5
phase_shift = np.pi / 4
time = np.linspace(0, 1, 500, endpoint = False)

# baseband signal

baseband_signal = np.sin(2 * np.pi * frequency * time)

print(baseband_signal)

# shifted signal

shifted_signal = np.sin(2 * np.pi * frequency * time + phase_shift)

print(shifted_signal)

# plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)

plt.plot(time, baseband_signal, label="baseband signal")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, shifted_signal, label="shifted signal")
plt.legend()



plt.show()
