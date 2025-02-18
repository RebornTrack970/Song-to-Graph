import librosa
import matplotlib.pyplot as plt
import numpy as np
import time

# Load the MP3 file
file_path = "path/to/.mp3"  # Replace with the path to your file
y, sr = librosa.load(file_path, sr=None, mono=True)

hop_length = 512
frame_length = 1024
energy = np.array([
    sum(abs(y[i:i + frame_length] ** 2))
    for i in range(0, len(y), hop_length)
])

times = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=hop_length)

fig, ax = plt.subplots()
ax.set_title(input("Song name: "))
ax.set_xlabel('Seconds')
ax.set_ylabel('Energy')
line, = ax.plot([], [], 'b-', linewidth=1)

window_size = 5  # seconds
ax.set_xlim(0, window_size)
ax.set_ylim(0, np.max(energy) * 1.1)

def update_plot():
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        current_time = min(elapsed_time, times[-1])
        idx = np.searchsorted(times, current_time)
        window_start_time = max(0, current_time - window_size)
        window_end_time = current_time
        start_idx = np.searchsorted(times, window_start_time)
        end_idx = np.searchsorted(times, window_end_time)
        if end_idx > 0:
            line.set_data(times[start_idx:end_idx], energy[start_idx:end_idx])
            ax.set_xlim(window_start_time, window_end_time)
            for collection in ax.collections:
                collection.remove()
            ax.fill_between(
                times[start_idx:end_idx],
                energy[start_idx:end_idx],
                color='blue',
                alpha=0.3
            )
        if current_time >= times[-1]:
            break
        plt.pause(0.05)


plt.ion()
update_plot()
plt.ioff()
plt.show()
