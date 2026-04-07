import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def read_frames(filename):
    """Read file where timesteps are separated by blank lines."""
    frames = []
    current = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    frames.append(np.array(current))
                    current = []
            else:
                current.append([float(v) for v in line.split()])
    if current:
        frames.append(np.array(current))
    return frames

frames = read_frames("tests/test0.txt")

fig, ax = plt.subplots()
scatter = ax.scatter([], [])

# set axis limits from data
all_data = np.concatenate(frames)
margin = 10
ax.set_xlim(all_data[:, 0].min() - margin, all_data[:, 0].max() + margin)
ax.set_ylim(all_data[:, 1].min() - margin, all_data[:, 1].max() + margin)
ax.set_aspect('equal')

def update(i):
    frame = frames[i]
    scatter.set_offsets(frame[:, :2])          # x, y
    scatter.set_sizes(frame[:, 3] * 20)        # scale mass to dot size
    ax.set_title(f"Step {i}/{len(frames)-1}")
    return scatter,

ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
plt.show()