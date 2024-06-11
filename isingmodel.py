import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
L = 50  # Linear size of the grid
J = 1  # Interaction strength
k_B = 1  # Boltzmann constant

# Function to calculate energy


def calc_energy(config):
    energy = 0
    for i in range(L):
        for j in range(L):
            S = config[i, j]
            neighbors = config[(i+1) % L, j] + config[i, (j+1) %
                                                      L] + config[(i-1) % L, j] + config[i, (j-1) % L]
            energy += -neighbors * S
    return energy / 2  # Each pair counted twice

# Function to calculate magnetization


def calc_magnetization(config):
    return np.sum(config)

# Metropolis algorithm


def metropolis(config, beta):
    for _ in range(L**2):  # Attempt as many updates as there are spins
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
        S = config[x, y]
        neighbors = config[(x+1) % L, y] + config[x, (y+1) %
                                                  L] + config[(x-1) % L, y] + config[x, (y-1) % L]
        dE = 2 * S * neighbors
        if dE < 0 or np.random.rand() < np.exp(-dE * beta):
            config[x, y] *= -1
    return config

# Create initial configuration


def initialize_spins(L):
    return np.random.choice([1, -1], size=(L, L))

# Update the frame for animation


def update_temp(frame, config, im, temperatures, n_frames):
    if frame < n_frames / 3:
        beta = 1 / temperatures[0]
    elif frame < 2 * n_frames / 3:
        if len(temperatures) > 1:
            beta = 1 / temperatures[1]
        else:
            # Use first temp if only one provided
            beta = 1 / temperatures[0]
    else:
        if len(temperatures) > 2:
            beta = 1 / temperatures[2]
        else:
            # Use second temp if only one provided
            beta = 1 / temperatures[0]
    metropolis(config, beta)
    im.set_array(config)
    return im,

# Create and save the animation


def animate_ising(temperatures, n_frames, save_path):
    fig, ax = plt.subplots()
    config = initialize_spins(L)
    im = ax.imshow(config, animated=True, cmap='coolwarm')

    ani = animation.FuncAnimation(
        fig, update_temp, frames=n_frames,
        fargs=(config, im, temperatures, n_frames), interval=100, repeat=False
    )

    # pillow writer instead of imagemagick
    ani.save(save_path, writer='pillow')
    plt.show(block=True)


# Parameters for animation
temperature_sets = [
    ([5.0, 2.27, 1.0], 'ising_model.gif'),
    ([5.0], 'ising_model_5.gif'),
    ([2.27], 'ising_model_2.27.gif'),
    ([1.0], 'ising_model_1.gif')
]
n_frames = 300

# Run the animation for each temperature set
for temperatures, save_path in temperature_sets:
    animate_ising(temperatures, n_frames, save_path)
