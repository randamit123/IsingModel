import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
L = 50  # Linear size of the grid
J = 1  # Interaction strength
k_B = 1  # Boltzmann constant
n_steps = 10000  # Number of Monte Carlo steps

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
    for i in range(L):
        for j in range(L):
            x = np.random.randint(0, L)
            y = np.random.randint(0, L)
            S = config[x, y]
            neighbors = config[(x+1) % L, y] + config[x, (y+1) %
                                                      L] + config[(x-1) % L, y] + config[x, (y-1) % L]
            '''
            dE = 2 * S * neighbors
            if dE < 0:
                S *= -1
            elif np.random.rand() < np.exp(-dE * beta):
                S *= -1
            config[x, y] = S
            '''
            dE = 2 * S * neighbors
            if dE < 0 or np.random.rand() < np.exp(-dE * beta):
                config[x, y] *= -1
    return config

# Create initial configuration


def initialize_spins(L):
    return np.random.choice([1, -1], size=(L, L))

# Function to update the frame for animation


def update(frame, config, im, beta):
    metropolis(config, beta)
    im.set_array(config)
    return im,

# Main function to create and save the animation


def animate_ising(temperatures, n_frames, save_path):
    fig, ax = plt.subplots()
    config = initialize_spins(L)
    im = ax.imshow(config, animated=True, cmap='coolwarm')
    ani = animation.FuncAnimation(fig, update, frames=n_frames, fargs=(
        config, im, 1/temperatures[0]), interval=100, repeat=False)

    # Update temperature for each phase of the animation
    def update_temp(frame, config, im):
        if frame < n_frames / 3:
            beta = 1 / temperatures[0]
        elif frame < 2 * n_frames / 3:
            beta = 1 / temperatures[1]
        else:
            beta = 1 / temperatures[2]
        metropolis(config, beta)
        im.set_array(config)
        return im,

    ani = animation.FuncAnimation(fig, update_temp, frames=n_frames, fargs=(
        config, im), interval=100, repeat=False)
    ani.save(save_path, writer='imagemagick')
    plt.show()


# Parameters for animation
# Below, at, and above the critical temperature
temperatures = [5.0, 2.27, 1.0]
n_frames = 300
save_path = 'ising_model.gif'

# Run the animation
animate_ising(temperatures, n_frames, save_path)
