import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
L = 50  # Linear size of the grid
J = 1  # Interaction strength
k_B = 1  # Boltzmann constant
n_steps = 10000  # Number of Monte Carlo steps
equilibration_steps = 2000  # Steps to skip for equilibration

# Function to calculate energy
def calc_energy(config):
    energy = 0
    for i in range(L):
        for j in range(L):
            S = config[i, j]
            neighbors = config[(i+1) % L, j] + config[i, (j+1) % L] + config[(i-1) % L, j] + config[i, (j-1) % L]
            energy += -J * neighbors * S
    return energy / 2  # Each pair counted twice

# Metropolis algorithm
def metropolis(config, beta):
    for i in range(L):
        for j in range(L):
            x = np.random.randint(0, L)
            y = np.random.randint(0, L)
            S = config[x, y]
            neighbors = config[(x+1) % L, y] + config[x, (y+1) % L] + config[(x-1) % L, y] + config[x, (y-1) % L]
            dE = 2 * J * S * neighbors
            if dE < 0 or np.random.rand() < np.exp(-dE * beta):
                config[x, y] *= -1
    return config

# Create initial configuration
def initialize_spins(L):
    return np.random.choice([1, -1], size=(L, L))

# Simulation to calculate average energy at different temperatures
def calculate_average_energy(temperatures, n_steps, equilibration_steps):
    avg_energies = []
    for temp in temperatures:
        config = initialize_spins(L)
        beta = 1 / (k_B * temp)
        energies = []
        for step in range(n_steps):
            metropolis(config, beta)
            if step >= equilibration_steps:
                energies.append(calc_energy(config))
        avg_energy = np.mean(energies)
        avg_energies.append(avg_energy)
    return avg_energies

# Temperatures to simulate
temperatures = [5.00, 2.27, 1.00]

# Run simulations
average_energies = calculate_average_energy(temperatures, n_steps, equilibration_steps)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(temperatures, average_energies, marker='o')
plt.title('Average Energy vs Temperature for the Ising Model')
plt.xlabel('Temperature (T)')
plt.ylabel('Average Energy')
plt.grid(True)
plt.show()
