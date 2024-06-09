import numpy as np
import matplotlib.pyplot as plt

def initialize_lattice(N):
    """ Initialize a NxN lattice with random spins """
    return np.random.choice([-1, 1], size=(N, N))

def delta_energy(lattice, i, j, J=1):
    """ Calculate the change in energy if spin at (i, j) is flipped """
    S = lattice[i, j]
    nb = lattice[(i-1) % N, j] + lattice[(i+1) % N, j] + lattice[i, (j-1) % N] + lattice[i, (j+1) % N]
    return 2 * J * S * nb

def metropolis_step(lattice, N, T):
    """ Perform one Metropolis update step """
    for _ in range(N**2):  # N^2 updates per step
        i, j = np.random.randint(0, N, 2)
        dE = delta_energy(lattice, i, j)
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            lattice[i, j] *= -1

def calculate_energy(lattice):
    """ Calculate the total energy of the lattice """
    E = 0
    for i in range(N):
        for j in range(N):
            S = lattice[i, j]
            E -= S * (lattice[(i-1) % N, j] + lattice[i, (j-1) % N])
    return E / 2  # Each pair counted twice

def simulate(N, T_range, steps, equilibration_steps):
    energies = []
    energy_squares = []
    for T in T_range:
        lattice = initialize_lattice(N)
        # Equilibrate the system
        for _ in range(equilibration_steps):
            metropolis_step(lattice, N, T)
        # Collect energy data
        local_energies = []
        for _ in range(steps):
            metropolis_step(lattice, N, T)
            E = calculate_energy(lattice)
            local_energies.append(E)
        avg_E = np.mean(local_energies)
        avg_E2 = np.mean([e**2 for e in local_energies])
        energies.append(avg_E)
        energy_squares.append(avg_E2)
    # Calculate specific heat
    specific_heats = [(E2 - E**2) / (N**2 * T**2) for E, E2, T in zip(energies, energy_squares, T_range)]
    return specific_heats

# Parameters
N = 20  # Lattice size
T_range = np.linspace(1.5, 3.5, 20)  # Temperature range
steps = 1000
equilibration_steps = 5000

# Run simulation
specific_heats = simulate(N, T_range, steps, equilibration_steps)

# Plotting results
plt.figure(figsize=(8, 5))
plt.plot(T_range, specific_heats, marker='o', linestyle='-', color='red')
plt.xlabel('Temperature')
plt.ylabel('Specific Heat')
plt.title('Specific Heat vs. Temperature')
plt.grid(True)
plt.show()
