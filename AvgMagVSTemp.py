import numpy as np
import matplotlib.pyplot as plt


def initialize_lattice(N):
    """Initialize a NxN lattice with random spins."""
    return np.random.choice([-1, 1], size=(N, N))


def delta_energy(lattice, i, j, J=1):
    """Calculate the change in energy if spin at (i, j) is flipped."""
    S = lattice[i, j]
    nb = lattice[(i-1) % N, j] + lattice[(i+1) % N, j] + \
        lattice[i, (j-1) % N] + lattice[i, (j+1) % N]
    return 2 * J * S * nb


def metropolis_step(lattice, N, T):
    """Perform one Metropolis update step on the lattice."""
    for _ in range(N**2):
        i, j = np.random.randint(0, N, 2)
        dE = delta_energy(lattice, i, j)
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            lattice[i, j] *= -1


def calculate_magnetization(lattice):
    """Calculate the magnetization of the lattice."""
    return np.abs(np.sum(lattice)) / lattice.size


def simulate(N, T_range, steps, equilibration_steps):
    """Simulate the Ising model over a range of temperatures and return magnetizations."""
    magnetizations = []
    for T in T_range:
        lattice = initialize_lattice(N)
        # Equilibrate the system
        for _ in range(equilibration_steps):
            metropolis_step(lattice, N, T)
        # Measure magnetization
        m_avg = 0
        for _ in range(steps):
            metropolis_step(lattice, N, T)
            m_avg += calculate_magnetization(lattice)
        magnetizations.append(m_avg / steps)
    return magnetizations


# Parameters
N = 20  # Lattice size
T_range = np.linspace(1.5, 3.5, 20)  # Temperature range
steps = 1000  # Steps to average over after equilibration
equilibration_steps = 5000  # Steps to reach equilibrium

# Run simulation
magnetizations = simulate(N, T_range, steps, equilibration_steps)

# Plotting results
plt.figure(figsize=(8, 5))
plt.plot(T_range, magnetizations, marker='o', linestyle='-', color='b')
plt.xlabel('Temperature')
plt.ylabel('Average Magnetization')
plt.title('Average Magnetization vs. Temperature')
plt.grid(True)
plt.show()
