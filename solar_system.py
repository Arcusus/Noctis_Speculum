"""Simple n-body simulation of the inner Solar System.

The model is intentionally lightweight so it can run in environments like
PyScript.  A basic Euler integrator is used; it is not meant for scientific
accuracy but provides a reasonable visualisation of planetary motion.
"""

import numpy as np

# numba is optional; fall back to pure Python if unavailable
try:  # pragma: no cover - environments without numba
    from numba import njit
except Exception:  # pragma: no cover
    def njit(func):
        return func


# Physical constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
AU = 1.495978707e11  # Astronomical unit (m)
DAY = 86400.0


# Masses (kg) of Sun, Mercury, Venus, Earth, Moon and Mars respectively
SUN_IDX = 0
EARTH_IDX = 3
masses = np.array([
    1.9885e30,   # Sun
    3.3011e23,   # Mercury
    4.8675e24,   # Venus
    5.97219e24,  # Earth
    7.34767309e22,  # Moon
    6.4171e23,   # Mars
], dtype=np.float64)


# Initial positions (m) in a heliocentric frame at t=0.
# Earth starts on the positive x-axis; other planets are placed for demo.
positions = np.array([
    [0.0, 0.0, 0.0],             # Sun
    [0.0, 0.387 * AU, 0.0],      # Mercury
    [0.0, 0.723 * AU, 0.0],      # Venus
    [AU, 0.0, 0.0],              # Earth
    [AU + 0.00257 * AU, 0.0, 0.0],  # Moon (approximate)
    [0.0, 1.524 * AU, 0.0],      # Mars
], dtype=np.float64)


# Initial velocities (m/s) approximating circular orbits around the Sun
velocities = np.array([
    [0.0, 0.0, 0.0],
    [np.sqrt(G * masses[SUN_IDX] / (0.387 * AU)), 0.0, 0.0],
    [np.sqrt(G * masses[SUN_IDX] / (0.723 * AU)), 0.0, 0.0],
    [0.0, np.sqrt(G * masses[SUN_IDX] / AU), 0.0],
    [0.0, np.sqrt(G * masses[SUN_IDX] / AU) + np.sqrt(G * masses[EARTH_IDX] / (0.00257 * AU)), 0.0],
    [np.sqrt(G * masses[SUN_IDX] / (1.524 * AU)), 0.0, 0.0],
], dtype=np.float64)


@njit
def step(pos, vel, masses, dt):
    """Advance the system by one time step using pairwise gravitation."""
    n = masses.shape[0]
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pos[j] - pos[i]
            dist_sq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
            dist = np.sqrt(dist_sq)
            inv_dist3 = 1.0 / (dist_sq * dist)
            force = G * masses[i] * masses[j] * inv_dist3 * diff
            acc[i] += force / masses[i]
            acc[j] -= force / masses[j]
    vel += acc * dt
    pos += vel * dt
    return pos, vel


def simulate(days, dt_minutes=60, center_idx=SUN_IDX):
    """Simulate for ``days`` and return positions relative to ``center_idx``."""
    dt = dt_minutes * 60.0
    steps = int(days * DAY / dt)
    pos = positions.copy()
    vel = velocities.copy()
    out = np.empty((steps, len(masses), 3), dtype=np.float64)
    for i in range(steps):
        pos, vel = step(pos, vel, masses, dt)
        out[i] = pos - pos[center_idx]
    return out


if __name__ == "__main__":
    # Run a simple demonstration when executed directly
    data = simulate(365, dt_minutes=60, center_idx=SUN_IDX)
    print("Simulated steps:", len(data))
    print("Final positions relative to the Sun (m):")
    print(data[-1])
