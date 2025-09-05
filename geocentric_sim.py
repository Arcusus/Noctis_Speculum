import numpy as np
from numba import njit

# Gravitational constant
G = 6.67430e-11  # m^3 kg^-1 s^-2
AU = 1.495978707e11
DAY = 86400.0

# Masses: Sun, Mercury, Venus, Earth, Mars (kg)
# index: 0=Sun,1=Mercury,2=Venus,3=Earth,4=Mars
masses = np.array([
    1.9885e30,
    3.3011e23,
    4.8675e24,
    5.97219e24,
    6.4171e23
], dtype=np.float64)

# Initial positions (m) in heliocentric frame at t=0
# Earth at (AU,0,0); others arranged for demonstration
positions = np.array([
    [0.0, 0.0, 0.0],            # Sun
    [0.0, 0.387*AU, 0.0],       # Mercury
    [0.0, 0.723*AU, 0.0],       # Venus
    [AU, 0.0, 0.0],             # Earth
    [0.0, 1.524*AU, 0.0],       # Mars
], dtype=np.float64)

# Circular velocities around Sun
velocities = np.array([
    [0.0, 0.0, 0.0],
    [np.sqrt(G*masses[0]/(0.387*AU)), 0.0, 0.0],
    [np.sqrt(G*masses[0]/(0.723*AU)), 0.0, 0.0],
    [0.0, np.sqrt(G*masses[0]/AU), 0.0],
    [np.sqrt(G*masses[0]/(1.524*AU)), 0.0, 0.0],
], dtype=np.float64)

@njit
def step(pos, vel, masses, dt):
    n = masses.shape[0]
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pos[j] - pos[i]
            dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
            dist = np.sqrt(dist_sq)
            inv_dist3 = 1.0 / (dist_sq * dist)
            force = G * masses[i] * masses[j] * inv_dist3 * diff
            acc[i] += force / masses[i]
            acc[j] -= force / masses[j]
    vel += acc * dt
    pos += vel * dt
    return pos, vel

def simulate(days, dt_minutes=60):
    dt = dt_minutes * 60.0
    steps = int(days * DAY / dt)
    pos = positions.copy()
    vel = velocities.copy()
    earth_idx = 3
    out = np.empty((steps, len(masses), 3), dtype=np.float64)
    for i in range(steps):
        pos, vel = step(pos, vel, masses, dt)
        out[i] = pos - pos[earth_idx]
    return out

if __name__ == "__main__":
    data = simulate(365, dt_minutes=60)
    print("Simulated steps:", len(data))
    print("Final positions relative to Earth (m):")
    print(data[-1])
