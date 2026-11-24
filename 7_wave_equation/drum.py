"""
A vibrating drum skin, simulated from its equation.

    python drum.py

The damped wave equation says how a stretched skin moves:

    acceleration  =  c^2 * (how curved the skin is here)  -  damping * velocity

    u_tt = c^2 * (u_xx + u_yy) - gamma * u_t

u is the height of the skin at each point. A point that sits lower than its
neighbours gets pulled up, and the other way round. Damping is air and
friction slowly taking the energy out.

The computer cannot handle a smooth skin, so the skin becomes a grid of
points and time becomes small steps. Each slope in the equation is replaced
by a difference between neighbouring grid points. That is the finite
difference method.

It saves:
    drum.gif      the skin moving, in 3D
    result.png    snapshots, plus two checks against known physics
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

here = Path(__file__).parent

# ----- the drum -------------------------------------------------------------
RADIUS = 1.0  # metres
WAVE_SPEED = 1.0  # c: how fast a ripple travels across the skin
DAMPING = 0.4  # gamma: how quickly the motion dies away
N = 121  # grid points across
SECONDS = 12.0

x = np.linspace(-RADIUS, RADIUS, N)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, x)
inside = X**2 + Y**2 < RADIUS**2  # the skin is a circle; the rim is clamped at height 0

# The time step has a hard upper limit. Take bigger steps than a ripple can
# cross one grid square and the simulation blows up (the CFL condition).
dt = 0.5 * dx / WAVE_SPEED  # the limit in 2D is dx / (c * sqrt(2)); this stays well under
steps = int(SECONDS / dt)


def curvature(u):
    """How much lower each point is than the average of its four neighbours (the Laplacian)."""
    out = np.zeros_like(u)
    out[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / dx**2
    return out


def energy(u, u_before):
    """Energy of motion plus energy stored in the stretch."""
    velocity = (u - u_before) / dt
    # Slopes now, times slopes one step ago: this form is the one the method conserves exactly.
    slope_x = (u[:, 1:] - u[:, :-1]) * (u_before[:, 1:] - u_before[:, :-1]) / dx**2
    slope_y = (u[1:, :] - u[:-1, :]) * (u_before[1:, :] - u_before[:-1, :]) / dx**2
    return 0.5 * dx**2 * (np.sum(velocity**2) + WAVE_SPEED**2 * (np.sum(slope_x) + np.sum(slope_y)))


def simulate(strike_at, damping, record_every=None, steps=steps):
    """Hit the drum at `strike_at` and step forward in time."""
    # The hit: a smooth dent, released from rest.
    u = np.exp(-((X - strike_at[0]) ** 2 + (Y - strike_at[1]) ** 2) / 0.02) * inside
    u_before = u.copy()
    a, b = 1 + damping * dt / 2, 1 - damping * dt / 2
    frames, energies, centre = [], [], []
    for step in range(steps):
        # The whole simulation is this one line: the wave equation, rearranged for "the next moment".
        u_next = (2 * u - b * u_before + (WAVE_SPEED * dt) ** 2 * curvature(u)) / a
        u_next *= inside  # the rim stays clamped
        u_before, u = u, u_next
        energies.append(energy(u, u_before))
        centre.append(u[N // 2, N // 2])
        if record_every and step % record_every == 0:
            frames.append(u.copy())
    return np.array(frames), np.array(energies), np.array(centre)


times = np.arange(steps) * dt
print(f"Grid: {N} x {N} points, {inside.sum():,} on the skin. {steps:,} time steps of {dt * 1000:.1f} ms.\n")

# ---------------------------------------------------------------------------
# Run 1: hit it off-centre, with damping. This is the one that gets animated.
# ---------------------------------------------------------------------------
frames, damped_energy, _ = simulate(strike_at=(0.35, 0.2), damping=DAMPING, record_every=12)

# ---------------------------------------------------------------------------
# Check 1: with damping switched off, energy must not change.
# ---------------------------------------------------------------------------
_, free_energy, centre = simulate(strike_at=(0.0, 0.0), damping=0.0)
drift = np.abs(free_energy / free_energy[0] - 1).max()
print("Check 1: with no damping, energy should stay constant")
print(f"  largest change over {steps:,} steps: {drift:.1e} of the starting energy")

# With damping on, theory says energy falls as exp(-gamma * t).
measured_rate = -np.polyfit(times, np.log(damped_energy), 1)[0]
print(f"  with damping on, energy should fall at rate {DAMPING}: measured {measured_rate:.3f}\n")

# ---------------------------------------------------------------------------
# Check 2: does it ring at the right notes?
# A round drum hit dead centre rings at frequencies set by the zeros of a
# Bessel function: f = zero * c / (2 * pi * R). Those numbers come from
# pen-and-paper maths, not from this code.
# ---------------------------------------------------------------------------
BESSEL_ZEROS = [2.404826, 5.520078, 8.653728]
theory = np.array(BESSEL_ZEROS) * WAVE_SPEED / (2 * np.pi * RADIUS)
# A longer run gives a sharper reading of the notes.
_, _, long_centre = simulate(strike_at=(0.0, 0.0), damping=0.0, steps=int(120.0 / dt))
spectrum = np.abs(np.fft.rfft(long_centre * np.hanning(len(long_centre))))
frequencies = np.fft.rfftfreq(len(long_centre), dt)
print("Check 2: the notes it rings at, against the pen-and-paper answer")
measured = []
for f in theory:
    near = (frequencies > f - 0.08) & (frequencies < f + 0.08)
    i = np.flatnonzero(near)[np.argmax(spectrum[near])]
    # Fine-tune the peak position between FFT bins (parabola through the three points around it).
    l, m, r = np.log(spectrum[i - 1 : i + 2])
    peak = frequencies[i] + 0.5 * (l - r) / (l - 2 * m + r) * (frequencies[1] - frequencies[0])
    measured.append(peak)
    print(f"  theory {f:.4f} Hz   simulation {peak:.4f} Hz   off by {abs(peak - f) / f:.2%}")

# ---------------------------------------------------------------------------
# Pictures.
# ---------------------------------------------------------------------------
shown = np.where(inside, 1.0, np.nan)  # hide the corners outside the circle
skip = 2  # draw every 2nd grid line; the surface looks the same and renders faster

fig = plt.figure(figsize=(17, 9))
for i, moment in enumerate([0, len(frames) // 12, len(frames) // 4, len(frames) - 1]):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.plot_surface(X[::skip, ::skip], Y[::skip, ::skip], (frames[moment] * shown)[::skip, ::skip], cmap="coolwarm", vmin=-0.4, vmax=0.4, linewidth=0)
    ax.set_zlim(-1, 1)
    ax.set_title(f"t = {moment * 12 * dt:.1f} s")
    ax.set_axis_off()

ax = fig.add_subplot(2, 2, 3)
ax.semilogy(times, damped_energy / damped_energy[0], color="#c0392b", lw=2, label="simulation, with damping")
ax.semilogy(times, np.exp(-DAMPING * times), "k:", label="theory: exp(-damping x time)")
ax.semilogy(times, free_energy / free_energy[0], color="#2a6fdb", lw=2, label="simulation, no damping")
ax.set_xlabel("seconds")
ax.set_ylabel("energy left (share of start)")
ax.set_title("Check 1: energy")
ax.legend()

ax = fig.add_subplot(2, 2, 4)
ax.plot(frequencies, spectrum / spectrum.max(), color="#2a6fdb", lw=1.5, label="simulation")
for i, f in enumerate(theory):
    ax.axvline(f, color="#c0392b", ls="--", lw=1, label="theory (Bessel zeros)" if i == 0 else None)
ax.set_xlim(0, 1.8)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("loudness")
ax.set_title("Check 2: the notes the drum rings at")
ax.legend()
fig.tight_layout()
fig.savefig(here / "result.png", dpi=100)
plt.close(fig)

fig = plt.figure(figsize=(6, 5.2))
ax = fig.add_subplot(projection="3d")


def draw(i):
    ax.clear()
    ax.plot_surface(X[::skip, ::skip], Y[::skip, ::skip], (frames[i] * shown)[::skip, ::skip], cmap="coolwarm", vmin=-0.4, vmax=0.4, linewidth=0)
    ax.set_zlim(-1, 1)
    ax.set_axis_off()
    ax.view_init(elev=32, azim=-60 + i * 0.25)
    ax.set_title(f"Drum skin, {i * 12 * dt:.1f} s after the hit")


fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)
FuncAnimation(fig, draw, frames=len(frames)).save(here / "drum.gif", writer=PillowWriter(fps=20), dpi=70)
print("\nSaved: result.png, drum.gif")
