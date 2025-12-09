import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
from tqdm import tqdm  # type: ignore

def generate_cartpole_animation(trajectory_states, out_path):
    states = np.asarray(trajectory_states)
    if states.ndim != 2 or states.shape[0] == 0:
        raise RuntimeError('trajectory_states must be a non-empty 2D array.')
    if states.shape[1] < 3:
        raise RuntimeError('Need at least 3 columns: x, state[1], state[2].')

    # ---- Config -----------------------------------------------------------------
    L = 1.0                     # pole length
    cart_w, cart_h = 0.4, 0.2   # cart size
    bob_r = 0.05
    interval_ms = 20
    fps = 24
    dpi = 120

    # ---- Helpers ----------------------------------------------------------------
    def geodesic_unwrap(a):
        """
        Make angle sequence geodesically continuous:
        at each step, choose the representative whose increment lies in [-pi, pi].
        """
        a = np.asarray(a, dtype=float)
        out = np.empty_like(a)
        if len(a) == 0:
            return out
        # start in canonical principal value
        out[0] = (a[0] + np.pi) % (2*np.pi) - np.pi
        for i in range(1, len(a)):
            d = a[i] - a[i - 1]
            # map increment to [-pi, pi]
            d = (d + np.pi) % (2*np.pi) - np.pi
            out[i] = out[i - 1] + d
        return out

    # ---- Data slices -------------------------------------------------------------
    x = states[:, 0]
    theta_geo = geodesic_unwrap(states[:, 1])
    # Precompute trig and pole endpoints for all frames (avoids per-frame sin/cos)
    sin_theta = np.sin(theta_geo)
    cos_theta = np.cos(theta_geo)
    x_end_all = x + L * sin_theta
    y_end_all = L * cos_theta

    # Plot extents
    x_min, x_max = np.min(x) - 1.5, np.max(x) + 1.5
    y_min, y_max = -L - 0.5, L + 0.5

    # ---- Figure & artists --------------------------------------------------------
    fig, (ax) = plt.subplots(1, 1, figsize=(12, 5), sharey=True)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.6)

    ax.set_title('θ = state[1] (geodesic)')
    fig.suptitle('Cartpole animation with geodesic angle unwrapping')

    line, = ax.plot([], [], '-', lw=3)
    line.set_animated(True)
    line.set_antialiased(False)
    cart = Rectangle((x[0] - cart_w/2, -cart_h/2), cart_w, cart_h, color='gray', zorder=2)
    ax.add_patch(cart)
    bob  = Circle((x[0], L), bob_r, zorder=3)
    pivot = Circle((x[0], 0.0), 0.03, color='k', zorder=3)
    ax.add_patch(bob); ax.add_patch(pivot)
    # Mark patches as animated and turn off antialiasing for speed
    cart.set_animated(True); cart.set_antialiased(False)
    bob.set_animated(True); bob.set_antialiased(False)
    pivot.set_animated(True); pivot.set_antialiased(False)

    def init():
        # Set initial artist states
        line.set_data([], [])
        cart.set_xy((x[0] - cart_w/2, -cart_h/2))
        bob.center = (x_end_all[0], y_end_all[0])
        pivot.center = (x[0], 0.0)
        return (line, cart, bob, pivot)

    def update(i):
        xi = x[i]
        x_end = x_end_all[i]
        y_end = y_end_all[i]
        line.set_data([xi, x_end], [0.0, y_end])
        cart.set_xy((xi - cart_w/2, -cart_h/2))
        bob.center = (x_end, y_end)
        pivot.center = (xi, 0.0)
        return (line, cart, bob, pivot)

    # ---- Build animation & save to MP4 (no embedding) ---------------------------
    plt.rcParams['animation.html'] = 'none'  # avoid large inline embeds

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=interval_ms,
        blit=True,
        init_func=init,
        cache_frame_data=False,
    )

    writer = animation.FFMpegWriter(
        fps=fps,
        metadata=dict(artist="CartPole"),
        extra_args=[
            '-vcodec', 'libx264',
            '-preset', 'veryfast',  # faster encode; larger files
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
        ],
    )

    pbar = tqdm(total=len(states), desc='Saving animation', unit='frame')

    def _progress(_i, _n):
        pbar.update(1)

    try:
        ani.save(out_path, writer=writer, dpi=dpi, progress_callback=_progress)
    finally:
        pbar.close()
    plt.close(fig)
    print(f"Saved animation to {out_path}")