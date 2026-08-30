import io
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image

def render_view(env, lookat=(0.45, 0.0, 0.30), distance=1.4,
                azimuth=135.0, elevation=-25.0, width=640, height=480):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    r = mujoco.Renderer(env.model, height=height, width=width)
    r.update_scene(env.data, camera=cam)
    img = r.render()
    r.close()
    return img

def show_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    display(Image(buf.read()))
    plt.close(fig)

def show_views(env, title="", lookat=(0.45, 0.0, 0.30)):
    front = render_view(env, lookat=lookat, azimuth=135, elevation=-25)
    top   = render_view(env, lookat=lookat, azimuth=0,   elevation=-70)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(front); axes[0].set_title("front"); axes[0].axis("off")
    axes[1].imshow(top);   axes[1].set_title("top");   axes[1].axis("off")
    if title:
        plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    show_fig(fig)
