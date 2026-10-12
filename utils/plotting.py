import io
import mujoco
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
    import os
    from pathlib import Path
    from time import time_ns
    if os.environ.get("GRASP_FIGURE_DIR"):
        directory = Path(os.environ["GRASP_FIGURE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"grasp_{time_ns()}.png", bbox_inches="tight", dpi=120)
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


def plot_results(summary_path, output_dir):
    import json
    from pathlib import Path
    results = json.loads(Path(summary_path).read_text())["results"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for metric, title in (("grasp_success_rate", "Isolated pick success / generated samples"),
                          ("pick_place_success_rate", "Pick-and-place success / tasks"),
                          ("ik_rejection_fraction", "IK rejection / IK-tested samples"),
                          ("pose_coverage", "Coverage of IK-valid reference poses"),
                          ("sampling_ms_per_candidate", "Sampling time (ms / candidate)")):
        rows = [row for row in results if row.get(metric) is not None]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([row["method"] for row in rows], [row[metric] for row in rows])
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=25)
        if not metric.startswith("sampling"):
            ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/comparison/summary.json")
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()
    plot_results(args.summary, args.output)
