from tampanda.planners.grasp_planner import GraspType
from tampanda import GraspPlanner

import numpy as np
import matplotlib.pyplot as plt
from tampanda import RRTStar
from pddl.reset import reset_robot
from pddl.checks import table_height, validate_candidate, detach_object, outcome
from utils.plotting import render_view, show_views, show_fig
# reset and get block info to search candidate
def run_baseline(
    env,
    planner=None,
    render=False,
    max_iterations=2000,
):
    planner = RRTStar(env) if planner is None else planner
    planner.max_iterations = max_iterations
    planner.step_size = 0.15

    # reset and get block info to search candidate
    reset_robot(env)

    can_pos = env.get_object_position("block_a")
    can_half = env.get_object_half_size("block_a")
    can_quat = env.get_object_orientation("block_a")

    # search grasp candidate
    grasp_planner = GraspPlanner(
        table_z=table_height(env)
    )

    candidates = grasp_planner.generate_candidates(
        can_pos,
        can_half,
        can_quat,
    )

    if render:
        print(candidates)

    candidate = candidates[0]

    before_img = (
        render_view(
            env,
            azimuth=135,
            elevation=-25,
        )
        if render
        else None
    )

    # move horizontally
    path = planner.plan_to_pose(
        candidate.approach_pos,
        candidate.grasp_quat,
        dt=0.005,
        max_iterations=max_iterations,
    )

    env.execute_path(
        path,
        planner,
        step_size=0.01,
    )

    env.wait_idle()

    if render:
        show_views(env)

    # move vertically
    env.add_collision_exception("block_a")

    path = planner.plan_to_pose(
        candidate.grasp_pos,
        candidate.grasp_quat,
        dt=0.005,
        max_iterations=max_iterations,
    )

    env.execute_path(
        path,
        planner,
        step_size=0.003,
    )

    env.wait_idle()

    # move end effector
    env.controller.close_gripper()

    for _ in range(600):
        env.controller.step()
        env.step()

    # attach and lift
    env.attach_object_to_ee("block_a")

    path = planner.plan_to_pose(
        candidate.lift_pos,
        candidate.grasp_quat,
        dt=0.005,
        max_iterations=max_iterations,
    )

    env.remove_collision_exception("block_a")

    env.execute_path(
        path,
        planner,
        step_size=0.003,
    )

    env.wait_idle()

    after_img = (
        render_view(
            env,
            azimuth=135,
            elevation=-25,
        )
        if render
        else None
    )

    # visualization
    if render:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 5),
        )

        axes[0].imshow(before_img)
        axes[0].set_title("before")
        axes[0].axis("off")

        axes[1].imshow(after_img)
        axes[1].set_title("after")
        axes[1].axis("off")

        plt.suptitle(
            f"Manual pick — {candidate.grasp_type.value}",
            fontsize=11,
        )

        plt.tight_layout()
        show_fig(fig)
