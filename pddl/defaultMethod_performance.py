from tampanda.planners.grasp_planner import GraspType
from tampanda import GraspPlanner

import numpy as np
import matplotlib.pyplot as plt
from tampanda import RRTStar
from pddl.reset import reset_robot
from pddl.checks import table_height, validate_candidate, detach_object, outcome
from pddl.physical_placement import track_physical_placement
from utils.plotting import render_view, show_views, show_fig


def run_baseline(env, planner=None, candidate=None, place_pos=None, render=False,
                 attachment=True, max_iterations=2000):
    """The original manual baseline, with missing inputs and failure checks added."""
    planner = RRTStar(env) if planner is None else planner
    planner.max_iterations = max_iterations
    planner.step_size = .15
    # reset and get block info to search candidate
    reset_robot(env)
    can_pos  = env.get_object_position("block_a")
    can_half = env.get_object_half_size("block_a")
    can_quat = env.get_object_orientation("block_a")

    # search grasp candidate
    grasp_planner = GraspPlanner(table_z=table_height(env))
    candidates = grasp_planner.generate_candidates(can_pos, can_half, can_quat)
    if render:
        print(candidates)
    if candidate is None:
        candidate = next((c for c in candidates if validate_candidate(env, c)[0]), None)
    initial_pos = np.asarray(can_pos).copy()
    if candidate is None:
        return outcome(env, initial_pos, reason="no_feasible_candidate")
    ok, reason = validate_candidate(env, candidate)
    if not ok:
        return outcome(env, initial_pos, reason=reason)
    before_img = render_view(env, azimuth=135, elevation=-25) if render else None

    # move horizontally
    path = planner.plan_to_pose(candidate.approach_pos, candidate.grasp_quat, dt=0.005, max_iterations=max_iterations)
    if path is None:
        return outcome(env, initial_pos, reason="approach_motion")
    env.execute_path(path, planner, step_size=0.01)
    env.wait_idle()
    if render:
        show_views(env)

    # move vertically
    env.add_collision_exception("block_a")
    path = planner.plan_to_pose(candidate.grasp_pos, candidate.grasp_quat, dt=0.005, max_iterations=max_iterations)
    if path is None:
        env.remove_collision_exception("block_a")
        return outcome(env, initial_pos, reason="grasp_motion")
    env.execute_path(path, planner, step_size=0.003)
    env.wait_idle()

    # move end effector
    env.controller.close_gripper()
    for _ in range(600):
        env.controller.step(); env.step()

    # attach and lift
    if attachment:
        env.attach_object_to_ee("block_a")
    path = planner.plan_to_pose(candidate.lift_pos, candidate.grasp_quat, dt=0.005, max_iterations=max_iterations)
    env.remove_collision_exception("block_a")
    if path is None:
        if attachment:
            detach_object(env)
        return outcome(env, initial_pos, reason="lift_motion")
    env.execute_path(path, planner, step_size=0.003)
    env.wait_idle()
    after_img = render_view(env, azimuth=135, elevation=-25) if render else None

    # visualization
    if render:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(before_img); axes[0].set_title("before"); axes[0].axis("off")
        axes[1].imshow(after_img);  axes[1].set_title("after");  axes[1].axis("off")
        plt.suptitle(f"Manual pick — {candidate.grasp_type.value}", fontsize=11)
        plt.tight_layout()
        show_fig(fig)

    picked = float(env.get_object_pose("block_a")[0][2]) - initial_pos[2] >= .025
    placed = place_object(env, planner, candidate, initial_pos, place_pos, attachment, max_iterations) if picked and place_pos is not None else False
    return outcome(env, initial_pos, picked, placed,
                   None if picked and (place_pos is None or placed) else "lift_or_place_execution")


@track_physical_placement
def place_object(env, planner, candidate, initial_pos, place_pos, attachment=True, max_iterations=2000):
    """Added final place stage after the original manual lift; no new executor."""
    place_pos = np.asarray(place_pos, dtype=float)
    grasp_offset = np.asarray(candidate.grasp_pos) - np.asarray(initial_pos)
    place_grasp = place_pos + grasp_offset
    above = place_grasp + np.array([0., 0., max(.12, candidate.lift_pos[2] - candidate.grasp_pos[2])])
    path = planner.plan_to_pose(above, candidate.grasp_quat, dt=env.rate.dt, max_iterations=max_iterations)
    if path is None:
        return False
    env.execute_path(path, planner, step_size=.01)
    env.wait_idle()
    env.add_collision_exception("block_a")
    try:
        path = planner.plan_to_pose(place_grasp, candidate.grasp_quat, dt=env.rate.dt, max_iterations=max_iterations)
        if path is None:
            return False
        env.execute_path(path, planner, step_size=.003)
        env.wait_idle()
        env.controller.open_gripper()
        for _ in range(600):
            env.controller.step(); env.step()
        if attachment:
            detach_object(env)
        else:
            # The released cube must stay at rest during retreat planning.
            env.clear_collision_held_body()
        path = planner.plan_to_pose(above, candidate.grasp_quat, dt=env.rate.dt, max_iterations=max_iterations)
        if path is None:
            return False
        env.execute_path(path, planner, step_size=.003)
        env.wait_idle()
    finally:
        env.remove_collision_exception("block_a")
    for _ in range(200):
        env.step()
    actual = np.asarray(env.get_object_pose("block_a")[0])
    return bool(np.linalg.norm(actual[:2] - place_pos[:2]) <= .03 and abs(actual[2] - place_pos[2]) <= .025)


if __name__ == "__main__":
    from pddl.benchmark import main
    main(default_methods=["baseline"])
