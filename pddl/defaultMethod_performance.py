from tampanda.planners.grasp_planner import GraspType
from tampanda import GraspPlanner

reset_robot(env)
can_pos  = env.get_object_position("block_a")
can_half = env.get_object_half_size("block_a")
can_quat = env.get_object_orientation("block_a")

grasp_planner = GraspPlanner(table_z=0.27)
candidates = grasp_planner.generate_candidates(can_pos, can_half, can_quat)
print(candidates)
candidate = candidates[0]#next(c for c in candidates if c.grasp_type == GraspType.FRONT)
before_img = render_view(env, azimuth=135, elevation=-25)

# move horizontally
path = planner.plan_to_pose(candidate.approach_pos, candidate.grasp_quat, dt=0.005, max_iterations=2000)
env.execute_path(path, planner, step_size=0.01)
env.wait_idle()
show_views(env)

# move vertically
env.add_collision_exception("block_a")
path = planner.plan_to_pose(candidate.grasp_pos, candidate.grasp_quat, dt=0.005, max_iterations=2000)
env.execute_path(path, planner, step_size=0.003)
env.wait_idle()


# move end effector
env.controller.close_gripper()
for _ in range(600):
    env.controller.step(); env.step()

# attach and lift
env.attach_object_to_ee("block_a")
path = planner.plan_to_pose(candidate.lift_pos, candidate.grasp_quat, dt=0.005, max_iterations=2000)
env.remove_collision_exception("block_a")
env.execute_path(path, planner, step_size=0.003)
env.wait_idle()
after_img = render_view(env, azimuth=135, elevation=-25)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(before_img); axes[0].set_title("before"); axes[0].axis("off")
axes[1].imshow(after_img);  axes[1].set_title("after");  axes[1].axis("off")
plt.suptitle(f"Manual pick — front approach", fontsize=11)
plt.tight_layout()
show_fig(fig)
