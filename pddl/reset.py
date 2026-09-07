def sim_steps(env, n: int) -> None:
    """Run n simulation steps (no controller stepping)."""
    for _ in range(n):
        env.step()

def reset_robot(env) -> None:
    """Teleport arm to home, open gripper, force controller IDLE."""
    env.reset_arm_to_home()
    env.data.qpos[:8] = env.initial_qpos[:8]
    env.data.ctrl[:8] = env.initial_ctrl[:8]   # ctrl[7]=255 → gripper open
    env.data.qvel[:]  = 0.0
    # IMPORTANT: do NOT call open_gripper() here – it sets status=GRASPING
    # and settle() doesn't step the controller, leaving it permanently busy.
    env.controller.stop()                        # force IDLE, clear trajectory
    mujoco.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)
    sim_steps(env, 30)
