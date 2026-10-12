"""Collision planning for a physically held object, without attaching it.

TAMPanda's kinematic attach operation also registers the object as carried
geometry for RRT collision queries. Physical execution needs that planning
registration too, while leaving step() governed entirely by MuJoCo physics.
"""
from functools import wraps

import mujoco


def track_physical_placement(place_function):
    """Scope carried-object collision registration around manual placement.

    The wrapped placement routine clears the registration after opening the
    gripper and before planning retreat. This wrapper also clears it on every
    return or exception, including failure to plan the first transport move.
    Attachment-assisted calls retain their original behavior.
    """

    @wraps(place_function)
    def wrapped(env, planner, candidate, initial_pos, place_pos, attachment=True,
                max_iterations=2000):
        if attachment:
            return place_function(env, planner, candidate, initial_pos, place_pos,
                                  attachment=attachment, max_iterations=max_iterations)

        register = getattr(env, "set_collision_held_body", None)
        clear = getattr(env, "clear_collision_held_body", None)
        if not callable(register) or not callable(clear):
            raise RuntimeError("Physical placement requires TAMPanda's held-body collision API")
        if getattr(env, "_attached", None) is not None:
            raise RuntimeError("Physical placement cannot run with a kinematic attachment active")
        if getattr(env, "_collision_held_body", None) is not None:
            raise RuntimeError("Physical placement started with an existing held-body registration")

        ee_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "block_a")
        if ee_id < 0 or body_id < 0:
            raise RuntimeError("The end-effector site or block_a body was not found")
        # Match TAMPanda's measured relative-pose convention. This updates only
        # derived kinematics; it neither moves nor attaches the physical cube.
        mujoco.mj_forward(env.model, env.data)
        ee_pos = env.data.site_xpos[ee_id].copy()
        ee_rotation = env.data.site_xmat[ee_id].reshape(3, 3).copy()
        object_pos = env.data.xpos[body_id].copy()
        object_rotation = env.data.xmat[body_id].reshape(3, 3).copy()
        relative_pos = ee_rotation.T @ (object_pos - ee_pos)
        relative_rotation = ee_rotation.T @ object_rotation
        try:
            # Native collision queries temporarily move this carried geometry
            # for each hypothetical arm configuration and restore the state.
            # The native physics step uses _attached, which remains None here.
            register("block_a", relative_pos, relative_rotation)
            return place_function(env, planner, candidate, initial_pos, place_pos,
                                  attachment=False, max_iterations=max_iterations)
        finally:
            clear()

    return wrapped
