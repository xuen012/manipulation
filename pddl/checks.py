"""Missing feasibility and scene helpers used by the existing execution scripts."""

from contextlib import contextmanager
import inspect

import numpy as np

from tampanda.symbolic.domains.blocks.blocks_domain import BlocksDomain


def table_height(env):
    return float(
        BlocksDomain(
            env.model,
            table_geom_name="table_surface",
        ).get_working_bounds()["table_height"]
    )


@contextmanager
def saved_state(env):
    names = (
        "qpos",
        "qvel",
        "ctrl",
        "act",
        "qacc_warmstart",
        "mocap_pos",
        "mocap_quat",
    )

    values = {
        name: getattr(env.data, name).copy()
        for name in names
    }

    time = env.data.time

    try:
        yield
    finally:
        for name, value in values.items():
            getattr(env.data, name)[:] = value

        env.data.time = time
        env.forward()
        env.ik.update_configuration(env.data.qpos)


def ik_pose(env, position, quaternion):
    with saved_state(env):
        env.ik.update_configuration(env.data.qpos)

        env.ik.set_target_position(
            np.asarray(position),
            np.asarray(quaternion),
        )

        return bool(
            env.ik.converge_ik(env.rate.dt)
        )


def validate_candidate(
    env,
    candidate,
    object_pos=None,
    half_size=None,
):
    pos = np.asarray(
        env.get_object_pose("block_a")[0]
        if object_pos is None
        else object_pos
    )

    half = np.asarray(
        env.get_object_half_size("block_a")
        if half_size is None
        else half_size
    )

    points = np.asarray([
        candidate.approach_pos,
        candidate.grasp_pos,
        candidate.lift_pos,
    ])

    if (
        not np.isfinite(points).all()
        or np.any(points[:, 2] <= table_height(env))
    ):
        return False, "geometry"

    if np.linalg.norm(points[1] - pos) > np.linalg.norm(half) + 0.08:
        return False, "geometry"

    if (
        not 0.015
        <= np.linalg.norm(points[0] - points[1])
        <= 0.4
        or not 0.025
        <= points[2, 2] - points[1, 2]
        <= 0.4
    ):
        return False, "geometry"

    with saved_state(env):
        env.ik.update_configuration(env.data.qpos)

        for stage, point in zip(
            ("approach", "grasp", "lift"),
            points,
        ):
            env.ik.set_target_position(
                point,
                candidate.grasp_quat,
            )

            if not env.ik.converge_ik(env.rate.dt):
                return False, f"{stage}_ik"

            env.ik.update_configuration(
                env.ik.configuration.q.copy()
            )

    return True, None


def detach_object(env):
    """Support the two public detach spellings used by course revisions."""

    detach = (
        getattr(env, "detach_object_from_ee", None)
        or getattr(env, "detach_object", None)
    )

    if not callable(detach):
        raise RuntimeError(
            "This TAMPanda revision has no "
            "detach_object_from_ee/detach_object method"
        )

    if inspect.signature(detach).parameters:
        detach("block_a")
    else:
        detach()


def outcome(
    env,
    initial_pos,
    picked=False,
    placed=False,
    reason=None,
):
    final_pos = np.asarray(
        env.get_object_pose("block_a")[0]
    )

    return {
        "grasp_success": bool(picked),
        "place_success": bool(placed),
        "pick_place_success": bool(picked and placed),
        "failure_reason": reason,
        "initial_pos": np.asarray(initial_pos).tolist(),
        "final_pos": final_pos.tolist(),
    }