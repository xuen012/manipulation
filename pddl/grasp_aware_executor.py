
from __future__ import annotations

import numpy as np
from tampanda.planners import pick_place as native_pick_place


def _vector(value, length, name):
    result = np.asarray(value, dtype=float).copy()
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _rotation_wxyz(quaternion):
    """Rotation from EE-local coordinates to world coordinates (MuJoCo order)."""
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


class GraspAwarePickPlaceExecutor(native_pick_place.PickPlaceExecutor):
    """Native executor with placement adjusted for the actual held-object pose.

    Constructor and pick() are inherited unchanged. After place(),
    last_place_result records the requested goal, target adjustment, native
    return value, and measured final error. A True return requires both native
    completion and the original benchmark's 30 mm XY / 25 mm Z pose check.

    As in the previous integration checks, allow 200 simulation steps after
    native placement for the object to settle. The caller need not add them.
    """

    def place(
        self,
        block_name: str,
        place_block_center: np.ndarray,
        ee_quat: np.ndarray | None = None,
        target_block_name: str | None = None,
        approach_height: float = 0.15,
        place_clearance: float = 0.003,
    ) -> bool:
        self.last_place_result = {"native_place_return": None,
                                  "verified_place_success": False}
        goal = _vector(place_block_center, 3, "place_block_center")
        if ee_quat is None:
            ee_quat = self._last_grasp_quat
            if ee_quat is None:
                raise ValueError("ee_quat is required if no prior pick has completed")
        quaternion = _vector(ee_quat, 4, "ee_quat")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12 or not np.isfinite(norm):
            raise ValueError("ee_quat must have a finite, nonzero norm")
        quaternion /= norm
        if self._ee_site_id < 0:
            raise RuntimeError("The attachment_site end-effector site was not found")

        object_pos = _vector(self.env.get_object_pose(block_name)[0], 3, "object position")
        ee_pos = _vector(self.env.data.site_xpos[self._ee_site_id], 3, "EE position")
        current_rotation = np.asarray(
            self.env.data.site_xmat[self._ee_site_id], dtype=float
        ).reshape(3, 3).copy()
        if not np.isfinite(current_rotation).all():
            raise ValueError("The current EE rotation must be finite")

        object_in_ee = current_rotation.T @ (object_pos - ee_pos)
        offset_at_place = _rotation_wxyz(quaternion) @ object_in_ee
        desired_ee = goal - offset_at_place

        # The inherited place() adds GRASP_CONTACT_OFFSET + place_clearance to
        # its argument's Z coordinate. Cancel only its fixed contact offset;
        # leave clearance to the native implementation so it is added once.
        contact_offset = float(native_pick_place.GRASP_CONTACT_OFFSET)
        if not np.isfinite(contact_offset):
            raise ValueError("Native GRASP_CONTACT_OFFSET must be finite")
        native_goal = desired_ee.copy()
        native_goal[2] -= contact_offset
        desired_ee_with_clearance = desired_ee.copy()
        desired_ee_with_clearance[2] += place_clearance
        self.last_place_result.update({
            "requested_object_goal": goal.tolist(),
            "measured_object_pos": object_pos.tolist(),
            "measured_ee_pos": ee_pos.tolist(),
            "measured_object_in_ee_frame": object_in_ee.tolist(),
            "object_offset_world_at_place_orientation": offset_at_place.tolist(),
            "placement_ee_quat_wxyz": quaternion.tolist(),
            "desired_ee_pos_with_clearance": desired_ee_with_clearance.tolist(),
            "adjusted_argument_to_native_place": native_goal.tolist(),
            "native_contact_offset_m": contact_offset,
            "place_clearance_m": float(place_clearance),
            "attachment_assisted": bool(self.use_attachment),
            "horizontal_tolerance_m": 0.03, "vertical_tolerance_m": 0.025,
            "post_place_settle_steps": 200,
        })

        native_ok = bool(super().place(
            block_name, native_goal, ee_quat=quaternion,
            target_block_name=target_block_name, approach_height=approach_height,
            place_clearance=place_clearance,
        ))
        self.last_place_result["native_place_return"] = native_ok
        for _ in range(200):
            self.env.step()
        final = _vector(self.env.get_object_pose(block_name)[0], 3, "final object position")
        xy_error = float(np.linalg.norm(final[:2] - goal[:2]))
        z_error = float(abs(final[2] - goal[2]))
        measured_ok = xy_error <= 0.03 and z_error <= 0.025
        verified_ok = bool(native_ok and measured_ok)
        self.last_place_result.update({
            "final_object_pos": final.tolist(), "horizontal_error_m": xy_error,
            "vertical_error_m": z_error, "measured_place_success": bool(measured_ok),
            "verified_place_success": verified_ok,
        })
        print(f"[GraspAwarePickPlaceExecutor] verified place: {verified_ok}; "
              f"XY error {1000 * xy_error:.2f} mm; Z error {1000 * z_error:.2f} mm",
              flush=True)
        return verified_ok
