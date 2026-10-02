from env.builder import make_builder
import numpy as np
import copy
from tampanda import GraspPlanner
from tampanda.symbolic.domains.blocks.blocks_domain import BlocksDomain

def collision_check(x, px, y, py, w, pw, d, pd, clearance):
    return (abs(x - px) < (w + pw) / 2 + clearance and abs(y - py) < (d + pd) / 2 + clearance)
def collect_data(n=1000, seed=42, obstacles=False, shape_index=None, verbose=False):

    if n < 1 or shape_index not in (None, 0, 1):
        raise ValueError("n must be positive; shape_index is None, 0 (cube), or 1 (block)")
    rng = np.random.default_rng(seed)
    env = None

    obj_list = []
    for i in range(n):
        if env is not None:
            close_env(env)
     
        # 1,2: default block with size 4, and 6. 3-6: blocks from ycb for extended advanced test.
        """
        Name ID weights dims(mm)
        cube: 40
        block: 60
        gelatin: 9 Gelatin Box 97g 28 x 85 x 73
        pudding: 8 Pudding Box 187g 35 x 110 x 89
        wood: 36 Wood Block 729g 85 x 85 x 200
        bricks: 60 Foam Brick 28g 50 x 75 x 50
        69 Colored Wood Blocks 10.8g 26
        """

        builder = make_builder()

        # sampling
        margin = 0.05
        
        #num_obstacles = np.randint(2,3)
        #for n in range(num_obstacles):
        obs_1_x = rng.uniform(0.15 + margin, 0.55 - margin)
        obs_1_y = rng.uniform(0.20 + margin, 0.60 - margin)
        obs_pose_1 = [obs_1_x, obs_1_y, 0.27]
        obs_idx_1 = rng.choice(np.arange(6), p=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        
        obs_2_x = rng.uniform(0.15 + margin, 0.55 - margin)
        obs_2_y = rng.uniform(0.20 + margin, 0.60 - margin)
        obs_pose_2 = [obs_2_x, obs_2_y, 0.27]
        obs_idx_2 = rng.choice(np.arange(6), p=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0])

        obs_list = [obs_pose_1, obs_pose_2]
        obs_idx = [obs_idx_1, obs_idx_2]
        if obstacles:
            builder.add_object("cube" if obs_idx_1 == 0 else "block", pos=obs_pose_1, name="obs_a")

        x = rng.uniform(0.15 + margin, 0.55 - margin)
        y = rng.uniform(0.20 + margin, 0.60 - margin)
        pose = [x, y, 0.27]
        
        idx = rng.choice(np.arange(6), p=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]) if shape_index is None else shape_index # original fixed-block mixture
        if idx == 0:
            builder.add_object("cube", pos=pose, name="block_a")
            shape = [0.04, 0.04, 0.04]
        elif idx == 1:
            builder.add_object("block", pos=pose, name="block_a")
            shape = [0.06, 0.06, 0.06]
        elif idx == 2:
            builder.add_object("pudding_box", pos=pose, name="block_a")
            shape = [0, 0, 0]
        elif idx == 3:
            builder.add_object("gelatin_box", pos=pose, name="block_a")
            shape = [0, 0, 0]
        elif idx == 4:
            builder.add_object("brick", pos=pose, name="block_a")
            shape = [0, 0, 0]
        elif idx == 5:
            builder.add_object("wood_block", pos=pose, name="block_a")
            shape = [0, 0, 0]
        
        env = builder.build_env(rate=200.0)
        
        if verbose:
            print(env.get_object_pose("block_a"))
        
        domain  = BlocksDomain(env.model, table_geom_name="table_surface")
        BOUNDS  = domain.get_working_bounds()
        #print(BOUNDS)

        BLOCK_HALF = env.get_object_half_size("block_a")
        shape = (2 * np.asarray(BLOCK_HALF)).tolist()
        # Correct initial heights regardless of whether XY resampling is needed.
        pose[2] = BOUNDS["table_height"] + BLOCK_HALF[2] + .003
        env.set_object_pose("block_a", np.asarray(pose))
        if obstacles:
            obs_half = env.get_object_half_size("obs_a")
            obs_pose_1[2] = BOUNDS["table_height"] + obs_half[2] + .003
            env.set_object_pose("obs_a", np.asarray(obs_pose_1))
        
        # check collision obs_1 and block
        if obs_idx_1 == 0:
            obs_width_1 = 0.04
            obs_depth_1 = 0.04
        else:
            obs_width_1 = 0.06
            obs_depth_1 = 0.06
        #if obs_idx_2 == 0:
        #    obs_width_2 = 0.04
        #    obs_depth_2 = 0.04
        #else:
        #    obs_width_2 = 0.06
        #    obs_depth_2 = 0.06
        clearance = 0.01
        if obstacles:
            obs_width_1, obs_depth_1 = 2 * obs_half[0], 2 * obs_half[1]
        collision = obstacles and collision_check(pose[0],obs_pose_1[0],pose[1],obs_pose_1[1],shape[0],obs_width_1,shape[1],obs_depth_1,clearance)
        attempts = 0
        while collision:
            if attempts >= 200:
                close_env(env)
                raise RuntimeError("Could not sample a collision-free block position")
            set_pose = sampling(BOUNDS["min_x"],BOUNDS["max_x"],BOUNDS["min_y"],BOUNDS["max_y"], margin, rng)
            pose[:2] = set_pose
            env.set_object_pose("block_a", np.array([set_pose[0], set_pose[1], BOUNDS["table_height"] + BLOCK_HALF[2] + 0.003]))
            collision = collision_check(pose[0],obs_pose_1[0],pose[1],obs_pose_1[1],shape[0],obs_width_1,shape[1],obs_depth_1,clearance)
            attempts += 1
        env.reset_velocities(); env.forward(); env.rest(0.4)

        if verbose:
            print(env.get_object_pose("block_a")[0])
        #print("world:",builder._resolve_world_poses())

        if verbose:
            print(env.get_object_pose("block_a"))

        grasp_planner = GraspPlanner(table_z=BOUNDS["table_height"])
        pos  = env.get_object_pose("block_a")[0]
        half = env.get_object_half_size("block_a")
        quat = env.get_object_pose("block_a")[1]
        candidates = grasp_planner.generate_candidates(pos, half, quat)

        obj_list.append([np.asarray(pos).copy(), np.asarray(half).copy(), np.asarray(quat).copy(), copy.deepcopy(candidates), shape])
        if (i + 1) % 100 == 0:
            print(f"Collected {i + 1}/{n} scenes", flush=True)

    return obj_list, builder, env, obs_list[:1] if obstacles else [], obs_idx[:1] if obstacles else []


def sampling(min_x, max_x, min_y, max_y, margin=.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.uniform([min_x + margin, min_y + margin], [max_x - margin, max_y - margin])


def close_env(env):
    close = getattr(env, "close", None)
    if callable(close):
        close()


def main():
    import argparse
    from data.storage import save_data
    parser = argparse.ArgumentParser(description="Collect demonstrations with the original scene sampler")
    parser.add_argument("--scenes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obstacles", action="store_true")
    parser.add_argument("--output", default="datasets/grasps.npz")
    args = parser.parse_args()
    records, _, env, _, _ = collect_data(args.scenes, args.seed, args.obstacles)
    try:
        count = save_data(args.output, records, vars(args))
        print(f"Saved {count} grasp candidates to {args.output}")
    finally:
        close_env(env)

if __name__ == "__main__":
    main()
