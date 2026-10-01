from env.builder import make_builder
import numpy as np
from tampanda import GraspPlanner
from tampanda.symbolic.domains.blocks.blocks_domain import BlocksDomain

def collision_check(x, px, y, py, w, pw, d, pd, clearance):
    return (abs(x - px) < (w + pw) / 2 + clearance and abs(y - py) < (d + pd) / 2 + clearance)
def collect_data(n):

    obj_list = []
    for i in range(n):
     
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
        rng = np.random.default_rng()
        margin = 0.05
        
        #num_obstacles = np.randint(2,3)
        #for n in range(num_obstacles):
        obs_1_x = rng.uniform(0.15 + margin, 0.55 - margin)
        obs_1_y = rng.uniform(0.20 + margin, 0.60 - margin)
        obs_pose_1 = [obs_1_x, obs_1_y, 0.27]
        obs_idx_1 = np.random.choice(np.arange(6), p=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        
        obs_2_x = rng.uniform(0.15 + margin, 0.55 - margin)
        obs_2_y = rng.uniform(0.20 + margin, 0.60 - margin)
        obs_pose_2 = [obs_2_x, obs_2_y, 0.27]
        obs_idx_2 = np.random.choice(np.arange(6), p=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0])

        obs_list = [obs_pose_1, obs_pose_2]
        obs_idx = [obs_idx_1, obs_idx_2]
        builder.add_object("cube", pos=obs_pose_1, name="obs_a")

        x = rng.uniform(0.15 + margin, 0.55 - margin)
        y = rng.uniform(0.20 + margin, 0.60 - margin)
        pose = [x, y, 0.27]
        
        idx = np.random.choice(np.arange(6), p=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]) # 0.28, 0.28, 0.15, 0.15, 0.02, 0.10, 0.02
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
        
        print(env.get_object_pose("block_a"))
        
        domain  = BlocksDomain(env.model, table_geom_name="table_surface")
        BOUNDS  = domain.get_working_bounds()
        #print(BOUNDS)

        BLOCK_HALF = env.get_object_half_size("block_a")
        
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
        collision = collision_check(pose[0],obs_pose_1[0],pose[1],obs_pose_1[1],shape[0],obs_width_1,shape[1],obs_depth_1,clearance)
        print(collision)
        if collision:
            set_pose = sampling(BOUNDS["min_x"],BOUNDS["max_x"],BOUNDS["min_y"],BOUNDS["max_y"], margin)
            env.set_object_pose("block_a", np.array([set_pose[0], set_pose[1], BOUNDS["table_height"] + BLOCK_HALF[2] + 0.003]))
            env.reset_velocities(); env.forward(); env.rest(0.4)

        print(builder._get_object_pos("block_a"))
        #print("world:",builder._resolve_world_poses())

        print(env.get_object_pose("block_a"))

        grasp_planner = GraspPlanner(table_z=0.00)
        pos  = env.get_object_pose("block_a")[0]
        half = env.get_object_half_size("block_a")
        quat = env.get_object_pose("block_a")[1]
        candidates = grasp_planner.generate_candidates(pos, half, quat)

        obj_list.append([pos, half, quat, candidates, shape])

    return obj_list, builder, env, obs_list, obs_idx

def sampling(
    min_x,
    max_x,
    min_y,
    max_y,
    margin=.05,
):
    return np.random.uniform(
        [min_x + margin, min_y + margin],
        [max_x - margin, max_y - margin],
    )

if __name__ == "__main__":
    print(collect_data())
