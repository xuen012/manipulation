def collect_data(n=5):

    obj_list = []
    for i in range(n):
        
        builder = ArmSceneBuilder()
        builder.add_resource("table",  TABLE_SYMBOLIC_TEMPLATE)
        builder.add_resource("cube", BLOCK_SMALL_TEMPLATE)
        builder.add_resource("block", BLOCK_MEDIUM_TEMPLATE)
        builder.add_resource("pudding_box", {"type": "ycb", "name": "pudding_box"})
        builder.add_resource("gelatin_box", {"type": "ycb", "name": "gelatin_box"})
        builder.add_resource("brick", {"type": "ycb", "name": "foam_brick"})
        builder.add_resource("wood_block", {"type": "ycb", "name": "wood_block"})
        builder.add_object("table", name="table", pos=[0.75,  0.80, 0.00])

        # sampling
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.30, 0.65)
        y = rng.uniform(-0.20, 0.20)
        pose = [x, y, 0.27]

        idx = np.random.choice(np.arange(6), p=[0.4, 0.4, 0.05, 0.05, 0.08, 0.02]) # 0.28, 0.28, 0.15, 0.15, 0.02, 0.10, 0.02
        if idx == 0:
            builder.add_object("cube", pos=pose, name="block_a")
        elif idx == 1:
            builder.add_object("block", pos=pose, name="block_a")
        elif idx == 2:
            builder.add_object("pudding_box", pos=pose, name="block_a")
        elif idx == 3:
            builder.add_object("gelatin_box", pos=pose, name="block_a")
        elif idx == 4:
            builder.add_object("brick", pos=pose, name="block_a")
        elif idx == 5:
            builder.add_object("wood_block", pos=pose, name="block_a")

        env = builder.build_env(rate=200.0)
        domain  = BlocksDomain(env.model, table_geom_name="table_surface")
        BOUNDS  = domain.get_working_bounds()        
        BLOCK_HALF = env.get_object_half_size("block_a")
        env.set_object_pose("block_a", np.array([x, y, BOUNDS["table_height"] + BLOCK_HALF[2] + 0.003]))
        env.reset_velocities(); env.forward(); env.rest(0.4)

        grasp_planner = GraspPlanner(table_z=0.00)
        pos  = env.get_object_position("block_a")
        half = env.get_object_half_size("block_a")
        quat = env.get_object_orientation("block_a")
        candidates = grasp_planner.generate_candidates(pos, half, quat)

        obj_list.append([pos, half, quat, candidates])

        print(f"working area x ∈ [{BOUNDS['min_x']:.2f},{BOUNDS['max_x']:.2f}], "
        f"y ∈ [{BOUNDS['min_y']:.2f},{BOUNDS['max_y']:.2f}], table_z={BOUNDS['table_height']:.3f}")

        show_views(env)
    return obj_list

if __name__ == "__main__":
    print(collect_data())
