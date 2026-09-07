def plan_loop(loaded_model, guidance_scale, noise_scheduler, uncond_block_cnn, test_block_cnn, generated_grasp_cnn):

    with torch.no_grad():
        for t in noise_scheduler.timesteps:
        
            cond = torch.cat([uncond_block_cnn, test_block_cnn], dim=0)
            grasp = torch.cat([generated_grasp_cnn] * 2, dim=0)
            time = torch.cat([torch.tensor([t])] * 2, dim=0)
        
            out = loaded_model(grasp, cond, time)
            out_uncond, out_cond = torch.chunk(out, 2, dim=0)
        
            guided = out_uncond + guidance_scale * (out_cond - out_uncond)
            generated_grasp_cnn = noise_scheduler.step(guided, t, generated_grasp_cnn).prev_sample

        final_grasp = generated_grasp_cnn
        #print(test_cands[...,0])
        print(final_grasp[0][:,0])

    return final_grasp


planner = RRTStar(env); planner.max_iterations = 2000; planner.step_size = 0.15
c = final_grasp[0][:,0][3:]
grasp_quat, approach_pos, grasp_pos, lift_pos, score = c[:4], c[4:7], c[7:10], c[10:13], c[13]

approach = planner.plan_to_pose(approach_pos, grasp_quat)
grasp = planner.plan_to_pose(grasp_pos, grasp_quat)
lift = planner.plan_to_pose(lift_pos, grasp_quat)
print("l:",lift,"a:", approach,"g:", grasp)
cnt = 0
while (approach == None or grasp == None or lift == None) and (cnt<5):
    reset_robot(env)
    final_grasp = plan_loop(loaded_model, guidance_scale, noise_scheduler, uncond_block_cnn, test_block_cnn, generated_grasp_cnn)
    c = final_grasp[0][:,0][3:]
    grasp_quat, approach_pos, grasp_pos, lift_pos, score = c[:4], c[4:7], c[7:10], c[10:13], c[13]
    approach = planner.plan_to_pose(approach_pos, grasp_quat)
    grasp = planner.plan_to_pose(grasp_pos, grasp_quat)
    lift = planner.plan_to_pose(lift_pos, grasp_quat)
    cnt += 1


reset_robot(env)
env.execute_path(approach, planner)
env.wait_idle()
env.add_collision_exception("block_a")
show_views(env)

env.execute_path(grasp, planner, step_size=0.005)
env.wait_idle()
show_views(env)

env.controller.close_gripper()
for _ in range(600):
    env.controller.step(); env.step()

env.attach_object_to_ee("block_a")
env.execute_path(lift, planner)
env.remove_collision_exception("block_a")
env.wait_idle()
print("success!")

after_img = render_view(env, azimuth=135, elevation=-25)
second_img = render_view(env, azimuth=105, elevation=-10)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(second_img); axes[0].set_title("top down");  axes[0].axis("off")
axes[1].imshow(after_img);  axes[1].set_title("after");  axes[1].axis("off")
plt.suptitle(f"Manual pick — front approach", fontsize=11)
plt.tight_layout()
show_fig(fig)
