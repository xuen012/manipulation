# manipulation

replace TAMP kinematics engineering pipeline with trained diffusion model, and use PDDL for verification.

Two main parts leads the designed pipeline for our framework under manipulation/pick and place problem.

In the original paper, object state classifier and robot configuration classifier exist because only point cloud latent space is known and the real status is unknown. A simplified version use the simulation data sampled from mujoco environment and default method, instead of real camera and point cloud latent space.
Diffusion sampler/DDPM obtains input of object state and trained on the derived grasp candidates. Thus we do not need them and use diffusion only. The sampled candidates are then sent to PDDL to check condition satisfied together with success rate and samples diversity as for evaluation standard.

PDDL conditions include geometry check - eval_ik, eval_place_free, eval_accessible, corridor_blocked, and motion check - plan_pose, execute_path, open/close_gripper, detach/attach_object,  collision_exception. These verification are done together with ik motion planner.

The design of sampler compares sampling method of guided DDPM, DDIM, and rejection sampling. Parameters, sufficient output information for success rate and learning steps are main concerns.
