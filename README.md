![framework overview](/overview.png)

An overview of structure taken from Documentation.md  
/data Collect demonstrations and Encode.  
/env Build a basis of simulation environment.  
/model Train a denoiser.  
/pddl Initialize, sample, validate, integrate.  
/utils Setup and Deliver results.  
Documentation.md Introduce detailed structure of github repository.  
Test parameter.md Introduce detailed parameters used in the experiments.  
Problem_Learning.md Include problems met and designed version not included in the slide.  
general_framework.png Show an overview of the first version idea applied on pick and place task.  
overview.png Show the final method pipeline.

# manipulation

replace TAMP kinematics engineering pipeline with trained diffusion model, and use PDDL for verification.

Two main parts leads the designed pipeline for our framework under manipulation/pick and place problem.

In the original paper, object state classifier and robot configuration classifier exist because only point cloud latent space is known and the real status is unknown. A simplified version use the simulation data sampled from mujoco environment and default method, instead of real camera and point cloud latent space.
Diffusion sampler/DDPM obtains input of object state and trained on the derived grasp candidates. Thus we do not need them and use diffusion only. The sampled candidates are then sent to PDDL to check condition satisfied, if not, the replan starts in the sampler for another possible solution. The success rate is an evaluated standard for a successful sampler. Finally, the verified solution is executed and evaluated with samples diversity in the evaluation phase.

Problem Solve/Verificater: PDDL conditions include geometry check - eval_ik, eval_place_free, eval_accessible, corridor_blocked, and motion check - plan_pose, execute_path, open/close_gripper, detach/attach_object,  collision_exception.

Sampler: The design of sampler compares sampling method of guided DDPM, DDIM, and rejection sampling. Parameters, sufficient output information for success rate and learning steps are main concerns. IK solver is included in this section, given the solution from diffusion and give out the robot configuration.

When designing the model, three main factors are considered. 1. dataset e.g., whether to use shape or more information during model training. 2. the output e.g., whether a reduced information of the solution affecting the IK rejection rate or not. 3. A different training model such as DDPM or DDIM differs the sampling speed and the quality of samples diversity.

The tasks are finished step by step with one block type only, several types of blocks, with obstacles on the table, training on corresponding training process.
