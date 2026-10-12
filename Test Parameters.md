# Model and reproducible experiment setup
Setting | Recorded value/implementation
|---|---:|
Architecture | Conv1d condition 10 to 64; concatenate noisy grasp + condition + time (145 channels); Conv1d 145 to 64; SiLU; Unet design; 2-head attention; Conv1d 64 to 64; SiLU; Conv1d 64 to 17
Parameter count / representation | 31,953 parameters. Kernel size 1 throughout. Four padded positions, only column 0 active. No multiscale U-Net pyramid.
Training data | 2,000 scenes, 4,968 teacher labels: 2,000 top-X, 2,000 top-Y, 968 front
Split and scaling | Seed 42. 1,700 train / 300 validation scenes (4,222 / 746 labels). Fit normalization only on training rows.
Optimization | 200 epochs, batch 32, AdamW learning rate 1e-4, gradient norm clip 1.0
Diffusion objective | Gaussian corruption at a uniform random timestep among 1,000 levels. Predict clean normalized grasp with active-column MSE.
Guidance training / inference | Drop condition for 10% of examples. DDIM and DDPM share one checkpoint. Final comparison: 100 steps, guidance 1.
Training outcome | Best epoch 197, validation MSE 0.277842. Recorded loop: 393.63 s on RTX 3080, below the course 30-minute target.

The checkpoint stores weights, normalizer, scene split and prediction_type="sample". The original
architecture/encoding were retained to respect the preservation constraint and small-model budget. Their
superiority over other architectures or reduced object-relative outputs was not experimentally established.
Teacher labels are geometrically generated, not filtered for physical grasp success.
### Paired evaluation and honest denominators
Three seed groups (20000, 30000, 40000) each provide 2000 scenes with alternating cube size. Each isolated
candidate executes in a fresh copy of its scene. Full tasks also start fresh and share deterministic placement
goals at least 12 cm from the initial object position. Motion and sampler randomness use separate seeds.
Learned tasks allow up to five proposal attempts before physical execution, not five complete physical retries.

Metric | Definition
|---|---:|
Isolated grasp success | Measured lifts >=25 mm / all generated candidates, including geometry and IK rejects (480/method)
Full-task success | Measured pick AND final placement / all 60 tasks; XY <=30 mm and absolute Z <=25 mm
Coverage / IK rejection | Mean coverage of IK-valid teacher poses within 25 mm at all waypoints and 20 degrees; failed IK / candidates sent to IK
Sampling speed | CUDA-synchronized, batch-amortized ms/candidate. Excludes model loading, IK, planningand physics.
