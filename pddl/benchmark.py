"""Measure the existing baseline and diffusion scripts on the same seeded scenes."""
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import argparse
import csv
import json
import random
import time
import numpy as np
import torch
from tampanda import GraspPlanner
from data.collect_data import collect_data, close_env, collision_check
from data.processing import candidate_from_vector
from model.sampler import GraspSampler
from pddl.checks import table_height, validate_candidate
from pddl.reset import reset_robot
from pddl.defaultMethod_performance import run_baseline
from pddl.evaluate_diffusion import run_diffusion


def seed_planning(seed):
    random.seed(seed)
    np.random.seed(seed)


def scene(seed, shape, obstacles):
    seed_planning(seed)
    _, _, env, _, _ = collect_data(1, seed=seed, obstacles=obstacles, shape_index=shape)
    reset_robot(env)
    return env


def object_info(env):
    pos, quat = env.get_object_pose("block_a")
    return np.asarray(pos).copy(), np.asarray(env.get_object_half_size("block_a")).copy(), np.asarray(quat).copy()


def reference_candidates(env):
    return GraspPlanner(table_z=table_height(env)).generate_candidates(*object_info(env))


def free_goal(env, seed, obstacles):
    pos, half, _ = object_info(env)
    rng = np.random.default_rng(seed + 100000)
    for _ in range(200):
        goal = np.r_[rng.uniform([.22, .27], [.48, .53]), table_height(env) + half[2] + .003]
        if np.linalg.norm(goal[:2] - pos[:2]) < .12:
            continue
        if obstacles:
            other = env.get_object_pose("obs_a")[0]
            other_half = env.get_object_half_size("obs_a")
            if collision_check(goal[0], other[0], goal[1], other[1], 2 * half[0], 2 * other_half[0], 2 * half[1], 2 * other_half[1], .02):
                continue
        return goal
    raise RuntimeError("Unable to sample a free placement goal")


def matches(a, b):
    error = max(np.linalg.norm(np.asarray(getattr(a, key)) - np.asarray(getattr(b, key)))
                for key in ("approach_pos", "grasp_pos", "lift_pos"))
    qa, qb = np.asarray(a.grasp_quat), np.asarray(b.grasp_quat)
    dot = abs(qa @ qb) / (np.linalg.norm(qa) * np.linalg.norm(qb))
    return error <= .025 and 2 * np.arccos(np.clip(dot, 0, 1)) <= np.deg2rad(20)


def fraction(a, b):
    return a / b if b else None


def write_results(args, summary, candidates, tasks):
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    definitions = {
        "grasp_success_rate": "successful isolated manual picks / every generated sample, including geometry and IK rejects; null unless --execute-samples",
        "ik_rejection_fraction": "failed IK / samples sent to IK; malformed geometry counted separately",
        "sampling_ms_per_candidate": "synchronized sampling time / emitted candidates; model load, IK and motion/execution excluded",
        "pose_coverage": "mean fraction of IK-valid teacher poses matched within 2.5cm at all three points and 20 degrees",
        "pick_place_success_rate": "measured pick AND placement successes / all task attempts",
        "attachment_assisted": "default uses the original kinematic attachment; this is a simulation execution proxy, not force closure"}
    (directory / "summary.json").write_text(json.dumps({"settings": vars(args), "definitions": definitions, "results": summary}, indent=2, allow_nan=False))
    for name, rows in (("summary.csv", summary), ("candidates.csv", candidates), ("tasks.csv", tasks)):
        if rows:
            with (directory / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                writer.writeheader()
                writer.writerows(rows)


def benchmark(args):
    if args.tasks < 1 or args.samples < 1 or args.max_attempts < 1:
        raise ValueError("Task/sample/retry budgets must be positive")
    train_steps = None
    if any(method != "baseline" for method in args.methods):
        train_steps = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["config"]["num_train_timesteps"]
    configurations = [(method, steps) for method in args.methods
                      for steps in ([0] if method == "baseline" else args.steps if method == "ddim" else [args.ddpm_steps or train_steps])]
    summaries, rows, task_rows = [], [], []
    for method, steps in configurations:
        label = "baseline" if method == "baseline" else f"{method}_{steps}"
        sampler = None if method == "baseline" else GraspSampler(args.checkpoint, method, steps, args.guidance, args.device, args.seed)
        counts, generated_types, accepted_types = Counter(), Counter(), Counter()
        pose_coverages, type_coverages = [], []
        for task in range(args.tasks):
            seed, shape = args.seed + task, task % 2
            env = scene(seed, shape, args.obstacles)
            try:
                pos, half, quat = object_info(env)
                reference = reference_candidates(env)
                if args.mode in ("isolation", "both"):
                    valid_reference = [c for c in reference if validate_candidate(env, c)[0]]
                    if sampler is None:
                        samples, vectors = [], None
                        started = time.perf_counter()
                        while len(samples) < args.samples:
                            batch = reference_candidates(env)
                            if not batch:
                                break
                            samples.extend(batch)
                        samples = samples[:args.samples]
                        sample_ms = (time.perf_counter() - started) * 1000
                    else:
                        if task == 0:
                            sampler.sample(pos, half, quat, args.samples)  # discarded timing warm-up
                        sampler.reseed(seed)
                        vectors = sampler.sample(pos, half, quat, args.samples)
                        sample_ms = sampler.last_sampling_ms
                        samples = []
                        for vector in vectors:
                            try:
                                samples.append(candidate_from_vector(vector[:, 0].numpy()))
                            except ValueError:
                                samples.append(None)
                    counts["generated"] += len(samples)
                    counts["sampling_ms"] += sample_ms
                    accepted = []
                    for index, candidate in enumerate(samples):
                        name = "invalid" if candidate is None else candidate.grasp_type.value
                        if candidate is not None:
                            generated_types[name] += 1
                        ok, reason = (False, "geometry") if candidate is None else validate_candidate(env, candidate)
                        if reason == "geometry":
                            counts["geometry_rejected"] += 1
                        else:
                            counts["ik_tested"] += 1
                            if not ok:
                                counts["ik_rejected"] += 1
                        if ok:
                            accepted.append(candidate)
                            accepted_types[name] += 1
                        pick_ok = None
                        if args.execute_samples:
                            pick_ok = False
                            if ok:
                                trial = scene(seed, shape, args.obstacles)
                                try:
                                    seed_planning(seed * 100 + index)
                                    if sampler is None:
                                        result = run_baseline(trial, candidate=candidate, attachment=not args.physical,
                                                              max_iterations=args.rrt_iterations)
                                    else:
                                        result = run_diffusion(trial, sampler, final_grasp=vectors[index:index + 1],
                                                               attachment=not args.physical, max_attempts=1,
                                                               max_iterations=args.rrt_iterations)
                                    pick_ok = result["grasp_success"]
                                    if not pick_ok:
                                        reason = result["failure_reason"]
                                finally:
                                    close_env(trial)
                            counts["pick_successes"] += int(pick_ok)
                        rows.append({"method": label, "task": task, "seed": seed, "shape": shape,
                                     "candidate": index, "grasp_type": name, "ik_accepted": ok,
                                     "grasp_success": pick_ok, "failure_reason": reason})
                    types = {c.grasp_type.value for c in valid_reference}
                    if types:
                        type_coverages.append(len(types & {c.grasp_type.value for c in accepted}) / len(types))
                    if valid_reference:
                        pose_coverages.append(sum(any(matches(c, ref) for c in accepted) for ref in valid_reference) / len(valid_reference))

                if args.mode in ("tasks", "both"):
                    # Run the original scripts on a new scene after isolation so
                    # rendering/IK/failed candidates cannot contaminate the task.
                    task_env = scene(seed, shape, args.obstacles)
                    try:
                        goal = free_goal(task_env, seed, args.obstacles)
                        seed_planning(seed + 200000)
                        with (task_env.launch_viewer() if args.viewer else nullcontext()):
                            if sampler is None:
                                result = run_baseline(task_env, place_pos=goal, attachment=not args.physical,
                                                      render=args.render, max_iterations=args.rrt_iterations)
                            else:
                                sampler.reseed(seed + 300000)
                                result = run_diffusion(task_env, sampler, place_pos=goal, attachment=not args.physical,
                                                       render=args.render, max_attempts=args.max_attempts,
                                                       max_iterations=args.rrt_iterations)
                        counts["tasks"] += 1
                        counts["task_pick_successes"] += int(result["grasp_success"])
                        counts["pick_place_successes"] += int(result["pick_place_success"])
                        task_rows.append({"method": label, "task": task, "seed": seed, "shape": shape,
                                          "goal": goal.tolist(), **result})
                    finally:
                        close_env(task_env)
                print(f"{label}: {task + 1}/{args.tasks} scenes", flush=True)
            finally:
                close_env(env)
        summary = {"method": label, "attachment_assisted": not args.physical, **dict(counts),
                   "grasp_success_rate": fraction(counts["pick_successes"], counts["generated"]) if args.execute_samples else None,
                   "ik_rejection_fraction": fraction(counts["ik_rejected"], counts["ik_tested"]),
                   "geometry_rejection_fraction": fraction(counts["geometry_rejected"], counts["generated"]),
                   "sampling_ms_per_candidate": fraction(counts["sampling_ms"], counts["generated"]),
                   "distinct_generated_types": len(generated_types), "distinct_accepted_types": len(accepted_types),
                   "type_coverage": float(np.mean(type_coverages)) if type_coverages else None,
                   "pose_coverage": float(np.mean(pose_coverages)) if pose_coverages else None,
                   "pick_place_success_rate": fraction(counts["pick_place_successes"], counts["tasks"])}
        for name in ("top_down_y", "top_down_x", "front"):
            summary[f"generated_{name}"] = generated_types[name]
            summary[f"accepted_{name}"] = accepted_types[name]
        summaries.append(summary)
        write_results(args, summaries, rows, task_rows)
    print(f"Results: {args.output}")


def main(default_methods=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/grasp/best.pt")
    p.add_argument("--methods", nargs="+", choices=("baseline", "ddim", "ddpm"), default=default_methods or ["baseline", "ddim", "ddpm"])
    p.add_argument("--steps", nargs="+", type=int, default=[50], help="DDIM step counts")
    p.add_argument("--ddpm-steps", type=int, default=0, help="0 uses the complete training chain")
    p.add_argument("--mode", choices=("isolation", "tasks", "both"), default="isolation")
    p.add_argument("--execute-samples", action="store_true")
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--guidance", type=float, default=3.)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--obstacles", action="store_true")
    p.add_argument("--physical", action="store_true")
    p.add_argument("--rrt-iterations", type=int, default=2000)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--render", action="store_true", help="Save the original before/after plots")
    p.add_argument("--output", default="results/comparison")
    args = p.parse_args()
    if args.render:
        import os
        os.environ["GRASP_FIGURE_DIR"] = str(Path(args.output) / "frames")
    benchmark(args)


if __name__ == "__main__":
    main()
