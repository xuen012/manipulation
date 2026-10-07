"""Drop-in native planner adapter around the existing model and sampling loop."""
from collections import Counter
from tampanda import GraspPlanner
from data.processing import candidate_from_vector
from model.sampler import GraspSampler
from pddl.checks import table_height, validate_candidate


class DiffusionGraspPlanner(GraspPlanner):
    def __init__(self, checkpoint, env, num_candidates=16, max_rounds=5, **options):
        super().__init__(table_z=table_height(env))
        if num_candidates < 1 or max_rounds < 1:
            raise ValueError("Sampling budgets must be positive")
        self.env = env
        self.sampler = GraspSampler(checkpoint, **options)
        self.num_candidates, self.max_rounds = num_candidates, max_rounds
        self.last_stats = {}

    def generate_candidates(self, object_pos, half_size, object_quat=None):
        object_quat = [1, 0, 0, 0] if object_quat is None else object_quat
        stats = Counter()
        accepted = []
        for _ in range(self.max_rounds):
            samples = self.sampler.sample(object_pos, half_size, object_quat, self.num_candidates)
            stats["sampling_ms"] += self.sampler.last_sampling_ms
            for sample in samples:
                stats["generated"] += 1
                try:
                    candidate = candidate_from_vector(sample[:, 0].numpy())
                except ValueError:
                    stats["geometry_rejected"] += 1
                    continue
                ok, reason = validate_candidate(self.env, candidate, object_pos, half_size)
                if reason == "geometry":
                    stats["geometry_rejected"] += 1
                else:
                    stats["ik_tested"] += 1
                    if not ok:
                        stats["ik_rejected"] += 1
                if ok:
                    accepted.append(candidate)
            if accepted:
                break
        stats["accepted"] = len(accepted)
        self.last_stats = dict(stats)
        return sorted(accepted, key=lambda c: c.score, reverse=True)
