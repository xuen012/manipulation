import numpy as np
import torch
from torch.utils.data import Dataset


class train_dataset(Dataset):
    def __init__(self, env_info, normalizer=None, sequence_length=4):
        self.env_info = env_info
        self.normalizer = normalizer
        self.sequence_length = sequence_length
        # Index candidates individually so the existing encoding loop does not
        # overwrite all but the last demonstration from each scene.
        self.index = [(scene, candidate) for scene, row in enumerate(env_info)
                      for candidate in range(len(row[3]))]
        if not self.index:
            raise ValueError("The dataset contains no grasp candidates")
    def __len__(self):
        return len(self.index)
    def __getitem__(self, idx):
        idx, candidate_idx = self.index[idx]
        cands = torch.zeros(17,self.sequence_length)
        block_info = torch.zeros(10,self.sequence_length)

        # conver Grasp_candidate to model training data format
        # block includes info about block to be grasped.
        for cand in self.env_info[idx][3][candidate_idx:candidate_idx + 1]:
            if cand.grasp_type.value == "top_down_y":
                gtype = [1.0, 0.0, 0.0]
            elif cand.grasp_type.value == "top_down_x":
                gtype = [0.0, 1.0, 0.0]
            elif cand.grasp_type.value == "front":
                gtype = [0.0, 0.0, 1.0]
            else:
                raise ValueError(f"Unsupported grasp type: {cand.grasp_type.value}")
            gscore = torch.tensor((cand.score - (-15))/(40 + 0.0001)).unsqueeze(-1)
            gtype = torch.tensor(gtype)
            gquat = torch.tensor(unit_quat(cand.grasp_quat))
            gapos =  torch.tensor(cand.approach_pos)
            gpos = torch.tensor(cand.grasp_pos)
            glpos = torch.tensor(cand.lift_pos)
            #print(gscore.shape, gtype.shape, gquat.shape, gapos.shape, gpos.shape, glpos.shape)
            cand_cat =torch.cat([gtype, gquat, gapos, gpos, glpos, gscore], dim=-1)

        cands[...,0] = cand_cat
        block_cat = torch.cat([torch.tensor(self.env_info[idx][0]), torch.tensor(self.env_info[idx][1]), torch.tensor(unit_quat(self.env_info[idx][2]))], dim = -1)
        block_info[...,0] = block_cat

        if not torch.isfinite(cands).all() or not torch.isfinite(block_info).all():
            raise ValueError("Non-finite demonstration")
        if self.normalizer is not None:
            cands[:, 0] = (cands[:, 0] - self.normalizer["grasp_mean"]) / self.normalizer["grasp_std"]
            block_info[:, 0] = (block_info[:, 0] - self.normalizer["block_mean"]) / self.normalizer["block_std"]
        return cands, block_info


# The original PyTorch script referred to this spelling.
train_dataset2 = train_dataset


def unit_quat(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-7:
        raise ValueError("Expected a finite, nonzero wxyz quaternion")
    quat /= np.linalg.norm(quat)
    if quat[np.argmax(np.abs(quat))] < 0:
        quat *= -1
    return quat


def fit_normalizer(env_info):
    dataset = train_dataset(env_info)
    grasps, blocks = zip(*(dataset[i] for i in range(len(dataset))))
    grasp = torch.stack(grasps)[:, :, 0]
    block = torch.stack(blocks)[:, :, 0]
    return {"grasp_mean": grasp.mean(0), "grasp_std": grasp.std(0, unbiased=False).clamp_min(.01),
            "block_mean": block.mean(0), "block_std": block.std(0, unbiased=False).clamp_min(.001)}


def condition_tensor(pos, half, quat, normalizer, sequence_length=4):
    pos, half = np.asarray(pos), np.asarray(half)
    if pos.shape != (3,) or half.shape != (3,) or not np.isfinite([pos, half]).all() or np.any(half <= 0):
        raise ValueError("Position and positive half-size must be finite xyz vectors")
    block_info = torch.zeros(10, sequence_length)
    block_info[:, 0] = torch.tensor(np.r_[pos, half, unit_quat(quat)], dtype=torch.float32)
    if normalizer is not None:
        block_info[:, 0] = (block_info[:, 0] - normalizer["block_mean"]) / normalizer["block_std"]
    return block_info


def denormalize_grasps(grasps, normalizer):
    grasps = grasps.detach().cpu().clone()
    if normalizer is not None:
        grasps[..., 0] = grasps[..., 0] * normalizer["grasp_std"] + normalizer["grasp_mean"]
    return grasps


def candidate_from_vector(vector):
    from tampanda.planners.grasp_planner import GraspCandidate, GraspType
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (17,) or not np.isfinite(vector).all():
        raise ValueError("Expected 17 finite grasp values")
    return GraspCandidate(
        grasp_type=GraspType(("top_down_y", "top_down_x", "front")[int(vector[:3].argmax())]),
        grasp_quat=unit_quat(vector[3:7]), approach_pos=vector[7:10].copy(),
        grasp_pos=vector[10:13].copy(), lift_pos=vector[13:16].copy(),
        score=float(vector[16] * 40.0001 - 15))
