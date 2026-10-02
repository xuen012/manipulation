"""Persistence for collect_data's existing [pos, half, quat, candidates, shape] records."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from data.processing import train_dataset


def save_data(path, records, metadata=None):
    dataset = train_dataset(records)
    pairs = [dataset[i] for i in range(len(dataset))]
    grasps = np.stack([g[:, 0].numpy() for g, _ in pairs])
    blocks = np.stack([b[:, 0].numpy() for _, b in pairs])
    scene_ids = np.asarray([scene for scene, _ in dataset.index], dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, grasps=grasps, blocks=blocks, scene_ids=scene_ids,
                            metadata=json.dumps(metadata or {}))
    return len(dataset)


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        grasps, blocks, scene_ids = archive["grasps"], archive["blocks"], archive["scene_ids"]
    if grasps.ndim != 2 or grasps.shape[1] != 17 or blocks.shape != (len(grasps), 10) or scene_ids.shape != (len(grasps),):
        raise ValueError("Expected grasps [N,17], blocks [N,10], and scene_ids [N]")
    if not len(grasps) or not np.isfinite(grasps).all() or not np.isfinite(blocks).all():
        raise ValueError("Empty or non-finite dataset")
    records = []
    for scene_id in np.unique(scene_ids):
        mask = scene_ids == scene_id
        block = blocks[mask][0]
        candidates = []
        for row in grasps[mask]:
            candidates.append(SimpleNamespace(
                grasp_type=SimpleNamespace(value=("top_down_y", "top_down_x", "front")[int(row[:3].argmax())]),
                grasp_quat=row[3:7].copy(), approach_pos=row[7:10].copy(),
                grasp_pos=row[10:13].copy(), lift_pos=row[13:16].copy(), score=float(row[16] * 40.0001 - 15)))
        records.append([block[:3].copy(), block[3:6].copy(), block[6:10].copy(), candidates, 2 * block[3:6]])
    return records
