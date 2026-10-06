import torch
from data.collect_data import collect_data, close_env
from data.processing import unit_quat


def prepare_test_data(n=3, seed=123, normalizer=None):
    # one time test data processing for model inference

    test_block, b, env, _, _ = collect_data(n, seed=seed)
    # collect_data returns the final live environment; use its matching scene.
    test_block = test_block[-1:]
    all_test_cands = []

    test_cands = torch.zeros(17,4)
    test_block_info = torch.zeros(10,4)

    for cand in test_block[0][3]:
        if cand.grasp_type.value == "top_down_y":
            gtype = [1.0, 0.0, 0.0]
        elif cand.grasp_type.value == "top_down_x":
            gtype = [0.0, 1.0, 0.0]
        elif cand.grasp_type.value == "front":
            gtype = [0.0, 0.0, 1.0]
        else:
            close_env(env)
            raise ValueError(f"Unsupported grasp type: {cand.grasp_type.value}")
        gscore = torch.tensor((cand.score - (-15))/(40 + 0.0001)).unsqueeze(-1)
        gtype = torch.tensor(gtype)
        gquat = torch.tensor(unit_quat(cand.grasp_quat))
        gapos =  torch.tensor(cand.approach_pos)
        gpos = torch.tensor(cand.grasp_pos)
        glpos = torch.tensor(cand.lift_pos)
        #print(gscore.shape, gtype.shape, gquat.shape, gapos.shape, gpos.shape, glpos.shape)
        test_cand_cat =torch.cat([gtype, gquat, gapos, gpos, glpos, gscore], dim=-1)
        all_test_cands.append(test_cand_cat)

    if not all_test_cands:
        close_env(env)
        raise ValueError("The selected scene has no grasp candidates")
    test_cands[...,0] = all_test_cands[0]
    test_block_cat = torch.cat([torch.tensor(test_block[0][0]), torch.tensor(test_block[0][1]), torch.tensor(unit_quat(test_block[0][2]))], dim = -1)
    test_block_info[...,0] = test_block_cat
    #block_info[...,1] = torch.tensor(obs_list[0])
    #block_info[...,2] = torch.tensor(obs_list[1])

    print("Test Input shape:", test_cands.shape, test_block_info.shape)
    print(test_cands, test_block_info)

    if normalizer is not None:
        test_cands[:, 0] = (test_cands[:, 0] - normalizer["grasp_mean"]) / normalizer["grasp_std"]
        test_block_info[:, 0] = (test_block_info[:, 0] - normalizer["block_mean"]) / normalizer["block_std"]
    return test_cands, test_block_info, env, all_test_cands


if __name__ == "__main__":
    _, _, env, _ = prepare_test_data()
    close_env(env)
