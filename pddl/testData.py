test_block, b, env, _, _ = collect_data(3)

test_cands = torch.zeros(17,4)
test_block_info = torch.zeros(10,4)

for cand in test_block[0][3]:
    if cand.grasp_type.value == "top_down_y":
        gtype = [1.0, 0.0, 0.0]
    elif cand.grasp_type.value == "top_down_x":
        gtype = [0.0, 1.0, 0.0]
    elif cand.grasp_type.value == "front":
        gtype = [0.0, 0.0, 1.0]
    gscore = torch.tensor((cand.score - (-15))/(40 + 0.0001)).unsqueeze(-1)
    gtype = torch.tensor(gtype)
    gquat = torch.tensor(cand.grasp_quat)
    gapos =  torch.tensor(cand.approach_pos)
    gpos = torch.tensor(cand.grasp_pos)
    glpos = torch.tensor(cand.lift_pos)
    #print(gscore.shape, gtype.shape, gquat.shape, gapos.shape, gpos.shape, glpos.shape)
    test_cand_cat =torch.cat([gtype, gquat, gapos, gpos, glpos, gscore], dim=-1)

test_cands[...,0] = test_cand_cat
test_block_cat = torch.cat([torch.tensor(test_block[0][0]), torch.tensor(test_block[0][1]), torch.tensor(test_block[0][2])], dim = -1)
test_block_info[...,0] = test_block_cat
#block_info[...,1] = torch.tensor(obs_list[0])
#block_info[...,2] = torch.tensor(obs_list[1])

print("Test Input shape:", test_cands.shape, test_block_info.shape)
print(test_cands, test_block_info)
