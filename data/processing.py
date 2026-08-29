class train_dataset(Dataset):
    def __init__(self, env_info):
        self.env_info = env_info
    def __len__(self):
        return len(self.env_info)
    def __getitem__(self, idx):
        cands = torch.zeros(17,4)
        block_info = torch.zeros(10,4)

        for cand in self.env_info[idx][3]:
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
            cand_cat =torch.cat([gtype, gquat, gapos, gpos, glpos, gscore], dim=-1)

        cands[...,0] = cand_cat
        block_cat = torch.cat([torch.tensor(cand_list[idx][0]), torch.tensor(cand_list[idx][1]), torch.tensor(cand_list[idx][2])], dim = -1)
        block_info[...,0] = block_cat

        return cands, block_info
