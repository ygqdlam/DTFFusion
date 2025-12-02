from model import MainNet, LTE, SearchTransfer

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.EDcoder import Decoder, FEncoder







class TTSR(nn.Module):
    def __init__(self, args):
        super(TTSR, self).__init__()
        self.args = args
        self.num_res_blocks = list( map(int, args.num_res_blocks.split('+')) )
        self.MainNet = MainNet.MainNet(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, 
            res_scale=args.res_scale)
        self.LTE      = LTE.LTE(requires_grad=True)
        self.LTE_copy = LTE.LTE(requires_grad=False) ### used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer.SearchTransfer()
        self.decoder = Decoder()


    def forward(self, lr=None, lrsr=None, ref_1=None, refsr_1=None, sr=None,ref_2=None,refsr_2=None,val = False):
        T_lv3_list = [None] * self.args.refs
        T_lv2_list = [None] * self.args.refs
        T_lv1_list = [None] * self.args.refs
        S_list = [None] * self.args.refs
        sr_1_list = []
        sr_2_list = []

        ref_list = [None] * self.args.refs

        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy(sr)
            return sr_lv1, sr_lv2, sr_lv3


        _, _, lrsr_lv3  = self.LTE(lrsr.detach())   #lrsr代表先降采样再上采样。LTE纹理特征提取器。准确而适当的纹理信息有助于生成SR图像。
        _, _, refsr_lv3 = self.LTE(refsr_1.detach())  #refer代表先将采样再上采样的参考图像。

        ref_lv1, ref_lv2, ref_lv3 = self.LTE(ref_1.detach())  #ref参考图像。

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)


        T_lv3_list[0] = T_lv3
        T_lv2_list[0] = T_lv2
        T_lv1_list[0] = T_lv1
        S_list[0] = S

        ### 反馈机制必须有的
        self.MainNet.reset_state()
        for _ in range(self.args.num_steps):
            sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)
            ref_list[0] = sr    ###拿出最后一个未经过解码器的特征。
            sr_result = self.decoder(sr)
            sr_1_list.append(sr_result)
        if(self.args.refs == 2):
            _, _, refsr_2_lv3 = self.LTE((refsr_2.detach() + 1.) / 2.)  # refer代表先将采样再上采样的参考图像。
            ref_2_lv1, ref_2_lv2, ref_2_lv3 = self.LTE((ref_2.detach() + 1.) / 2.)  # ref参考图像。

            S_2, T_2_lv3, T_2_lv2, T_2_lv1 = self.SearchTransfer(lrsr_lv3, refsr_2_lv3, ref_2_lv1, ref_2_lv2, ref_2_lv3)
            T_lv3_list[1] = T_2_lv3
            T_lv2_list[1] = T_2_lv2
            T_lv1_list[1] = T_2_lv1
            S_list[1] = S_2

            ### 反馈机制必须有的
            self.MainNet.reset_state()
            for _ in range(self.args.num_steps):
                sr_2 = self.MainNet(lr, S_2, T_2_lv3, T_2_lv2, T_2_lv1)
                ref_list[1] = sr_2  ###拿出最后一个未经过解码器的特征。
                sr_2_result = self.decoder(sr_2)
                sr_2_list.append(sr_2_result)

            if(self.args.eval == True or self.args.test == True  or val == True):
                one = ref_list[0].new_tensor(1.0)  # 这就是新建一个值为1的向量，干嘛还非要用inputs[0]新建一个tensor
                epsilon = ref_list[0].new_tensor(1e-8)
                prev_dist = torch.abs(ref_list[0]) + epsilon  # 此时的特征也可能有负值，首先取绝对值，再加上一个值，这是为了防止太接近0
                next_dist = torch.abs(ref_list[1]) + epsilon
                prev_mask = one.div(prev_dist).div(one.div(prev_dist) + one.div(next_dist))  # 公式规定就是这样
                prev_mask = prev_mask.clamp_(0.0, 1.0)  # 融合时的权重
                next_mask = one - prev_mask
                result = (prev_mask * (ref_list[0]) +
                          next_mask * (ref_list[1]))
                res = self.decoder(result)
                return res, S_list, T_lv3_list, T_lv2_list, T_lv1_list
            else:
                return sr_1_list,sr_2_list, S_list, T_lv3_list, T_lv2_list, T_lv1_list
        if (self.args.eval == True or self.args.test == True or val == True):
            return sr_1_list[len(sr_1_list)-1], S_list, T_lv3_list, T_lv2_list, T_lv1_list
        return sr_1_list,sr_2_list, S_list, T_lv3_list, T_lv2_list, T_lv1_list
