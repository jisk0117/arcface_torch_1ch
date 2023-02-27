"""
@author: Signatrix GmbH
Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)
"""
import torch.nn as nn
import torch.cuda.amp as amp
from math import sqrt

NUM_CLASSES = 512


class AnyNetX(nn.Module):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio, fp16, mode):
        import logging
        if "silu" in mode:
            from backbones_anynet.modules_silu import Stem, Stage, Head
            logging.info(": [anynet mode] programmed: SiLU / input: " + mode)
        elif "prelu" in mode:
            from backbones_anynet.modules_prelu import Stem, Stage, Head
            logging.info(": [anynet mode] programmed: PReLU / input: " + mode)
        elif "squeeze" in mode:
            from backbones_anynet.modules_squeeze import Stem, Stage, Head
            logging.info(": [anynet mode] programmed: Squeeze / input: " + mode)
        elif "prlsq" in mode:
            from backbones_anynet.modules_prlsq import Stem, Stage, Head
            logging.info(": [anynet mode] programmed: PReLU+Squeeze / input: " + mode)
        elif "pgc" in mode:
            from backbones_anynet.modules_pgc import Stem, Stage, Head
            logging.info(": [anynet mode] programmed: PGC / input: " + mode)
        else:
            from backbones_anynet.modules import Stem, Stage, Head
            logging.info(": [anynet mode] programmed: None / input: " + mode)

        super(AnyNetX, self).__init__()
        self.fp16 = fp16
        # For each stage, at each layer, number of channels (block width / bottleneck ratio) must be divisible by group width
        for block_width, bottleneck_ratio, group_width in zip(ls_block_width, ls_bottleneck_ratio, ls_group_width):
            assert block_width % (bottleneck_ratio * group_width) == 0
        self.net = nn.Sequential()
        #prev_block_width = 32
        prev_block_width = 16
        self.net.add_module("stem", Stem(prev_block_width))

        for i, (num_blocks, block_width, bottleneck_ratio, group_width) in enumerate(zip(ls_num_blocks, ls_block_width,
                                                                                         ls_bottleneck_ratio,
                                                                                         ls_group_width)):
            # self.net.add_module("stage_{}".format(i), Stage(num_blocks, prev_block_width, block_width, bottleneck_ratio, group_width, stride, se_ratio))
            if "pgc" in mode:
                self.net.add_module("stage_{}".format(i), Stage(num_blocks, prev_block_width, block_width, bottleneck_ratio, group_width, stride, se_ratio, mode))
            else:
                self.net.add_module("stage_{}".format(i), Stage(num_blocks, prev_block_width, block_width, bottleneck_ratio, group_width, stride, se_ratio))
            prev_block_width = block_width
        self.net.add_module("head", Head(ls_block_width[-1], NUM_CLASSES, fp16))
        self.initialize_weight()

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                if hasattr(m.bias, "data"):
                    m.bias.data.zero_()

    def forward(self, x):
        with amp.autocast(self.fp16):
            x = self.net(x)
        return x


class AnyNetXb(AnyNetX):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio, fp16):
        super(AnyNetXb, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio, fp16)
        assert len(set(ls_bottleneck_ratio)) == 1


class AnyNetXc(AnyNetXb):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio, fp16):
        super(AnyNetXc, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio, fp16)
        assert len(set(ls_group_width)) == 1


class AnyNetXd(AnyNetXc):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio, fp16):
        super(AnyNetXd, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio, fp16)
        assert all(i <= j for i, j in zip(ls_block_width, ls_block_width[1:])) is True


class AnyNetXe(AnyNetXd):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio, fp16):
        super(AnyNetXe, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio, fp16)
        if len(ls_num_blocks) > 2:
            assert all(i <= j for i, j in zip(ls_num_blocks[:-2], ls_num_blocks[1:-1])) is True


def main():
    import os
    from backbones_anynet.config_params import ConfigParams

    # param_path = r"D:\AnyNetX_01b\AnyNetXe_500_02_200\new_3\param\AnyNetXf_200"
    # param_path = r"D:\AnyNetX_01b\AnyNetXe_500_02_200\new_3\param\AnyNetXf_200_SE"
    param_path = "\\AnyNetXf_200_SE"
    param_idx = "0197"
    p = ConfigParams()
    p.read_params(os.path.join(param_path, param_idx+".param"))
    p.se_ratio = 16

    from __init__ import get_model
    model = get_model(p)
    print(p)

    model.eval()
    import time
    import torch
    dummy = torch.randn(1, 3, 112, 112, dtype=torch.float).to("cpu")
    t = []
    for i in range(100):
        start = time.time()
        output = model(dummy)
        t.append(time.time() - start)
    import numpy as np
    print('Processing time: {:f} ({:f}) / len: {:d}'.format(np.mean(t), np.std(t), len(t)))

    from util import write_model_summary
    write_model_summary(model, os.path.join(param_path, param_idx+"_details.txt"), depth=8)


if __name__ == '__main__':
    main()
