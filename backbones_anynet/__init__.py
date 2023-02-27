# from backbones_anynet.anynet import AnyNetXe
from backbones_anynet.anynet import AnyNetX
from backbones_anynet.config_params import ConfigParams


def get_model(params, mode=""):
    # model = AnyNetXe(ls_num_blocks=params.ls_num_blocks,
    model = AnyNetX(ls_num_blocks=params.ls_num_blocks,
                    ls_block_width=params.ls_block_width,
                    ls_bottleneck_ratio=params.bottleneck_ratio,
                    ls_group_width=params.group_width,
                    stride=2,
                    se_ratio=params.se_ratio,
                    fp16=params.fp16,
                    mode=mode)
    return model
