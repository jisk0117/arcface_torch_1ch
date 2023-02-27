import os
from random import choice, choices
from easydict import EasyDict
from backbones_anynet.config_AnyNetXf import *


class ConfigParams(EasyDict):
    def __init__(self, path=None, ls_num_blocks=None, ls_block_width=None, bottleneck_ratio=None,
                 group_width=None, k=4):
        if path is None:
            self.k = int(k)
            self.ls_num_blocks = self.get_choices(base_range=NUM_BLOCKS, sort=True) if ls_num_blocks is None else ls_num_blocks
            self.ls_block_width = self.get_choices(base_range=BLOCK_WIDTH, sort=True) if ls_block_width is None else ls_block_width
            self.bottleneck_ratio = self.get_choice(base_range=BOTTLENECK_RATIO) if bottleneck_ratio is None else [bottleneck_ratio] * k
            self.group_width = self.get_choice(base_range=GROUP_WIDTH) if group_width is None else [group_width] * k
            self.se_ratio = SE_RATIO
            self.fp16 = FP16

            while self.assert_wbg():
                self.ls_block_width = self.get_choices(base_range=BLOCK_WIDTH, sort=True)
                self.bottleneck_ratio = self.get_choice(base_range=BOTTLENECK_RATIO)
                # self.group_width = self.get_choice(base_range=GROUP_WIDTH)
            while self.assert_n():
                self.ls_num_blocks = self.get_choices(base_range=NUM_BLOCKS, sort=True)
        else:
            self.read_params(path=path)

    def get_choice(self, base_range, target_range=None):
        return [choice(base_range)] * self.k if target_range is None else [target_range] * self.k

    def get_choices(self, base_range, target_range=None, sort=False):
        samples = choices(base_range, k=self.k) if target_range is None else target_range
        samples.sort() if sort else None
        return samples

    def assert_wbg(self):
        for w, b, g in zip(self.ls_block_width, self.bottleneck_ratio, self.group_width):
            if w % (b * g) is not 0:
                return True
        return False

    def assert_n(self):
        if sum(self.ls_num_blocks) > 25:
            return True
        return False

    def print_params(self):
        for key, value in self.items():
            num_space = 25 - len(key)
            print(": " + key + " "*num_space + str(value))

    def write_params(self, path, i):
        with open(os.path.join(path, "{:04d}.param".format(i)), 'w') as f:
            for key, value in self.items():
                num_space = 25 - len(key)
                f.write(key + " "*num_space + str(value) + '\n')

    def read_params(self, path):
        with open(path, 'r') as f:
            for line in f:
                items = line.strip().split()
                if '[' in items[1]:
                    if hasattr(self, items[0]):
                        ls_name = getattr(self, items[0])
                        ls_name.clear()
                    else:
                        setattr(self, items[0], [])
                        ls_name = getattr(self, items[0])
                    for x in items[1:]:
                        ls_name.append(int(x.strip(',').strip('[').strip(']')))
                else:
                    if items[1].lower() == 'none':
                        items[1] = None
                    elif items[1].lower() == 'true':
                        items[1] = True
                    elif items[1].lower() == 'false':
                        items[1] = False
                    setattr(self, items[0], items[1])
