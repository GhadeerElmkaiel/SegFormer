# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp

from mmcv.utils import TORCH_VERSION
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook
import torch
import torchvision


@HOOKS.register_module()
class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        if TORCH_VERSION < '1.1' or TORCH_VERSION == 'parrots':
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()

@HOOKS.register_module()
class TensorboardLoggerImagesHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 img_interval=50,
                 num_classes=6,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=False,
                 norm_cfg=None):
        super(TensorboardLoggerImagesHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.img_interval = img_interval
        self.log_dir = log_dir
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg

    @master_only
    def before_run(self, runner):
        if TORCH_VERSION < '1.1' or TORCH_VERSION == 'parrots':
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        # tags = self.get_loggable_tags(runner, allow_text=True)
        if self.get_iter(runner) % self.img_interval == 0:
            img = self.logImages(runner.outputs['log_images'])
            self.writer.add_image('Original | Segmentation', img)
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()

    def logImages(self,log_images):
        if self.num_classes == 7:
            # #           -- Void --     -- Mirror --      -- FUO --      -- Glass --      -- OOP --     -- Floor --   -- background  --
            # palette = [[255,255,255],[102, 255, 102], [245, 147, 49], [51, 221, 255], [184, 61, 245], [250, 50, 83], [0, 0, 0]]
            palette = [[102, 255, 102], [51, 221, 255], [245, 147, 49], [184, 61, 245], [250, 50, 83], [0, 0, 0],[255,255,255]]
        elif self.num_classes == 6:
            #           -- Mirror --      -- Glass --      -- FUO --      -- OOP --     -- Floor --   -- background  --
            palette = [[102, 255, 102], [51, 221, 255], [245, 147, 49], [184, 61, 245], [250, 50, 83], [0, 0, 0]]
        else:
            raise AssertionError('Wrong number of classes')
        palette = torch.Tensor(palette).to(torch.uint8)

        seg = torch.argmax(log_images['prediction'],1, keepdims=True).to(torch.uint8)

        color_seg = torch.zeros((seg.shape[0],3,*seg.shape[-2:],), dtype=torch.uint8)

        # Recolor the resulted image to match the needed colors
        for label, color in enumerate(palette):
            color_seg.permute(1,0,2,3)[:,(seg.permute(1,0,2,3)==label).squeeze(axis=0)] = color.view(-1,1)

        img_resize = torchvision.transforms.Resize(size=log_images['original'].shape[-2:])
        if self.norm_cfg is not None:
            mean = torch.Tensor(self.norm_cfg['mean'])
            std = torch.Tensor(self.norm_cfg['std'])
            mean = -mean / std
            std = 1 / std
            img_denormalize = torchvision.transforms.Normalize(mean=mean, std=std, inplace=True)
            original_img = img_denormalize(log_images['original'])
        else:
            original_img = log_images['original']
        return torchvision.utils.make_grid(torch.cat([original_img.to(torch.uint8).cpu(),img_resize(color_seg).to(torch.uint8)]),
                                            nrow=log_images['original'].shape[0], padding=10)
