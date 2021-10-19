# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from ...models.builder import BACKBONES
from ...utils import get_root_logger
from mmcv.runner import load_checkpoint
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x

class MixVisionDepthTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], depth_embed_type="repeat", weights_only_MVF=False, style='pytorch'):
        super().__init__()
        self.weights_only_MVF = weights_only_MVF
        self.rgb_branch = MixVisionTransformer(img_size, patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios)

        # depth_embed_type should be in ["CNN", "repeat", "HHA"] 
        # Only in the case of the CNN the first part of the depth_branch will be Conv2d, Otherwise will be Identity
        if depth_embed_type=="CNN":
            self.depth_branch= nn.Sequential(OrderedDict([
                ("Depth_embed", nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, padding_mode="replicate")),
                ("Depth_MixVisionTransformer", MixVisionTransformer(img_size, patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios))
            ])
            )

        elif depth_embed_type=="repeat":
            self.depth_branch= nn.Sequential(OrderedDict([
                ("Depth_embed", nn.Identity(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, padding_mode="same")),
                ("Depth_MixVisionTransformer", MixVisionTransformer(img_size, patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios))
            ])
            )

        elif depth_embed_type=="HHA":
            self.depth_branch= nn.Sequential(OrderedDict([
                ("Depth_embed", nn.Identity(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, padding_mode="same")),
                ("Depth_MixVisionTransformer", MixVisionTransformer(img_size, patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios))
            ])
            )

    # This function is for laoding the weights after training, that is why it is the same as in other models
    def init_weights(self, pretrained=None):
        """
        This function is for laoding the weights after training, that is why it is the same as in other models (such as MixVisionTransformer)
        """
        if not self.weights_only_MVF: 
            if isinstance(pretrained, str):
                logger = get_root_logger()
                load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
        else:
            if isinstance(pretrained, str):
                self.init_transformers_same_weights(pretrained)


    # This function is for initialize the weights of the backbones in the same way
    def init_transformers_same_weights(self, weights):
        """
        This function is for initialize the weights of the backbones in the same way
        """
        print("Initializing RGB branch")
        self.rgb_branch.init_weights(weights)
        print("Initializing Depth branch")
        # self.depth_branch["Depth_MixVisionTransformer"].init_weights(weights)
        self.depth_branch[1].init_weights(weights)
        print("Initializing done!")

        # raise ValueError
    # def __call__(self, x, x_metas):
    #     raise NotImplementedError


    # The forward should contain 2 images, RGB, Depth
    def forward(self, x, x_metas):
        # rgb = x["rgb"]
        # depth = x["depth"]
        # rgb_res = self.rgb_branch.forward(rgb)
        # depth_res = self.depth_branch["Depth_embed"](depth)
        # depth_res = self.depth_branch["Depth_MixVisionTransformer"].forward(depth_res)
        # # The image shape in one example was (1, 3, 512, 910) (the shape of x in forward)
        # # The output shape of the same example was (4, ) (the shape of xafter applying forward)
        # x = torch.cat((rgb_res, depth_res), 0) Here I am not sure about the axe

        # return x
        channels = x_metas[0].get('channels', {})
        data_sizes = []
        data_tensor_indices = {} 
        idx = 0
        for key in channels.keys():
            data_sizes.append(len(channels[key]))
            data_tensor_indices[key] = idx
            idx+=1

        splited_tensors = torch.split(x, data_sizes, dim=1)
        rgb_tensor = splited_tensors[data_tensor_indices["img"]]
        depth_tensor = splited_tensors[data_tensor_indices["depth"]]
        rgb_featurs = self.rgb_branch(rgb_tensor)
        depth_featurs = self.depth_branch(depth_tensor)

        features_merged =rgb_featurs+ depth_featurs

        return features_merged
        # raise NotImplementedError


class MixVisionDepthConcatTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], depth_embed_type="repeat", weights_only_MVF=False, style='pytorch'):
        super().__init__()
        self.weights_only_MVF = weights_only_MVF            # initializing only the partial MixedVisionTransformer
        self.rgb_branch = MixVisionTransformer(img_size, patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios)

        # depth_embed_type should be in ["CNN", "repeat", "HHA"] 
        # Only in the case of the CNN the first part of the depth_branch will be Conv2d, Otherwise will be Identity
        if depth_embed_type=="CNN":
            self.depth_branch= nn.Sequential(OrderedDict([
                ("Depth_embed", nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, padding_mode="replicate")),
                ("Depth_MixVisionTransformer", MixVisionTransformer(img_size, patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios))
            ])
            )

        elif depth_embed_type=="repeat":
            self.depth_branch= nn.Sequential(OrderedDict([
                ("Depth_embed", nn.Identity(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, padding_mode="same")),
                ("Depth_MixVisionTransformer", MixVisionTransformer(img_size, patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios))
            ])
            )

        elif depth_embed_type=="HHA":
            self.depth_branch= nn.Sequential(OrderedDict([
                ("Depth_embed", nn.Identity(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, padding_mode="same")),
                ("Depth_MixVisionTransformer", MixVisionTransformer(img_size, patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios))
            ])
            )

        self.conc_conv_1 = nn.Conv2d(embed_dims[0]*2,embed_dims[0])
        self.conc_conv_2 = nn.Conv2d(embed_dims[1]*2,embed_dims[1])
        self.conc_conv_3 = nn.Conv2d(embed_dims[2]*2,embed_dims[2])
        # self.conc_conv_4 = nn.Conv2d(embed_dims[3]*2,embed_dims[3])

        self._itter_=0

    # This function is for laoding the weights after training, that is why it is the same as in other models
    def init_weights(self, pretrained=None):
        """
        This function is for laoding the weights after training, that is why it is the same as in other models (such as MixVisionTransformer)
        """
        if not self.weights_only_MVF: 
            if isinstance(pretrained, str):
                print("weights_only_MVF: ",self.weights_only_MVF)
                logger = get_root_logger()
                load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
        else:
            print("weights_only_MVF: ",self.weights_only_MVF)
            if isinstance(pretrained, str):
                self.init_transformers_same_weights(pretrained)

    # This function is for initialize the weights of the backbones in the same way
    def init_transformers_same_weights(self, weights):
        """
        This function is for initialize the weights of the backbones in the same way
        """
        print("Initializing RGB branch")
        self.rgb_branch.init_weights(weights)
        print("Initializing Depth branch")
        # self.depth_branch["Depth_MixVisionTransformer"].init_weights(weights)
        self.depth_branch[1].init_weights(weights)
        print("Initializing done!")

        # raise ValueError

    # def __call__(self, x, x_metas):
    #     raise NotImplementedError


    # The forward should contain 2 images, RGB, Depth
    def forward(self, x, x_metas):


        # return x
        channels = x_metas[0].get('channels', {})
        data_sizes = []
        data_tensor_indices = {} 
        idx = 0
        for key in channels.keys():
            data_sizes.append(len(channels[key]))
            data_tensor_indices[key] = idx
            idx+=1

        splited_tensors = torch.split(x, data_sizes, dim=1)
        rgb_tensor = splited_tensors[data_tensor_indices["img"]]
        depth_tensor = splited_tensors[data_tensor_indices["depth"]]

        depth_featurs = self.depth_branch(depth_tensor)

        x = rgb_tensor
        rgb_featurs = []
        B = x.shape[0]

        # stage 1
        x, H, W = self.rgb_branch.patch_embed1(x)
        for i, blk in enumerate(self.rgb_branch.block1):
            x = blk(x, H, W)
        x = self.rgb_branch.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_featurs.append(x)

        # Concatinate first depth features with first RGB features
        concat = torch.cat([x, depth_featurs[0]])
        x = self.conc_conv_1(concat)

        # stage 2
        x, H, W = self.rgb_branch.patch_embed2(x)
        for i, blk in enumerate(self.rgb_branch.block2):
            x = blk(x, H, W)
        x = self.rgb_branch.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_featurs.append(x)

        # Concatinate first depth features with first RGB features
        concat = torch.cat([x, depth_featurs[1]])
        x = self.conc_conv_2(concat)


        # stage 3
        x, H, W = self.rgb_branch.patch_embed3(x)
        for i, blk in enumerate(self.rgb_branch.block3):
            x = blk(x, H, W)
        x = self.rgb_branch.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_featurs.append(x)

        # Concatinate first depth features with first RGB features
        concat = torch.cat([x, depth_featurs[2]])
        x = self.conc_conv_3(concat)


        # stage 4
        x, H, W = self.rgb_branch.patch_embed4(x)
        for i, blk in enumerate(self.rgb_branch.block4):
            x = blk(x, H, W)
        x = self.rgb_branch.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_featurs.append(x)

        # rgb_featurs = self.rgb_branch(rgb_tensor)
        # depth_featurs = self.depth_branch(depth_tensor)



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x



@BACKBONES.register_module()
class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
##########################################################################################


@BACKBONES.register_module()
class mit_depth_b0(MixVisionDepthTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


@BACKBONES.register_module()
class mit_depth_b1(MixVisionDepthTransformer):
    def __init__(self, **kwargs):
        super(mit_depth_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


@BACKBONES.register_module()
class mit_depth_b2(MixVisionDepthTransformer):
    def __init__(self, **kwargs):
        super(mit_depth_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


@BACKBONES.register_module()
class mit_depth_b3(MixVisionDepthTransformer):
    def __init__(self, **kwargs):
        super(mit_depth_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


@BACKBONES.register_module()
class mit_depth_b4(MixVisionDepthTransformer):
    def __init__(self, **kwargs):
        super(mit_depth_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


@BACKBONES.register_module()
class mit_depth_b5(MixVisionDepthTransformer):
    def __init__(self, **kwargs):
        super(mit_depth_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs) 