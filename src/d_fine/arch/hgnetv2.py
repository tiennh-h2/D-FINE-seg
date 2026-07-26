"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dist_utils import get_rank, synchronize
from .common import FrozenBatchNorm2d

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ["HGNetv2"]


def _create_vit_backbone():
    """Create ViT backbone for visual feature extraction."""
    from sam3.model.vitdet import ViT
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        # compile_mode=compile_mode,
        compile_mode=None,
    )

class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        groups=1,
        padding="",
        use_act=True,
        use_lab=False,
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == "same":
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(in_chs, out_chs, kernel_size, stride, groups=groups, bias=False),
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        groups=1,
        use_lab=False,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    # for HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class HG_Block(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        layer_num,
        kernel_size=3,
        residual=False,
        light_block=False,
        use_lab=False,
        agg="ese",
        drop_path=0.0,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    )
                )

        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        if agg == "se":
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        block_num,
        layer_num,
        downsample=True,
        light_block=False,
        kernel_size=3,
        use_lab=False,
        agg="se",
        drop_path=0.0,
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class HGNetv2(nn.Module):
    """
    HGNetV2
    Args:
        stem_channels: list. Number of channels for the stem block.
        stage_type: str. The stage configuration of HGNet. such as the number of channels, stride, etc.
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Layer. Specific HGNetV2 model depends on args.
    """

    arch_configs = {
        "B0": {
            "stem_channels": [3, 16, 16],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth",
        },
        "B1": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth",
        },
        "B2": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth",
        },
        "B3": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth",
        },
        "B4": {
            "stem_channels": [3, 32, 48],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth",
        },
        "B5": {
            "stem_channels": [3, 32, 64],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth",
        },
        "B6": {
            "stem_channels": [3, 48, 96],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth",
        },
    }

    def __init__(
        self,
        name,
        use_lab=False,
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        pretrained=True,
        local_model_dir="weight/hgnetv2/",
        in_channels: int = 3,
        use_sam2_features: bool = False,
        use_sam3_features: bool = False,
        sam_checkpoint_path: str | None = None,
        fuse_sam: bool = True,
        train_adapter_block: bool = False
    ):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        # Copy stem_channels (it's a class-level list — mutating in place would
        # corrupt arch_configs for subsequent builds).
        stem_channels = list(self.arch_configs[name]["stem_channels"])
        stem_channels[0] = in_channels
        stage_config = self.arch_configs[name]["stage_config"]
        download_url = self.arch_configs[name]["url"]

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.stem = StemBlock(
            in_chs=stem_channels[0],
            mid_chs=stem_channels[1],
            out_chs=stem_channels[2],
            use_lab=use_lab,
        )

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            (
                in_channels,
                mid_channels,
                out_channels,
                block_num,
                downsample,
                light_block,
                kernel_size,
                layer_num,
            ) = stage_config[k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                )
            )

        # Skip freeze when stem was inflated for extra channels
        if freeze_at >= 0 and in_channels == 3:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:
                model_path = local_model_dir + "PPHGNetV2_" + name + "_stage1.pth"
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location="cpu")
                    print(f"Loaded stage1 {name} HGNetV2 from local file.")
                else:
                    # If the file doesn't exist locally, download from the URL
                    if get_rank() == 0:
                        print(
                            GREEN
                            + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection."
                            + RESET
                        )
                        print(
                            GREEN
                            + "Please check your network connection. Or download the model manually from "
                            + RESET
                            + f"{download_url}"
                            + GREEN
                            + " to "
                            + RESET
                            + f"{local_model_dir}."
                            + RESET
                        )
                        state = torch.hub.load_state_dict_from_url(
                            download_url, map_location="cpu", model_dir=local_model_dir
                        )
                        synchronize()
                    else:
                        synchronize()
                        state = torch.load(local_model_dir)

                    print(f"Loaded stage1 {name} HGNetV2 from URL.")

                self.load_state_dict(state)

            except (Exception, KeyboardInterrupt) as e:
                if get_rank() == 0:
                    print(f"{str(e)}")
                    logging.error(
                        RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET
                    )
                    logging.error(
                        GREEN
                        + "Please check your network connection. Or download the model manually from "
                        + RESET
                        + f"{download_url}"
                        + GREEN
                        + " to "
                        + RESET
                        + f"{local_model_dir}."
                        + RESET
                    )
                exit()
        
        self.use_sam3_features = use_sam3_features
        self.use_sam2_features = use_sam2_features
        if self.use_sam2_features or self.use_sam3_features:
            assert sam_checkpoint_path is not None, "'sam_checkpoint_path' must be provided"

            ckpt = torch.load(sam_checkpoint_path, map_location="cpu")
            if "model" in ckpt:
                ckpt = ckpt["model"]
            
            if self.use_sam2_features:
                from sam2.modeling.backbones.hieradet import Hiera
                self.sam_encoder = Hiera(
                    embed_dim=144,
                    num_heads=2,
                    stages=[2, 6, 36, 4],
                    global_att_blocks=[23, 33, 43],
                    window_pos_embed_bkg_spatial_size=[7, 7],
                    window_spec=[8, 4, 16, 8],
                )

                state_dict = {}
                for k, v in ckpt.items():
                    if k.startswith("image_encoder.trunk."):
                        state_dict[k[len("image_encoder.trunk."):]] = v

                self.sam_encoder.load_state_dict(state_dict)

            elif self.use_sam3_features:
                self.sam_encoder = _create_vit_backbone()
                new_ckpt = dict()
                for k, v in ckpt.items():
                    if "detector.backbone.vision_backbone.trunk." in k:
                        new_ckpt[k[len("detector.backbone.vision_backbone.trunk.") :]] = v
                
                self.sam_encoder.load_state_dict(new_ckpt)

            for p in self.sam_encoder.parameters():
                p.requires_grad_(False)

            if train_adapter_block:
                for param in self.sam_encoder.parameters():
                    param.requires_grad = False

                blocks = []
                for block in self.sam_encoder.blocks:
                    blocks.append(Adapter(block))
                    
                self.sam_encoder.blocks = nn.Sequential(*blocks)

            self.fuse_sam = fuse_sam

            self.sam_return_idx = list(self.return_idx)
            all_sam_channels = [144, 288, 576, 1152]

            self.sam_channels = [
                all_sam_channels[i]
                for i in self.sam_return_idx
            ]

            hg_channels = [
                self._out_channels[i]
                for i in self.return_idx
            ]

            if self.use_sam3_features:
                self.reduce1 = nn.Conv2d(1024, hg_channels[0], 1)
                self.reduce2 = nn.Conv2d(1024, hg_channels[1], 1)
                self.reduce3 = nn.Conv2d(1024, hg_channels[2], 1)
            else:
                self.sam_proj = nn.ModuleList([ 
                    nn.Conv2d(sam_c, hg_c, kernel_size=1) for sam_c, hg_c in zip(self.sam_channels, hg_channels) 
                ])

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        image = x
        x = self.stem(x)
        hg_outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                hg_outs.append(x)

        if not self.use_sam2_features and not self.use_sam3_features:
            return hg_outs
        
        if self.use_sam3_features:
            image = F.interpolate(image, size=(1008, 1008), mode="bilinear")

        sam_outs = self.sam_encoder(image)

        if self.use_sam3_features:
            sam_out = sam_outs[-1]
            x1 = F.interpolate(
                self.reduce1(sam_out),
                size=hg_outs[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            x2 = F.interpolate(
                self.reduce2(sam_out),
                size=hg_outs[1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            x3 = F.interpolate(
                self.reduce3(sam_out),
                size=hg_outs[2].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            outs = [x1 + hg_outs[0], x2 + hg_outs[1], x3 + hg_outs[2]]
        else:
            # Select SAM feature levels matching HGNet return_idx.
            sam_outs = [
                sam_outs[i]
                for i in self.sam_return_idx
            ]

            if self.fuse_sam:
                sam_outs = [
                    proj(feat)
                    for proj, feat in zip(self.sam_proj, sam_outs)
                ]

                outs = []

                for h, s in zip(hg_outs, sam_outs):
                    if h.shape[-2:] != s.shape[-2:]:
                        s = F.interpolate(
                            s,
                            size=h.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    outs.append(h + s)

        return outs
