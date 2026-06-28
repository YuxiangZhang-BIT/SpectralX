"""Segmentation model definitions."""

import math
import einops
import mmseg
import mmseg.models.necks
import mmseg.models.decode_heads
import torch
import torch.nn as nn
from src.adapters.Are_apt import Attribute_refinedAdapters

class UPerNetWrapper(torch.nn.Module):
    def __init__(
        self, vit_backbone, feature_map_indices, num_classes=150, deepsup=False
    ):
        """
        Upernet-style wrapper around timm vit_large_patch16_224 with neck and decode_head from mmsegmentation
        """
        super().__init__()
        self.vit_backbone = vit_backbone
        self.deepsup = deepsup
        norm_cfg = dict(type="BN", eps=1e-6)
        self.neck = mmseg.models.necks.MultiLevelNeck(
            in_channels=[1024] * 4, out_channels=1024, scales=[4, 2, 1, 0.5]
        )
        self.head = mmseg.models.decode_heads.UPerHead(
            in_channels=[1024] * 4,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            align_corners=False,
        )

        if self.deepsup:
            self.aux_head = mmseg.models.decode_heads.FCNHead(
                in_channels=1024,
                in_index=3,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=norm_cfg,
                align_corners=False,
            )
        self.feature_map_indices = feature_map_indices
        self.resize = mmseg.models.utils.resize

        assert max(feature_map_indices) <= len(
            self.vit_backbone.blocks
        ), "feature map index out of bounds"

    def forward_features(self, x):
        """return feature maps after specific transformer blocks"""
        x, _ = self.vit_backbone.patch_embed(x)
        x = self.vit_backbone._pos_embed(x)
        x = self.vit_backbone.norm_pre(x)

        features = []
        for idx, block in enumerate(self.vit_backbone.blocks):
            x = block(x)

            if idx in self.feature_map_indices:
                features.append(x.clone())

        return features

    def forward_cd(self, x):
        assert x.shape[1] % 2 == 0
        mid = x.shape[1] // 2
        x1 = x[:, :mid]
        x2 = x[:, mid:]
        feat1 = self.forward_features(x1)
        feat2 = self.forward_features(x2)

        features = [torch.abs(f1 - f2) for f1, f2 in zip(feat1, feat2)]

        h = w = int(math.sqrt(features[0].shape[1] - 1))
        features = [
            einops.rearrange(z[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w)
            for z in features
        ]
        features = self.neck(features)

        # `deepsup` prediction from intermediate feature map
        if self.deepsup:
            aux_map = self.aux_head(features)
            aux_map = self.resize(
                aux_map,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                warning=False,
            )

        # output feature map from extracted all feature maps
        output_map = self.head(features)
        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            warning=False,
        )

        if self.deepsup:
            return output_map, aux_map

        return output_map

    def forward(self, x):
        features = self.forward_features(x)

        # remove cls token reshape into maps
        h = w = int(math.sqrt(features[0].shape[1] - 1))
        features = [
            einops.rearrange(z[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w)
            for z in features
        ] 
        features = self.neck(features)

        # `deepsup` prediction from intermediate feature map
        if self.deepsup:
            aux_map = self.aux_head(features)
            aux_map = self.resize(
                aux_map,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                warning=False,
            )

        # output feature map from extracted all feature maps
        output_map = self.head(features) # torch.Size([2, 23, 56, 56])
        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            warning=False,
        )

        if self.deepsup:
            return output_map, aux_map

        return output_map

class UPerNetWrapper_DINOv3(torch.nn.Module):
    def __init__(
        self, vit_backbone, feature_map_indices, num_classes=150, deepsup=False
    ):
        """
        Upernet-style wrapper around dinov3 with neck and decode_head from mmsegmentation
        """
        super().__init__()
        self.vit_backbone = vit_backbone
        self.deepsup = deepsup
        norm_cfg = dict(type="BN", eps=1e-6)
        self.neck = mmseg.models.necks.MultiLevelNeck(
            in_channels=[1024] * 4, out_channels=1024, scales=[4, 2, 1, 0.5]
        )
        self.head = mmseg.models.decode_heads.UPerHead(
            in_channels=[1024] * 4,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            align_corners=False,
        )

        if self.deepsup:
            self.aux_head = mmseg.models.decode_heads.FCNHead(
                in_channels=1024,
                in_index=3,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=norm_cfg,
                align_corners=False,
            )
        self.feature_map_indices = feature_map_indices
        self.resize = mmseg.models.utils.resize

        assert max(feature_map_indices) <= len(
            self.vit_backbone.blocks
        ), "feature map index out of bounds"

    def forward_features(self, x):
        """return feature maps after specific transformer blocks"""
        output = self.vit_backbone(x, None, self.feature_map_indices)
        return output['features']
    
    def forward_cd(self, x):
        assert x.shape[1] % 2 == 0
        mid = x.shape[1] // 2
        x1 = x[:, :mid]
        x2 = x[:, mid:]
        feat1 = self.forward_features(x1)
        feat2 = self.forward_features(x2)

        features = [torch.abs(f1 - f2) for f1, f2 in zip(feat1, feat2)]

        h = w = int(math.sqrt(features[0].shape[1] - 1))
        features = [
            einops.rearrange(z[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w)
            for z in features
        ]
        features = self.neck(features)

        # `deepsup` prediction from intermediate feature map
        if self.deepsup:
            aux_map = self.aux_head(features)
            aux_map = self.resize(
                aux_map,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                warning=False,
            )

        # output feature map from extracted all feature maps
        output_map = self.head(features)
        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            warning=False,
        )

        if self.deepsup:
            return output_map, aux_map

        return output_map

    def forward(self, x):
        features = self.forward_features(x)

        # remove cls token reshape into maps
        h = w = int(math.sqrt(features[0].shape[1]))
        features = [
            einops.rearrange(z, "b (h w) d -> b d h w", h=h, w=w)
            for z in features
        ] 
        features = self.neck(features)

        # `deepsup` prediction from intermediate feature map
        if self.deepsup:
            aux_map = self.aux_head(features)
            aux_map = self.resize(
                aux_map,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                warning=False,
            )

        # output feature map from extracted all feature maps
        output_map = self.head(features)
        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            warning=False,
        )

        if self.deepsup:
            return output_map, aux_map

        return output_map
    
class UPerNetWrapperSpectralX(torch.nn.Module):
    def __init__(
        self, vit_backbone, feature_map_indices, num_classes=150, deepsup=False
    ):
        """
        Upernet-style wrapper around timm vit_large_patch16_224 with neck and decode_head from mmsegmentation
        """
        super().__init__()
        self.vit_backbone = vit_backbone
        self.deepsup = deepsup
        self.num_classes = num_classes
        norm_cfg = dict(type="BN", eps=1e-6)
        self.neck = mmseg.models.necks.MultiLevelNeck(
            in_channels=[1024] * 4, out_channels=1024, scales=[4, 2, 1, 0.5]
        )
        self.head = mmseg.models.decode_heads.UPerHead(
            in_channels=[1024] * 4,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            align_corners=False,
        )

        if self.deepsup:
            self.aux_head = mmseg.models.decode_heads.FCNHead(
                in_channels=512,
                in_index=3,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=norm_cfg,
                align_corners=False,
            )
        self.feature_map_indices = feature_map_indices
        self.resize = mmseg.models.utils.resize

        assert max(feature_map_indices) <= len(
            self.vit_backbone.blocks
        ), "feature map index out of bounds"
        
        self.AreApt = Attribute_refinedAdapters()
        for param in self.AreApt.parameters():
            param.requires_grad = True

        self.norm = nn.LayerNorm(1024) # , eps=1e+3
    
    def forward_features(self, x):
        """return feature maps after specific transformer blocks"""
        x, conv_feats = self.vit_backbone.patch_embed(x, flag='conv_feats')
        x = self.vit_backbone._pos_embed(x)
        x = self.vit_backbone.norm_pre(x)

        features = []
        for idx, block in enumerate(self.vit_backbone.blocks):
            x = block(x)
            if idx in self.feature_map_indices:
                x, conv_feats = self.AreApt.forward(
                    x,
                    idx,
                    conv_feats,
                    batch_first=True,
                    has_cls_token=True,
                )
                features.append(x.clone())
        return features

    def forward_cd(self, x):
        assert x.shape[1] % 2 == 0
        mid = x.shape[1] // 2
        x1 = x[:, :mid]
        x2 = x[:, mid:]
        feat1 = self.forward_features(x1)
        feat2 = self.forward_features(x2)

        features = [torch.abs(f1 - f2) for f1, f2 in zip(feat1, feat2)]

        h = w = int(math.sqrt(features[0].shape[1] - 1))
        features = [
            einops.rearrange(z[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w)
            for z in features
        ]
        features = self.neck(features)

        # `deepsup` prediction from intermediate feature map
        if self.deepsup:
            aux_map = self.aux_head(features)
            aux_map = self.resize(
                aux_map,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                warning=False,
            )

        # output feature map from extracted all feature maps
        output_map = self.head(features)
        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            warning=False,
        )

        if self.deepsup:
            return output_map, aux_map

        return output_map

    def forward(self, x):
        features = self.forward_features(x)
        # remove cls token reshape into maps
        h = w = int(math.sqrt(features[0].shape[1] - 1))
        features = [
            einops.rearrange(z[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w)
            for z in features
        ] 
        features = self.neck(features)

        # `deepsup` prediction from intermediate feature map
        if self.deepsup:
            aux_map = self.aux_head(features)
            aux_map = self.resize(
                aux_map,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                warning=False,
            )

        # output feature map from extracted all feature maps
        output_map = self.head(features) # torch.Size([2, 23, 56, 56])
        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear", # bicubic  bilinear
            align_corners=False,
            warning=False,
        )

        if self.deepsup:
            return output_map, aux_map

        return output_map
    
class UPerNetWrapperSpectralX_DINOv3(torch.nn.Module):
    def __init__(
        self, vit_backbone, feature_map_indices, num_classes=150, deepsup=False
    ):
        """
        Upernet-style wrapper around dinov3 with neck and decode_head from mmsegmentation
        """
        super().__init__()
        self.vit_backbone = vit_backbone
        self.deepsup = deepsup
        self.num_classes = num_classes
        norm_cfg = dict(type="BN", eps=1e-6)
        self.neck = mmseg.models.necks.MultiLevelNeck(
            in_channels=[1024] * 4, out_channels=1024, scales=[4, 2, 1, 0.5]
        )
        self.head = mmseg.models.decode_heads.UPerHead(
            in_channels=[1024] * 4,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            align_corners=False,
        )

        if self.deepsup:
            self.aux_head = mmseg.models.decode_heads.FCNHead(
                in_channels=512,
                in_index=3,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=norm_cfg,
                align_corners=False,
            )
        self.feature_map_indices = feature_map_indices
        self.resize = mmseg.models.utils.resize

        assert max(feature_map_indices) <= len(
            self.vit_backbone.blocks
        ), "feature map index out of bounds"

        self.AreApt = Attribute_refinedAdapters()
        for param in self.AreApt.parameters():
            param.requires_grad = True

        self.norm = nn.LayerNorm(1024)
    
    def forward_features(self, x):
        """return feature maps after specific transformer blocks"""
        output = self.vit_backbone(x, None, self.AreApt, self.feature_map_indices)
        return output['features']

    def forward_cd(self, x):
        assert x.shape[1] % 2 == 0
        mid = x.shape[1] // 2
        x1 = x[:, :mid]
        x2 = x[:, mid:]
        feat1 = self.forward_features(x1)
        feat2 = self.forward_features(x2)

        features = [torch.abs(f1 - f2) for f1, f2 in zip(feat1, feat2)]

        h = w = int(math.sqrt(features[0].shape[1] - 1))
        features = [
            einops.rearrange(z[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w)
            for z in features
        ]
        features = self.neck(features)

        # `deepsup` prediction from intermediate feature map
        if self.deepsup:
            aux_map = self.aux_head(features)
            aux_map = self.resize(
                aux_map,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                warning=False,
            )

        # output feature map from extracted all feature maps
        output_map = self.head(features)
        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            warning=False,
        )

        if self.deepsup:
            return output_map, aux_map

        return output_map

    def forward(self, x):
        features = self.forward_features(x)
        # remove cls token reshape into maps
        h = w = int(math.sqrt(features[0].shape[1]))
        features = [
            einops.rearrange(z, "b (h w) d -> b d h w", h=h, w=w)
            for z in features
        ] 
        features = self.neck(features)

        # `deepsup` prediction from intermediate feature map
        if self.deepsup:
            aux_map = self.aux_head(features)
            aux_map = self.resize(
                aux_map,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                warning=False,
            )

        # output feature map from extracted all feature maps
        output_map = self.head(features)
        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear", # bicubic  bilinear
            align_corners=False,
            warning=False,
        )

        if self.deepsup:
            return output_map, aux_map

        return output_map

class ViTWithFCNHead(torch.nn.Module):
    """ViT with fully-convolutional head for dense predictions."""

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.deepsup = False
        norm_cfg = dict(type="BN", eps=1e-6)
        self.head = mmseg.models.decode_heads.FCNHead(
            in_channels=1024,
            in_index=0,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
        )
        self.resize = mmseg.models.utils.resize

    def forward_features(self, x, feature_map_indices=[5, 11, 17, 23]):
        """return feature maps after specific transformer blocks"""
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)

        features = []
        for idx, block in enumerate(self.backbone.blocks):
            x = block(x)

            if idx in feature_map_indices:
                features.append(x.clone())

        if len(features) == 1:
            return features[0]
        return features

    def forward(self, x):
        # features = self.backbone.forward_features(x)
        features = self.forward_features(x)
        if isinstance(features, list):
            h = w = int(math.sqrt(features[0].shape[1] - 1))
            features = [
                einops.rearrange(f[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w)
                for f in features
            ]
            # note: this belongs below at output_map
            # features = [
            #     self.resize(
            #         f,
            #         size=(224, 224),
            #         mode="bilinear",
            #         align_corners=False,
            #         warning=False,
            #     )
            #     for f in features
            # ]
        else:
            h = w = int(math.sqrt(features.shape[1] - 1))
            features = einops.rearrange(
                features[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w
            )
            features = [features]

        output_map = self.head(features)

        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            warning=False,
        )

        return output_map

    def forward_cd(self, x):
        # x contains two images at different points in time
        # concatenated along the channel axis
        assert x.shape[1] % 2 == 0
        mid = x.shape[1] // 2
        x1 = x[:, :mid]
        x2 = x[:, mid:]

        feat1 = self.forward_features(x1)
        feat2 = self.forward_features(x2)

        if isinstance(feat1, list):
            feat = [torch.abs(f1 - f2) for f1, f2 in zip(feat1, feat2)]
            h = w = int(math.sqrt(feat[0].shape[1] - 1))
        else:
            feat = feat2 - feat1
            h = w = int(math.sqrt(feat.shape[1] - 1))

        if not isinstance(feat, list):
            feat = [feat]
        feat = [
            einops.rearrange(f[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w)
            for f in feat
        ]

        output_map = self.head(feat)
        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            warning=False,
        )

        return output_map