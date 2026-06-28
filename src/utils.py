"""Collection of utility methods for model training and evaluation."""

import copy
import os
import random
from typing import Callable, Optional, Tuple
import glob
import kornia
import lightning.pytorch.callbacks
import lightning.pytorch.loggers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeo.transforms
import wandb
import yaml
import einops_exts
import einops
import src.datamodules
from functools import partial
from timm.layers import to_2tuple
from timm.models.vision_transformer_sam import (DropPath, LayerScale,
                                                window_partition,
                                                window_unpartition)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class SAM_Block(nn.Module):

    def __init__(
            self,
            dim,
            out_dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            window_size=0,
            spe=False
    ):
        super().__init__()
        self.window_size = window_size
        self.spe = spe
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            spe=spe
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            out_features=out_dim,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                    
    def forward(self, x):
        B, H, W, _ = x.shape
        if self.spe:
            x = x.reshape(B, H * W, -1).permute(0,2,1)
        shortcut = x
        x = self.norm1(x)
        # Window partition
        pad_hw: Optional[Tuple[int, int]] = None
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)

        x = self.drop_path1(self.ls1(self.attn(x)))

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, (H, W), pad_hw)

        x = shortcut + x

        if self.spe:
            # x = x.reshape(B, H * W, -1).permute(0,2,1)
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            x = x.reshape(B, -1, H, W).permute(0,2,3,1)
        else:
            x = x.reshape(B, H * W, -1)  # MLP is faster for N, L, C tensor
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            x = x.reshape(B, H, W, -1)

        return x


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            spe=False
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.spe = spe
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        if self.spe:
            B, N, _ = x.shape
        else:
            B, H, W, _ = x.shape
            N = H * W
        x = x.reshape(B, N, -1)
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv with shape (3, B, nHead, H * W, C)
        q, k, v = qkv.unbind(0)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k = self.q_norm(q), self.k_norm(k)
        
        x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        if not self.spe:
            x = x.view(B, H, W, -1)
        return x

    
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False)
    )


class PerceiverAttention(nn.Module):
    def __init__(
            self,
            input_dim,
            out_dim,
            dim_head=64,
            heads=8
    ):
        super().__init__()

        self.input_dim = input_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(input_dim)
        self.norm_latents = nn.LayerNorm(out_dim)

        self.to_q = nn.Linear(out_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(input_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, out_dim, bias=False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = einops_exts.rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        # attention
        sim = torch.einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()        
        attn = sim.softmax(dim=-1)

        out = torch.einsum('... i j, ... j d -> ... i d', attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out), attn


class DownsampleConv(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU()
        )
        
        self.module2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0),
            # nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.BatchNorm2d(128),
        )

        self.module3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        # Module 4 - Reduce spatial size to (56, 56)
        self.module4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0), 
            # nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.BatchNorm2d(256),
        )
        
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=out_dim, kernel_size=3, stride=2, padding=1) 


    def forward(self, x):
        x = self.conv1(x)
        x = x + self.module1(x)
        x = self.module2(x)
        x = x + self.module3(x)
        x = self.module4(x)
        x = self.conv2(x)
        return x
    
        
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, with_cls_token=True):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if with_cls_token else None
        
    def forward(self, inputs):
        x = self.proj(inputs)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
        return x
    
def update_configs(config: dict) -> dict:
    """Creates a new config dict without Dotdict entries.

    Args:
        config: a dict that might contain Dotdict type entries.

    Returns:
        a new dictionary where Dotdicts objects are resolved into dicts.

    """

    updated_configs = {}
    for k, v in config.__dict__.items():
        if isinstance(v, Dotdict):
            updated_configs[k] = v.__dict__
        else:
            updated_configs[k] = v

    return updated_configs


def set_resources(num_threads: int, wand_cache_dir: str = None):
    """Sets environment variables to control resource usage.

    The environment variables control the number of used threads
    for different vector op libraries and GDAL. The cache dir controls
    where wandb cache is stored locally.

    Args.
        num_threads: the max number of threads.
        wand_cache_dir: path to the desired cache dir

    """

    num_threads = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    os.environ["NUMEXPR_NUM_THREADS"] = num_threads
    os.environ["GDAL_NUM_THREADS"] = num_threads

    if wand_cache_dir:
        os.environ["WANDB_CACHE_DIR"] = wand_cache_dir


def set_seed(seed: int):
    """Set the seed across multiple libraries.

    Sets seed for builtin random, numpy, and torch libraries.

    Args:
        seed: the seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Dotdict:
    """Wraps dictionaries to allow value access in dot notation.

    Instead of data[key], access value as data.key"""

    def __init__(self, data: dict):
        super().__init__()
        for k, v in data.items():
            if isinstance(v, dict):
                # take care of nested dicts
                v = Dotdict(v)
            self.__dict__[k] = v


def setup_wandb(
    config: Dotdict,
) -> Tuple[wandb.run, lightning.pytorch.loggers.WandbLogger, Dotdict]:
    """Sets up wandb logging for a training run.

    This will run wandb.init with arguments based on the config and store
    the used config on disk.

    Args:
        config: config of the training run.

    Returns:
        A tuple consisting of the wandb run, the lightning logger, and the updated config.

    Side-effects:
        Stores the config used to initialize the wandb run to the run directory.
    """

    os.environ["WANDB_CACHE_DIR"] = config.wandb.cache_dir

    if not os.path.exists(config.wandb.experiment_dir):
        os.makedirs(config.wandb.experiment_dir)

    run = wandb.init(
        mode=config.wandb.mode,
        # entity=config.wandb.entity,
        project=config.wandb.project,
        dir=config.wandb.experiment_dir,
    )
    wandb_logger = lightning.pytorch.loggers.WandbLogger(
        log_model=config.wandb.log_model,
        config=config,
        experiment=run,
        dir=run.dir,
    )

    config.__dict__.update(
        wandb.config
    )  # when using a wandb sweep, the wandb agent might update some params

    # upload up-to-date config to wandb
    wandb.config["setup_config"] = update_configs(config)

    if config.verbose:
        print(run.dir)
    with open(os.path.join(run.dir, "updated_setup_configs.yml"), "w") as outfile:
        yaml.dump(wandb.config["setup_config"], outfile, default_flow_style=False)

    return run, wandb_logger, config


def get_datamodule(
    config: Dotdict,
    flag='train',
) -> Tuple[lightning.pytorch.LightningDataModule, Dotdict]:
    """Creates the lightning datamodule for the dataset defined in the config.

    Args:
        config: the training run config.

    Returns:
        a tuple of the lightning datamodule for a dataset and the latest config.
    """

    # get the correct datamodule and dataset objects
    datamodule = src.datamodules.__dict__[config.data.datamodule]
    dataset = src.datasets.__dict__[config.data.datamodule.replace("DataModule", "")]
    if any(keyword in config.data.datamodule.lower() for keyword in ('whuohs', 'dfc', 'mts12')):
        datamodule = datamodule(
            root=config.data.root,
            batch_size=config.optim.batch_size,
            num_workers=config.optim.num_workers,
            cfg=config,
        )
    else:
        # scale images to expected size and standardize
        if (
            "benge" in config.data.datamodule.lower()
            or "treesatai" in config.data.datamodule.lower()
        ):
            if config.data.modality == "s1":
                datamodule.mean = datamodule.s1_mean
                datamodule.std = datamodule.s1_std
            elif config.data.modality == "s2":
                datamodule.mean = datamodule.s2_mean
                datamodule.std = datamodule.s2_std
            elif config.data.modality == "aerial":
                datamodule.mean = datamodule.aerial_mean
                datamodule.std = datamodule.aerial_std
            else:
                raise AttributeError()

        # ensure bands are correctly selected for multi-spectral data
        band_idx = range(len(datamodule.mean))
        if "eurosat" in config.data.datamodule.lower():
            band_idx = []
            for b in config.data.bands:
                band_idx.append(dataset.all_band_names.index(b))
            if config.verbose:
                print(f"Band indices: {band_idx=}")

        data_keys = ["image"]
        if config.task == "segmentation":
            data_keys.append("mask")

        if config.verbose:
            print(f"Augmentation keys: {data_keys=}")

        additional_transforms = torchgeo.transforms.AugmentationSequential(
            kornia.augmentation.Normalize(
                mean=datamodule.mean[band_idx],
                std=datamodule.std[band_idx],
                keepdim=True,
            ),
            kornia.augmentation.Resize(
                (config.data.img_size, config.data.img_size), keepdim=True
            ),
            data_keys=data_keys,
        )

        root = config.data.root
        if config.verbose:
            print(f"Dataset root directory: {root=}")

        # handle directory structure of different datasets
        if hasattr(src.datamodules.__dict__[config.data.datamodule], "folder"):
            root = f"data/{src.datamodules.__dict__[config.data.datamodule].folder}"

        # initialze few-shot datamodule with limited number of samples
        if config.data.few_shot_k is not None:
            if "eurosat" in config.data.datamodule.lower():
                datamodule = datamodule(
                    root=root,
                    bands=config.data.bands,
                    batch_size=config.optim.batch_size,
                    num_workers=config.optim.num_workers,
                    train_split_file_suffix=f"-k{config.data.few_shot_k}-seed{config.data.few_shot_seed}.txt",
                    transforms=additional_transforms,
                )
            elif "treesatai" in config.data.datamodule.lower():
                # no few-shot split defined yet for TreeSatAI
                raise NotImplementedError()
            else:
                datamodule = datamodule(
                    root=root,
                    batch_size=config.optim.batch_size,
                    num_workers=config.optim.num_workers,
                    train_split_file_suffix=f"-k{config.data.few_shot_k}-seed{config.data.few_shot_seed}.txt",
                    transforms=additional_transforms,
                )
        else:
            # initialze the full dataset
            if "eurosat" in config.data.datamodule.lower():
                datamodule = datamodule(
                    root=root,
                    bands=config.data.bands,
                    batch_size=config.optim.batch_size,
                    num_workers=config.optim.num_workers,
                    transforms=additional_transforms,
                )
            elif "benge" in config.data.datamodule.lower():
                datamodule = datamodule(
                    root=root,
                    modality=config.data.modality,
                    bands=config.data.bands,
                    batch_size=config.optim.batch_size,
                    num_workers=config.optim.num_workers,
                    transforms=additional_transforms,
                )
            elif "treesatai" in config.data.datamodule.lower():
                datamodule = datamodule(
                    root=root,
                    modality=config.data.modality,
                    bands=config.data.bands,
                    batch_size=config.optim.batch_size,
                    num_workers=config.optim.num_workers,
                    transforms=additional_transforms,
                    size=config.data.size,
                )
            elif 'yerb' in config.data.datamodule.lower():
                datamodule = datamodule(
                    root=root,
                    batch_size=config.optim.batch_size,
                    num_workers=config.optim.num_workers,
                    transforms=additional_transforms,
                )
            else:
                datamodule = datamodule(
                    root=root,
                    batch_size=config.optim.batch_size,
                    num_workers=config.optim.num_workers,
                    transforms=additional_transforms,
                )

    datamodule.setup("fit")
    if 'hyperx' in config.data.datamodule.lower():
        if flag == 'test':
            config.data.num_classes = datamodule.test_dataset.classes
            config.data.in_chans = datamodule.test_dataset[0]["image"].shape[0]
        else:
            config.data.num_classes = datamodule.train_dataset.classes
            config.data.in_chans = datamodule.train_dataset[0]["image"].shape[0]
    else:
        if flag == 'test':
            config.data.num_classes = len(datamodule.val_dataset.classes)
            config.data.in_chans = datamodule.val_dataset[0]["image"].shape[0]
            # config.data.num_classes = len(datamodule.test_dataset.classes)
            # config.data.in_chans = datamodule.test_dataset[0]["image"].shape[0]
        else:
            config.data.num_classes = len(datamodule.train_dataset.classes)
            config.data.in_chans = datamodule.train_dataset[0]["image"].shape[0]

    return datamodule, config


def get_callbacks(
    dir: str,
    flag=None,
) -> Tuple[
    lightning.pytorch.callbacks.ModelCheckpoint,
    lightning.pytorch.callbacks.EarlyStopping,
    lightning.pytorch.callbacks.LearningRateMonitor,
]:
    """Initialze lightning callbacks for checkpointing, early stopping and LR monitoring.

    Args:
        dir: a directory where model checkpoints will be stored.

    Returns:
        a tuple of the three callback objects.
    """
    if flag == 'mae':
        checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
            monitor="val_loss",
            # dirpath=args.experiment_dir,
            dirpath=dir,
            filename="best-checkpoint",
            # filename="last",
            save_top_k=1,  # save best
            save_last=True,
        )
        early_stopping_callback = lightning.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=10,
        )
    else:
        checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
            monitor="val_mIoU",
            mode="max",
            # monitor="train_mIoU",
            # dirpath=args.experiment_dir,
            dirpath=dir,
            filename="best-checkpoint",
            # filename="last",
            save_top_k=1,  # save best
            save_last=True,
        )
        # early_stopping_callback = lightning.pytorch.callbacks.EarlyStopping(
        #     monitor="val_mIoU",
        #     mode="max",
        #     # monitor="val_loss",
        #     min_delta=0.00,
        #     patience=10,
        # )

    lr_monitor = lightning.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step"
    )
    return checkpoint_callback, lr_monitor
    # return checkpoint_callback, early_stopping_callback, lr_monitor


def get_ckpt_path_from_wandb_run(
    config: Dotdict, state: str = "best"
):
    """Returns the path to a model checkpoint associated with a wandb run.

    Args:
        config: config of the wandb/training run.
        state: best or latest checkpoint.

    Returns:
        path to the checkpoint
    """
    ckpt = "best-checkpoint.ckpt" if state == "best" else "last.ckpt"
    log_path = getattr(config, 'log_path', None)
    pretrain_run = getattr(config, 'continual_pretrain_run', None)

    if log_path:
        best_ckpt = os.path.join(config.log_path, "files", ckpt)
    elif pretrain_run:
        best_ckpt = os.path.join(config.continual_pretrain_run, "files", ckpt)
    else:
        raise ValueError("Either log_path or continual_pretrain_run must be specified")

    return best_ckpt


def assert_model_compatibility(
    pretrain_config: Dotdict, downstream_config: Dotdict, ignore: list = []
):
    """Performs some checks to ensure pre-training run and downstream tasks are compatible.

    Args:
        pretrain_config: config of the pretraining run.
        downstream_config: config of the downstream task.
        ignore: list of checks to be skipped.

    Returns:
        True if all checks passed.

    Raises:
        AssertionError if any check fails.
    """

    if not "model" in ignore:
        assert (
            pretrain_config["model"] == downstream_config.model.name
        ), f"{pretrain_config['model']=}, {downstream_config.model.name=}"
    assert pretrain_config["in_channels"] == downstream_config.data.in_chans
    if not "embed_dim" in ignore:
        assert pretrain_config["embed_dim"] == downstream_config.model.embed_dim
    assert pretrain_config["input_size"] == downstream_config.data.img_size
    assert pretrain_config["patch_size"] == downstream_config.model.patch_size
    assert pretrain_config["adapter"] == downstream_config.model.adapter
    assert (
        pretrain_config["adapter_type"] == downstream_config.model.adapter_type
    ), f"{pretrain_config['adapter_type']=}, {downstream_config.model.adapter_type=}"
    assert pretrain_config["adapter_shared"] == downstream_config.model.adapter_shared
    assert pretrain_config["adapter_scale"] == downstream_config.model.adapter_scale
    assert (
        pretrain_config["adapter_hidden_dim"]
        == downstream_config.model.adapter_hidden_dim
    ), f"{pretrain_config['adapter_hidden_dim']=}, {downstream_config.model.adapter_hidden_dim=}"
    assert (
        pretrain_config["patch_embed_adapter"]
        == downstream_config.model.patch_embed_adapter
    ), f"{pretrain_config['patch_embed_adapter']=}, {downstream_config.model.patch_embed_adapter=}"
    assert (
        pretrain_config["patch_embed_adapter_scale"]
        == downstream_config.model.patch_embed_adapter_scale
    )

    return True


def get_config_from_wandb_run(
    config: Dotdict,
    return_ckpt_path: bool = False,
    device=None
) -> Dotdict:
    """Get the config associated with a finished wandb run.

    Args:
        config: the config for the run of interest.
        return_ckpt_path: if the path to the checkout should also be returned.

    Returns:
        the config, or a tuple of config and checkpoint path.
    """

    ckpt_path = get_ckpt_path_from_wandb_run(config)
    ckpt = torch.load(ckpt_path, map_location=f'cuda:{device[0]}')

    args = copy.deepcopy(ckpt["hyper_parameters"])
    del ckpt

    if return_ckpt_path:
        return args, ckpt_path
    return args


def load_weights_from_wandb_run(
    model: torch.nn.Module,
    config: Dotdict,
    prefix: str = None,
    return_ckpt: bool = False,
    which_state: str = "best",
    device=None
):
    """Load weights from a finished wandb run into a model object.

    Args:
        model: the torch model.
        config: the config of the finished run.
        prefix: prefix in model layer names that wasn't present in the pre-training run.
        return_ckpt: if the checkpoint is returned as well.
        which_state: best or latest checkpoint will be used.

    Returns:
        the model initialzed with weights from ´config´, or a tuple with the checkpoint.
    """

    best_ckpt = get_ckpt_path_from_wandb_run(
        config,
        state=which_state,
    )
    print(f"Loading checkpoint {best_ckpt=}...")

    # ckpt = torch.load(best_ckpt)
    ckpt = torch.load(best_ckpt, map_location=f'cuda:{device[0]}')
    state = ckpt["state_dict"]

    # remove prefix from state dict keys
    for k in list(state.keys()):
        state[k.replace("model.", "")] = state[k]
        del state[k]

    if "cls_token" in model.state_dict() and not "cls_token" in state.keys():
        state["cls_token"] = model.state_dict()["cls_token"]

    if "pos_embed" in model.state_dict() and not "pos_embed" in state.keys():
        state["pos_embed"] = model.state_dict()["pos_embed"]


    if prefix is not None:
        for k in list(state.keys()):
            state[prefix + k] = state[k]
            del state[k]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Missing weights in pre-training: {missing=}")
    print(
        f"Unexpected weights (except decoder): {[k for k in unexpected if not 'decoder' in k]}"
    )

    if return_ckpt:
        return model, ckpt

    return model