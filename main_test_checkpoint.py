"""Main script to evaluate the performance of trained models."""

import os
import lightning.pytorch
import torch
import yaml
import src.utils
import src.trainers.segmentation

def test_run(log_path, chosen_ckpt):

    config_file = os.path.join(log_path, "files", "updated_setup_configs.yml")

    if not os.path.isfile(config_file):
        old_config = True
        config_file = os.path.join(
            log_path,
            "files",
            "config.yaml",
        )
    else:
        old_config = False

    with open(config_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    config = src.utils.Dotdict(config)
    if old_config:
        if hasattr(config, "final_configs"):
            config = config.final_configs.value
        else:
            config = config.setup_config.value
    config.log_path = log_path
    config.optim.num_workers = 0 #6
    src.utils.set_seed(config.seed)
    datamodule, config = src.utils.get_datamodule(config, flag='test')

    if not hasattr(config.model, "norm_trainable"):
        config.model.norm_trainable = False
    if not hasattr(config.optim, "head_lr"):
        config.optim.head_lr = config.optim.lr

    task = src.trainers.segmentation.SegmentationTrainer(
    segmentation_model=config.model.name,
    deepsup=config.model.deepsup,
    model=config.model.backbone,
    model_type=config.model.backbone_type,
    feature_map_indices=config.model.feature_map_indices,
    aux_loss_factor=config.optim.aux_loss_factor,
    dataname=config.data.datamodule,
    num_classes=config.data.num_classes,
    in_channels=config.data.in_chans,
    loss="ce", # focal  jaccard
    lr=config.optim.lr,
    input_size=config.data.img_size,
    patch_size=config.model.patch_size,
    token_length=config.model.token_length,
    patience=config.optim.lr_schedule_patience,
    freeze_backbone=config.model.freeze_backbone,
    pretrained=config.model.pretrained,
    callbacks=[],
    input_res=config.model.input_res,
    adapter=config.model.adapter,
    adapter_hidden_dim=config.model.adapter_hidden_dim,
    norm_trainable=config.model.norm_trainable,
    adapter_scale=config.model.adapter_scale,
    adapter_shared=config.model.adapter_shared,
    fixed_output_size=config.model.fixed_output_size,
    adapter_type=config.model.adapter_type,
    patch_embed_adapter=config.model.patch_embed_adapter,
    use_mask_token=config.model.use_mask_token,
    train_patch_embed=config.model.train_patch_embed,
    patch_embed_adapter_scale=config.model.patch_embed_adapter_scale,
    train_all_params=config.model.train_all_params,
    train_cls_mask_tokens=config.model.train_cls_mask_tokens,
    adapter_trainable=config.model.adapter_trainable,
    only_bias_trainable=config.model.only_bias_trainable,
    only_scaler_trainable=config.model.only_scaler_trainable,
    ignore_index=config.data.IGNORE_LABEL,
    wavelength=config.model.wavelength,
    )

    task.model = src.utils.load_weights_from_wandb_run(
        task.model,
        config,
        which_state=chosen_ckpt,
        device=config.device,
    )
    task.model.eval()

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = lightning.pytorch.Trainer(
        fast_dev_run=False,
        logger=None,
        accelerator=accelerator,
        devices=config.device,
    )

    datamodule.setup("test")
    test_stats = trainer.test(
        model=task,
        dataloaders=datamodule.test_dataloader(),
    )

    return test_stats

if __name__ == "__main__":

    log_path = 'logs/SpectralX_stage2_WHUOHS_ST/wandb/offline-run-20260627_201059-4u6yr2qx'
    chosen_ckpt = "best"  # 'last', or 'best'
    filename = log_path.split('/')[-1] + ".txt"
    file = f"logs/tests/{filename}"
    if not os.path.exists('logs/tests/'):
        os.mkdir('logs/tests/')

    with open(file, "a+") as f:
        test_stats = test_run(log_path, chosen_ckpt)
        print(f"{test_stats=}")

        f.write(str(test_stats) + "\n")
