"""Main script for step 2 (see Fig. 2) of the training pipeline.
Task-oriented generalization training.

Use this script to:
    * load a pre-trained weighted foundation model (from stage 1)
    * initialize Aree-adapter parameters
    * train adapter on a segmentation objective.
"""
import hydra
import lightning.pytorch
import omegaconf
import torch
import wandb
import src.utils
import src.trainers.segmentation

# import os
# import tempfile
# # 强制所有临时文件到 /mnt/storage
# os.environ['TMPDIR'] = '/mnt/storage/yzhang/tmp'
# os.environ['TEMP'] = '/mnt/storage/yzhang/tmp'
# os.environ['TMP'] = '/mnt/storage/yzhang/tmp'

# # 创建目录
# os.makedirs('/mnt/storage/yzhang/tmp', exist_ok=True)
# tempfile.tempdir = '/mnt/storage/yzhang/tmp'

# # PyTorch 扩展编译缓存
# os.environ['TORCH_EXTENSIONS_DIR'] = '/mnt/storage/yzhang/.cache/torch_extensions'

# # wandb 缓存
# os.environ['WANDB_CACHE_DIR'] = '/mnt/storage/yzhang/.cache/wandb'
# os.environ['WANDB_DIR'] = '/mnt/storage/yzhang/wandb'

# # HuggingFace 缓存
# os.environ['HF_HOME'] = '/mnt/storage/yzhang/.cache/huggingface'
# os.environ['TRANSFORMERS_CACHE'] = '/mnt/storage/yzhang/.cache/huggingface'

src.utils.set_resources(num_threads=0)
@hydra.main(version_base=None, config_path="configs/stage2", config_name="experiment_DFC.yaml")
def main(cfg):
    config = src.utils.Dotdict(
        omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    if "scale_mae" in config.model.name:
        assert hasattr(
            config.model, "input_res"
        ), "input_res is required for config.model=scale-mae"

    run, wandb_logger, config = src.utils.setup_wandb(config)
    src.utils.set_seed(
        config.seed
    )  # after setup_wandb in case seed is provided by wandb sweep
    datamodule, config = src.utils.get_datamodule(config)
    callbacks = src.utils.get_callbacks(run.dir)

    if config.continual_pretrain_run is not None:
        pretrain_args = src.utils.get_config_from_wandb_run(config, device=config.device)
        src.utils.assert_model_compatibility(pretrain_args, config, ignore=["model",'embed_dim'])
        # src.utils.assert_model_compatibility(pretrain_args, config, ignore=["model"])

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
        callbacks=callbacks,
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
        outpath=run.dir,
        writer_path=run.dir.replace('files','tensorboardlogs'),
        wavelength=config.model.wavelength,
        devices=config.device[0],
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    if config.continual_pretrain_run is not None:
        task.model = src.utils.load_weights_from_wandb_run(
            task.model,
            config,
            prefix='vit_backbone.',
            device=config.device
        )

    trainer = lightning.pytorch.Trainer(
        fast_dev_run=config.wandb.fast_dev_run,
        # callbacks=[checkpoint_callback, early_stopping_callback], these will be overridden by callbacks in the task
        logger=[wandb_logger],
        default_root_dir=config.wandb.experiment_dir,
        min_steps=config.optim.min_steps,
        max_steps=config.optim.max_steps,
        accelerator=accelerator,
        log_every_n_steps=1,
        devices=config.device,
        enable_checkpointing=False
        )

    config.model.params = sum([p.numel() for p in task.model.parameters()])
    config.model.trainable_params = sum(
        [p.numel() for p in task.model.parameters() if p.requires_grad]
    )
    wandb.config["params"] = config.model.params
    wandb.config["trainable_params"] = config.model.trainable_params

    if config.verbose:
        print("Trainable parameters:")
        for n, p in task.model.named_parameters():
            if p.requires_grad:
                print(n, p.shape)

    trainer.fit(
        model=task,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
        # ckpt_path='logs/stage2/SpectralX_stage2_WHUOHS/wandb/run-20251213_215716-49cqntzf/files/model_epoch_0024.ckpt'
    )

    if config.verbose:
        print(
            f"Eval performance: {trainer.test(model=task, dataloaders=datamodule.val_dataloader())}"
        )

    wandb.config["final_configs"] = src.utils.update_configs(config)


if __name__ == "__main__":
    main()
