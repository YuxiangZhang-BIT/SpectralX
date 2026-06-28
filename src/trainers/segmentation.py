"""Trainer for image segmentation."""

from typing import Any, Optional
import torch
import numpy as np
import PIL
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassJaccardIndex,
    MulticlassF1Score,
)
import torchgeo
import torchgeo.trainers

import src.models
import src.models_segmentation
import src.utils
import os
src.utils.set_resources(num_threads=4)

class SegmentationTrainer(torchgeo.trainers.SemanticSegmentationTask):
    def __init__(
        self,
        segmentation_model,
        deepsup,
        model,
        model_type="",
        weights=None,
        feature_map_indices=(5, 11, 17, 23),
        aux_loss_factor=0.5,
        input_size=224,
        patch_size=16,
        token_length=100,
        in_channels: int = 3,
        dataname='DFC',
        num_classes: int = 1000,
        num_filters: int = 3,
        loss: str = "ce",
        pretrained=True,
        input_res=10,
        adapter=False,
        adapter_trainable=True,
        adapter_shared=False,
        adapter_scale=1.0,
        adapter_type="lora",
        adapter_hidden_dim=16,
        norm_trainable=True,
        fixed_output_size=0,
        use_mask_token=False,
        train_patch_embed=False,
        patch_embed_adapter=False,
        patch_embed_adapter_scale=1.0,
        train_all_params=False,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        lr: float = 1e-3,
        patience: int = 10,
        train_cls_mask_tokens=False,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        callbacks=None,
        only_scaler_trainable=False,
        only_bias_trainable=False,
        only_proj_trainable=False,
        sub_backbone=None,
        outpath=None,
        writer_path=None,
        wavelength=None,
        devices=None,
    ) -> None:
        super().__init__()
        self.outpath = outpath
        self.ignore_index = ignore_index
        self.log_file = os.path.join(outpath, 'log.txt') if outpath is not None else ''
        self.confusionmat = torch.zeros((num_classes,) * 2)
        self.num_classes = num_classes
        # self.writer= SummaryWriter(writer_path)
    def configure_callbacks(self):
        return self.hparams["callbacks"]  # self.callbacks

    def configure_models(self):
        backbone = src.models.get_model(**self.hparams)

        # add segmentation head
        if self.hparams["segmentation_model"] == "fcn":
            self.model = src.models_segmentation.ViTWithFCNHead(
                backbone,
                num_classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "upernet":
            self.model = src.models_segmentation.UPerNetWrapper(
                backbone,
                self.hparams["feature_map_indices"],
                num_classes=self.hparams["num_classes"],
                deepsup=self.hparams["deepsup"],
            )
        elif self.hparams["segmentation_model"] == "upernet-DINOv3":
            self.model = src.models_segmentation.UPerNetWrapper_DINOv3(
                backbone,
                self.hparams["feature_map_indices"],
                num_classes=self.hparams["num_classes"],
                deepsup=self.hparams["deepsup"],
            )
        elif self.hparams["segmentation_model"] == "upernetSpectralX":
            self.model = src.models_segmentation.UPerNetWrapperSpectralX(
                backbone,
                self.hparams["feature_map_indices"],
                num_classes=self.hparams["num_classes"],
                deepsup=self.hparams["deepsup"],
            )
        elif self.hparams["segmentation_model"] == "upernetSpectralX-DINOv3":
            self.model = src.models_segmentation.UPerNetWrapperSpectralX_DINOv3(
                backbone,
                self.hparams["feature_map_indices"],
                num_classes=self.hparams["num_classes"],
                deepsup=self.hparams["deepsup"],
            )
        else:
            raise NotImplementedError(
                f"`model` must be in [fcn, upernet], not {self.hparams['model']}"
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams["lr"], betas=(0.9, 0.95), weight_decay=1e-4
        )
        return {
            "optimizer": optimizer
        }
    
    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"]
        ignore_index: Optional[int] = self.hparams["ignore_index"]

        metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index, average="macro"
                ),
                MulticlassF1Score(num_classes=num_classes, ignore_index=ignore_index, average="macro")
            ]
        )
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=None)
        self.test_iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=None)
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.train_aux_metrics = metrics.clone(prefix="train_aux_")
        self.val_aux_metrics = metrics.clone(prefix="val_aux_")
        self.test_aux_metrics = metrics.clone(prefix="test_aux_")
    
    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        
        x = batch["image"]
        y = batch["mask"]

        if self.hparams['deepsup']:
            y_hat, y_aux = self(x)
            y_aux_hard = y_aux.argmax(dim=1)
            loss_aux = self.criterion(y_aux.squeeze(), y.squeeze())
            self.log("train_aux_loss", loss_aux)
            self.train_aux_metrics(y_aux_hard.squeeze(), y.squeeze())
            self.log_dict(self.train_aux_metrics)
        else:
            y_hat = self(x)

        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.log("train_loss", loss)
        self.log_dict(self.train_metrics)
        self.train_metrics.update(y_hat_hard.squeeze(), y.squeeze())

        if self.hparams['deepsup']:
            loss = loss + self.hparams["aux_loss_factor"] * loss_aux
        return loss

    def on_train_epoch_end(self):
        value = self.train_metrics.compute()
        OA_value = list(value.values())[0]
        mIoU_value = list(value.values())[1]
        mF1_value = list(value.values())[2]
        self.log('train_mIoU', mIoU_value, on_epoch=True, prog_bar=True)
        self.log('train_OA', OA_value, on_epoch=True, prog_bar=True)
        self.log('train_mF1', mF1_value, on_epoch=True, prog_bar=True)

        self.train_metrics.reset()


    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]

        if self.hparams['deepsup']:
            y_hat, y_aux = self(x)
            y_aux_hard = y_aux.argmax(dim=1)
            loss_aux = self.criterion(y_aux.squeeze(), y.squeeze())
            self.log("val_aux_loss", loss_aux)
            self.val_aux_metrics(y_aux_hard.squeeze(), y.squeeze())
            self.log_dict(self.val_aux_metrics)
        else:
            y_hat = self(x)
            
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat, y.squeeze(1))
        self.log("val_loss", loss)
        self.log_dict(self.val_metrics)
        self.val_metrics.update(y_hat_hard.squeeze(), y.squeeze())
        self.val_iou.update(y_hat_hard.squeeze(), y.squeeze())

        mask = (y.squeeze(1) != -1)
        label = self.num_classes * y.squeeze(1)[mask] + y_hat_hard[mask]
        count_conf = torch.bincount(label, minlength=self.num_classes ** 2)
        confusionmat_tmp = count_conf.reshape(self.num_classes, self.num_classes)
        self.confusionmat = self.confusionmat.to(y.device)
        self.confusionmat += confusionmat_tmp

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch.
        """
        value = self.val_metrics.compute()
        OA_value = list(value.values())[0]
        mIoU_value = list(value.values())[1]
        mF1_value = list(value.values())[2]
        self.log('val_mIoU', mIoU_value, on_epoch=True, prog_bar=True)
        self.log('val_OA', OA_value, on_epoch=True, prog_bar=True)
        self.log('val_mF1', mF1_value, on_epoch=True, prog_bar=True)
        val_ious = self.val_iou.compute()
        for i, iou in enumerate(val_ious):
            self.log(f"val_iou_class_{i}", iou)

        self.val_metrics.reset()
        self.val_iou.reset()

        PA, UA, F1, mean_F1, OA, Kappa, IoU, mIoU = calculate_index(self.confusionmat)
        mtx1 = '|OA:' + str((OA*100).round(2)) + '\n'
        mtx2 = '|mIoU:' + str((mIoU*100).round(2)) + '\n'
        mtx3 = '|mean Fscore:' + str((mean_F1*100).round(2)).replace('\n', '') + '\n'
        mtx4 = '|IoU:' + str(list((IoU*100).round(2))).replace('\n', '') + '\n'
        mtx5 = '|PA:' + str((PA*100).round(2)).replace('\n', '') + '\n'
        mtx6 = '|Fscore:' + str((F1*100).round(2)).replace('\n', '') + '\n'
        mtx7 = '|Kappa:' + str((Kappa*100).round(2)).replace('\n', '') + '\n'
        print(mtx1, mtx2, mtx3, mtx4, mtx5, mtx6, mtx7)

        with open(self.log_file, 'a') as appender:
            appender.write(mtx1)
            appender.write(mtx2)
            appender.write(mtx3)
            appender.write(mtx4)
            appender.write(mtx5)
            appender.write(mtx6)
            appender.write(mtx7)

    def on_validation_end(self):
        self.save_checkpoint(self.current_epoch,)

    def save_checkpoint(self, epoch):
        filename = f"best-checkpoint.ckpt"
        save_path = os.path.join(self.outpath, filename)
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizers().state_dict(),
            'lr_scheduler_state_dict': self.lr_schedulers().state_dict() if self.lr_schedulers() else None,
            'trainer_state': self.state_dict(),
            'hyper_parameters': dict(self.hparams) if hasattr(self, 'hparams') else {},
        }
        
        torch.save(checkpoint, save_path)
        print(f"checkpoint saved: {save_path}")

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]

        if self.hparams['deepsup']:
            y_hat, y_aux = self(x)
            y_aux_hard = y_aux.argmax(dim=1)
            loss_aux = self.criterion(y_aux.squeeze(), y.squeeze())
            self.log("test_aux_loss", loss_aux)
            self.test_aux_metrics(y_aux_hard.squeeze(), y.squeeze())
            self.log_dict(self.test_aux_metrics)
        else:
            y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat, y.squeeze(1))
        self.log("test_loss", loss)
        self.test_metrics(y_hat_hard.squeeze(), y.squeeze())
        self.log_dict(self.test_metrics)
        self.test_metrics.update(y_hat_hard.squeeze(), y.squeeze())
        self.test_iou.update(y_hat_hard.squeeze(), y.squeeze())

        mask = (y.squeeze(1) != -1)
        label = self.num_classes * y.squeeze(1)[mask] + y_hat_hard[mask]
        count_conf = torch.bincount(label, minlength=self.num_classes ** 2)
        confusionmat_tmp = count_conf.reshape(self.num_classes, self.num_classes)
        self.confusionmat = self.confusionmat.to(y.device)
        self.confusionmat += confusionmat_tmp

        if self.hparams['deepsup']:
            loss = loss + self.hparams["aux_loss_factor"] * loss_aux
        return loss
    
    def on_test_epoch_end(self):
        self.test_metrics.compute()
        test_ious = self.test_iou.compute()
        for i, iou in enumerate(test_ious):
            self.log(f"test_iou_class_{i}", iou)

        self.test_metrics.reset()
        self.test_iou.reset()

        PA, UA, F1, mean_F1, OA, Kappa, IoU, mIoU = calculate_index(self.confusionmat)
        mtx1 = '|OA:' + str((OA*100).round(2)) + '\n'
        mtx2 = '|mIoU:' + str((mIoU*100).round(2)) + '\n'
        mtx3 = '|mean Fscore:' + str((mean_F1*100).round(2)).replace('\n', '') + '\n'
        mtx4 = '|IoU:' + str(list((IoU*100).round(2))).replace('\n', '') + '\n'
        mtx5 = '|PA:' + str((PA*100).round(2)).replace('\n', '') + '\n'
        mtx6 = '|Fscore:' + str((F1*100).round(2)).replace('\n', '') + '\n'
        mtx7 = '|Kappa:' + str((Kappa*100).round(2)).replace('\n', '') + '\n'
        print(mtx1, mtx2, mtx3, mtx4, mtx5, mtx6, mtx7)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        if self.model.deepsup:
            y_hat, _ = self(x)
            y_hat = y_hat.softmax(dim=q)
        else:
            y_hat: torch.Tensor = self(x).softmax(dim=1)
        return y_hat

    def PIL_imgs_from_batch(self, x, n=4):
        """return list of PIL images from tensor input images"""
        imgs = []
        for img in x[:n]:
            img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
            # assert img.shape[-1] == 3
            if img.shape[-1] not in [3, 1]:
                img = img[:, :, [3, 2, 1]]  # S2 RGB
            # img = img.detach().cpu().numpy()
            img /= img.max(axis=(0, 1))
            img *= 255
            img = np.clip(img, 0, 255).astype(np.uint8)
            imgs.append(PIL.Image.fromarray(img))

        return imgs

    def PIL_masks_from_batch(self, x, n=4):
        """return list of PIL images from tensor input images"""
        imgs = []
        for img in x[:n]:
            # img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
            assert len(img.shape) == 2 or img.shape[-1] == 1, f"{img.shape=}"
            assert img.min() >= 0
            assert img.max() <= 255
            img = img.detach().cpu().numpy()
            img = img.astype(np.uint8) * (255 // self.hparams["num_classes"])
            imgs.append(PIL.Image.fromarray(img, mode="P"))

        return imgs

def calculate_index(confusionmat, IGNORE_LABEL=0):
    confusionmat = confusionmat.cpu().detach().numpy()

    unique_index = np.where(np.sum(confusionmat, axis=1) != 0)[0]
    assert IGNORE_LABEL == 0
    if 0 in unique_index: unique_index = unique_index[1:]
    confusionmat = confusionmat[unique_index, :]
    confusionmat = confusionmat[:, unique_index]

    a = np.diag(confusionmat)
    b = np.sum(confusionmat, axis=0)
    c = np.sum(confusionmat, axis=1)

    eps = 0.0000001

    PA = a / (c + eps)
    UA = a / (b + eps)

    F1 = 2 * PA * UA / (PA + UA + eps)

    mean_F1 = np.nanmean(F1)

    OA = np.sum(a) / np.sum(confusionmat)

    PE = np.sum(b * c) / (np.sum(c) * np.sum(c))
    Kappa = (OA - PE) / (1 - PE)

    intersection = np.diag(confusionmat)
    union = np.sum(confusionmat, axis=1) + np.sum(confusionmat, axis=0) - np.diag(confusionmat)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return PA, UA, F1, mean_F1, OA, Kappa, IoU, mIoU