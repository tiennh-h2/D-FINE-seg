"""Semantic segmentation loss: CE + multi-class soft Dice + auxiliary CE (deep supervision)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemSegCriterion(nn.Module):
    def __init__(
        self,
        weight_dict,
        num_classes,
        ignore_index=255,
        class_weights=None,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.weight_dict = weight_dict
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.class_weights = (
            torch.tensor(list(class_weights), dtype=torch.float32)
            if class_weights
            else None
        )

    @staticmethod
    def _structure_loss(pred, mask):
        """
        pred: (B,1,H,W) logits
        mask: (B,1,H,W) float in {0,1}
        """
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )

        wbce = F.binary_cross_entropy_with_logits(
            pred, mask, reduction="none"
        )
        wbce = (weit * wbce).sum((2, 3)) / weit.sum((2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum((2, 3))
        union = ((pred + mask) * weit).sum((2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()

    def _dice(self, logits, target, valid):
        prob = logits.softmax(1)

        one_hot = F.one_hot(
            torch.where(valid, target, 0),
            self.num_classes,
        )
        one_hot = one_hot.permute(0, 3, 1, 2).to(prob.dtype)

        v = valid.unsqueeze(1).to(prob.dtype)
        prob = prob * v
        one_hot = one_hot * v

        inter = (prob * one_hot).sum((0, 2, 3))
        denom = prob.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))

        dice = (2 * inter + 1.0) / (denom + 1.0)
        return 1.0 - dice.mean()

    def forward(self, outputs, targets):
        logits = outputs["sem_seg_logits"].float()
        target = torch.stack([t["sem_mask"] for t in targets])

        valid = target != self.ignore_index

        if not valid.any():
            zero = logits.sum() * 0.0
            losses = {
                "loss_ce": zero,
                "loss_dice": zero,
                "loss_structure": zero,
            }

            if "sem_seg_logits_aux" in outputs:
                losses["loss_aux"] = zero

        else:
            weight = (
                self.class_weights.to(logits.device)
                if self.class_weights is not None
                else None
            )

            losses = {
                "loss_ce": F.cross_entropy(
                    logits,
                    target,
                    weight=weight,
                    ignore_index=self.ignore_index,
                    label_smoothing=self.label_smoothing,
                ),
                "loss_dice": self._dice(logits, target, valid),
            }

            # Structure loss (binary segmentation only)
            if self.num_classes == 2:
                target_binary = target.clone()
                target_binary[~valid] = 0
                target_binary = target_binary.float().unsqueeze(1)

                # Use foreground logit
                pred_binary = logits[:, 1:2]

                losses["loss_structure"] = self._structure_loss(
                    pred_binary,
                    target_binary,
                )

            if "sem_seg_logits_aux" in outputs:
                losses["loss_aux"] = F.cross_entropy(
                    outputs["sem_seg_logits_aux"].float(),
                    target,
                    ignore_index=self.ignore_index,
                )

        return {
            k: v * self.weight_dict[k]
            for k, v in losses.items()
        }
