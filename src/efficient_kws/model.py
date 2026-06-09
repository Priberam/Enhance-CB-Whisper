import re
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple, Optional
from .utils import KWSOutput
from .resnet import Resnet
from torch.optim import AdamW
from torchmetrics import PrecisionRecallCurve
import numpy as np
import pandas as pd
from confidence_intervals import evaluate_with_conf_int
import urllib3
import json
import torch.nn.functional as F


class KWSModel(pl.LightningModule):
    def __init__(
        self,
        num_domains: int = 72,
        sampling: str = "utterance-examples",
        resample_every_epoch: bool = True,
        kw_type: str = "tts",
        kw_p: float = 0.5,
        features_size: Tuple[int, int] = (160, 1000),
        learn_features: bool = False,
        load_embeddings: bool = True,
        n_layers: int = 12,
        pad_long_before_resize: bool = False,
        kws_whisper_ckpt: str = "openai/whisper-large-v2",
        embedding_dim: int = 1024,
        features_with_conv: bool = False,
        features_with_attn: bool = False,
        frames_conv: bool = False,
        proj_mlp: bool = False,
        proj_mlp_units: int = 64,
        batch_size: int = 1,
        accumulate_grad_batches: int = 1,
        learning_rate_sru: float = 1e-4,
        learning_rate: float = 1e-4,
        warmup_proportion: float = 0.0,
        max_epochs: int = 200,
        features_lr: float = 1e-4,
        classifier_lr: float = 1e-4,
        lr_step: int = 40,
        weight_decay: float = 0.0,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        condensed_dimension: str = "embeddings",
        resnet_version: str = "resnet-50",
        compile: bool = False,
        threshold: float = 0.5,
        task_type: str = "keyword-spotting",
        diag_size: int = 5,
        alpha_max_epochs: int = 10,
        min_alpha: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()

        print("threshold: ", self.hparams.threshold)

        # instantiate the model
        # it contains the feature extractor (Resnet) and, for the projected
        # variants, a per-layer dimensionality-reduction MLP and (optionally)
        # a per-layer temporal Conv1d.
        if not self.hparams.learn_features:
            # L: cosine-similarity matrices computed on the raw Whisper
            # activations are classified directly by the Resnet.
            self.model = Resnet(
                num_channels=self.hparams.n_layers, num_classes=2
            )

        elif self.hparams.proj_mlp:
            # LE / LEF: project the embeddings (and optionally the frames) before
            # computing the cosine-similarity matrices.
            self.model = Resnet(
                num_channels=self.hparams.n_layers,
                num_classes=2,
                version=self.hparams.resnet_version,
            )

            # embeddings' dim compression MLP projector
            self.projector = nn.ModuleList()
            if self.hparams.frames_conv:
                self.time_projector = nn.ModuleList()
            for _ in range(self.hparams.n_layers):
                self.projector.append(
                    nn.Sequential(
                        nn.Linear(
                            self.hparams.embedding_dim,
                            self.hparams.embedding_dim // 2,
                        ),
                        nn.ReLU(),
                        nn.Linear(
                            self.hparams.embedding_dim // 2,
                            self.hparams.proj_mlp_units,
                        ),
                    )
                )

                # time's dim compression Conv1D
                if self.hparams.frames_conv:
                    self.time_projector.append(
                        torch.nn.Sequential(
                            nn.Conv1d(
                                in_channels=self.hparams.proj_mlp_units,
                                out_channels=self.hparams.proj_mlp_units,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                            ),
                            torch.nn.BatchNorm1d(
                                num_features=self.hparams.proj_mlp_units
                            ),
                            torch.nn.MaxPool1d(
                                kernel_size=3, stride=2, padding=1
                            ),
                        )
                    )

        # instantiate a PrecisionRecallCurve object
        self.pr_curve = PrecisionRecallCurve(task="binary")

    def forward(
        self,
        kwd_features: torch.Tensor,
        utt_features: torch.Tensor,
        labels: torch.Tensor = None,
        kwd_mask: Optional[torch.Tensor] = None,
        utt_mask: Optional[torch.Tensor] = None,
    ):
        """
        - 'utt_features': Tensor([batch_size, n_layers, n_frames_utt, emb_dim])
        - 'kwd_features': Tensor([batch_size, n_layers, n_frames_kwd, emb_dim])
        - 'labels': Tensor([batch_size])
        """

        if self.hparams.proj_mlp:
            projected_utt = []
            projected_kwd = []
            for i in range(self.hparams.n_layers):
                projected_kwd.append(self.projector[i](kwd_features[:, i]))
                projected_utt.append(self.projector[i](utt_features[:, i]))
            projected_kwd = torch.stack(projected_kwd, dim=1)
            projected_utt = torch.stack(projected_utt, dim=1)

            if self.hparams.frames_conv:
                projected_utt_conv = []
                projected_kwd_conv = []
                for i in range(self.hparams.n_layers):
                    proj_utt = self.time_projector[i](
                        projected_utt[:, i].transpose(1, 2)
                    ).transpose(1, 2)
                    projected_utt_conv.append(proj_utt)

                    proj_kwd = self.time_projector[i](
                        projected_kwd[:, i].transpose(1, 2)
                    ).transpose(1, 2)
                    projected_kwd_conv.append(proj_kwd)
                projected_utt = torch.stack(projected_utt_conv, dim=1)
                projected_kwd = torch.stack(projected_kwd_conv, dim=1)
        else:
            projected_kwd = kwd_features
            projected_utt = utt_features

        # cosine similarity matrices: one per Whisper layer, stacked into the
        # Resnet channel dimension (num_channels == n_layers). The utterance
        # batch (1 at eval, == n_keywords at train) is expanded to n_keywords.
        n_keywords = projected_kwd.size(0)
        input_features = torch.stack(
            [
                self.sim_matrix(
                    projected_utt[:, layer].expand(n_keywords, -1, -1),
                    projected_kwd[:, layer],
                ).permute(0, 2, 1)
                for layer in range(projected_kwd.size(1))
            ],
            dim=1,
        )  # Tensor([n_keywords, n_layers, n_frames_kwd, n_frames_utt])

        # zero out padded frames; masks broadcast across the n_layers channel dim
        input_features = (
            input_features
            * utt_mask.unsqueeze(2)
            * kwd_mask.unsqueeze(-1)
        )

        logits_resnet = self.resnet_forward(input_features)

        if labels != None:
            loss_resnet = F.cross_entropy(logits_resnet, labels.view(-1))
        else:
            loss_resnet = None

        loss = loss_resnet

        return KWSOutput(
            loss=loss,
            logits=logits_resnet,
            features=input_features,
            logits_alt=None,
            loss_alt={"loss_diag": None, "loss_resnet": loss_resnet},
        )

    def sim_matrix(self, a, b, eps=1e-6):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=-1)[:, :, None], b.norm(dim=-1)[:, :, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.bmm(a_norm, b_norm.transpose(-2, -1))
        return sim_mt

    def resnet_forward(self, input_features: torch.Tensor):
        return self.model(input_features)

    def backward(self, loss):
        loss.backward(retain_graph=True)

    def training_step(self, batch, batch_idx):
        """
        batch:
        - 'utt_features': Tensor([batch_size, n_layers, n_frames_utt, emb_dim])
        - 'kwd_features': Tensor([batch_size, n_layers, n_frames_kwd, emb_dim])
        - 'utt_mask' / 'kwd_mask': Tensor([batch_size, n_layers, n_frames])
        - 'labels': Tensor([batch_size])
        - 'domain': Tensor([batch_size])
        """
        input_features = {
            "utt_features": batch["utt_features"],
            "kwd_features": batch["kwd_features"],
            "utt_mask": batch["utt_mask"],
            "kwd_mask": batch["kwd_mask"],
        }

        if self.hparams.kw_type == "all":
            # whether to use tts or natural -based keywords
            k_mask = (
                torch.rand(int(input_features["utt_features"].size(dim=0) / 2))
                > self.hparams.kw_p
            )
            k_mask = torch.stack(
                (k_mask, torch.logical_not(k_mask)), dim=1
            ).flatten()
            for k in ("utt_features", "kwd_features", "utt_mask", "kwd_mask"):
                input_features[k] = input_features[k][k_mask]

            c_labels = batch["labels"][k_mask]
        else:
            c_labels = batch["labels"]

        batch_size = input_features["utt_features"].size(dim=0)

        # forward minibatch through the model
        kws_output = self.forward(
            kwd_features=input_features["kwd_features"],
            utt_features=input_features["utt_features"],
            labels=c_labels,
            kwd_mask=input_features.get("kwd_mask", None),
            utt_mask=input_features.get("utt_mask", None),
        )

        # compute training loss
        loss = kws_output.loss
        loss_resnet = kws_output.loss_alt.get("loss_resnet", None)

        # log losses
        try:
            self.log(
                "train/loss",
                loss,
                batch_size=batch_size,
                prog_bar=True,
                on_epoch=True,
            )
            self.log(
                "train/loss_resnet",
                loss_resnet,
                batch_size=batch_size,
                prog_bar=True,
                on_epoch=True,
            )

            sch = self.lr_schedulers()
            self.log("lr", sch.get_last_lr()[0])

        except urllib3.exceptions.NewConnectionError:
            print("unable to log.")

        return loss

    def on_validation_epoch_start(self):
        # list for storing the outputs of each validation step
        # in the past this was done automatically using the validation_epoch_end hook
        # but https://github.com/Lightning-AI/lightning/pull/16520
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        batch:
        - 'hotword_labels': list[ Tensor([group_size]) ]
        - 'hotword_mask': list[ Tensor([group_size]) ]
        - 'utt': Tensor([n_layers, n_frames, emb_dim])
        - 'utt_mask': Tensor([n_layers, n_frames])
        - 'kwd': list[ list[ Tensor([n_layers, n_frames, emb_dim]) ] ]
        - 'kwd_mask': list[ list[ Tensor([n_layers, n_frames]) ] ]
        """
        for k in ("kwd", "kwd_mask"):
            batch[k] = [
                torch.stack(batch[k][i]) for i in range(len(batch[k]))
            ]

        n_groups = len(batch["kwd"])

        input_features = []
        for i in range(n_groups):
            input_features.append(
                {
                    "utt_features": batch["utt"].unsqueeze(0),
                    "kwd_features": batch["kwd"][i],
                    "utt_mask": batch["utt_mask"].unsqueeze(0),
                    "kwd_mask": batch["kwd_mask"][i],
                }
            )

        # forward each group through the model
        output = []
        for input_features_, labels in zip(
            input_features, batch["hotword_labels"]
        ):
            output.append(
                self.forward(
                    kwd_features=input_features_["kwd_features"],
                    utt_features=input_features_["utt_features"],
                    labels=labels,
                    kwd_mask=input_features_.get("kwd_mask", None),
                    utt_mask=input_features_.get("utt_mask", None),
                )
            )

        # get loss
        loss = sum(out.loss for out in output).detach()

        if batch.get("hotword_mask", None) != None:
            preds = torch.cat(
                [
                    out.logits.softmax(dim=-1)[:, 1].detach() * mask
                    for out, mask in zip(output, batch["hotword_mask"])
                ],
                dim=0,
            )
        else:
            preds = torch.cat(
                [
                    out.logits.softmax(dim=-1)[:, 1].detach()
                    for out in output
                ],
                dim=0,
            )

        targets = torch.cat(
            [labels for labels in batch["hotword_labels"]], dim=0
        )

        # append to validation step outputs for correct dataloader
        if dataloader_idx > len(self.validation_step_outputs) - 1:
            self.validation_step_outputs += [[]] * (
                dataloader_idx - len(self.validation_step_outputs) + 1
            )

        self.validation_step_outputs[dataloader_idx].append(
            {
                "loss": loss,
                "loss_alt": None,
                "preds": preds,
                "preds_alt": None,
                "targets": targets,
            }
        )

    def on_validation_epoch_end(self):
        is_expanded = any(
            [
                dataloader.dataset.is_expanded()
                for dataloader in self.trainer.val_dataloaders
            ]
        )
        n_languages = len(self.validation_step_outputs)
        if not is_expanded:
            n_languages = n_languages // 2
        else:
            n_languages = n_languages // 4

        if len(self.trainer.val_dataloaders) == 1:
            n_languages = 1

        # to store the average metrics
        avg_metrics = {
            metric: 0.0
            for metric in [
                "metrics/loss",
                "metrics/loss_alt",
                "metrics/precision",
                "metrics/recall",
                "metrics/f1",
                "metrics/recall_at_10",
                "metrics/precision_alt",
                "metrics/recall_alt",
                "metrics/f1_alt",
                "metrics/recall_alt_at_10",
                "val/recall_at_20",
                "val/recall_at_50",
                "val/recall_at_100",
                "val/recall_at_200",
                "val/recall_at_1",
            ]
        }

        language_metrics = {
            f"l{l_idx}": {
                f"metrics/loss_l{l_idx}": 0.0,
                f"metrics/loss_alt_l{l_idx}": 0.0,
                f"metrics/precision_l{l_idx}": 0.0,
                f"metrics/recall_l{l_idx}": 0.0,
                f"metrics/f1_l{l_idx}": 0.0,
                f"metrics/recall_at_10_l{l_idx}": 0.0,
                f"metrics/precision_alt_l{l_idx}": 0.0,
                f"metrics/recall_alt_l{l_idx}": 0.0,
                f"metrics/f1_alt_l{l_idx}": 0.0,
                f"metrics/recall_alt_at_10_l{l_idx}": 0.0,
                f"val/recall_at_20_l{l_idx}": 0.0,
                f"val/recall_at_50_l{l_idx}": 0.0,
                f"val/recall_at_100_l{l_idx}": 0.0,
                f"val/recall_at_200_l{l_idx}": 0.0,
                f"val/recall_at_1_l{l_idx}": 0.0,
            }
            for l_idx in range(n_languages)
        }

        best_thresholds = []
        # loop through the results of each dataloader
        for dataloader_idx, validation_step_outputs in enumerate(
            self.validation_step_outputs
        ):
            if not self.trainer.val_dataloaders[
                dataloader_idx
            ].dataset.is_expanded():
                lst_keys = (
                    ["preds", "preds_alt"]
                    if self.hparams.task_type == "mixed"
                    else ["preds"]
                )

                best_precision, best_precision_alt = 0.0, 0.0
                best_recall, best_recall_alt = 0.0, 0.0
                best_f1, best_f1_alt = 0.0, 0.0
                recall_at_10, recall_at_10_alt = [], []
                # compute precision-recall curve
                for k in lst_keys:
                    precision, recall, thresholds = self.pr_curve(
                        torch.cat(
                            [
                                step_output[k]
                                for step_output in validation_step_outputs
                            ],
                            dim=0,
                        ),
                        torch.cat(
                            [
                                step_output["targets"]
                                for step_output in validation_step_outputs
                            ],
                            dim=0,
                        ),
                    )

                    idx = self.find_best_idx(precision, recall, thresholds)

                    if len(thresholds.shape) == 0:
                        best_thresholds.append(thresholds.item())
                    else:
                        best_thresholds.append(thresholds[idx].item())

                    if k == "preds_alt":
                        best_precision_alt = precision[idx].item()
                        best_recall_alt = recall[idx].item()
                        best_f1_alt = (
                            2.0
                            * precision[idx].item()
                            * recall[idx].item()
                            / (precision[idx].item() + recall[idx].item())
                            if (
                                precision[idx].item() != 0
                                and recall[idx].item() != 0
                            )
                            else 0.0
                        )
                    else:
                        best_precision = precision[idx].item()
                        best_recall = recall[idx].item()
                        best_f1 = (
                            2.0
                            * precision[idx].item()
                            * recall[idx].item()
                            / (precision[idx].item() + recall[idx].item())
                            if (
                                precision[idx].item() != 0
                                and recall[idx].item() != 0
                            )
                            else 0.0
                        )

                    recall_10 = []
                    for step_output in validation_step_outputs:
                        preds = step_output[k]
                        targets = step_output["targets"]
                        top_scores_10, top_indices_10 = torch.topk(preds, k=10)
                        r_10 = 0
                        for target in targets.nonzero()[:, 0]:
                            if target.item() in top_indices_10:
                                r_10 += 1
                        r_10 = (
                            (r_10 / targets.sum())
                            if targets.sum() > 0
                            else torch.tensor(-1, device=self.device)
                        )
                        if r_10 >= 0:
                            recall_10.append(r_10)

                    if len(recall_10) > 0:
                        recall_10 = torch.stack(recall_10)
                    else:
                        recall_10 = torch.stack([torch.tensor(0.0)])

                    if k == "preds_alt":
                        recall_at_10_alt = recall_10
                    else:
                        recall_at_10 = recall_10

                # save precision-recall curve as a json file
                try:
                    path_prcurve = "/".join(
                        self.trainer.ckpt_path.split("/")[:-1]
                    )

                    path_prcurve += f"/prcurve_{dataloader_idx}.json"

                    print("Saving precision-recall curve at: ", path_prcurve)

                    pr_curve_dict = {
                        "precision": precision.cpu().tolist(),
                        "recall": recall.cpu().tolist(),
                        "thresholds": thresholds.cpu().tolist(),
                    }

                    with open(
                        path_prcurve,
                        "w",
                    ) as f:
                        json.dump(pr_curve_dict, f)
                except Exception:
                    print("Not saving precision-recall curve.")

                metrics = {
                    "metrics/loss"
                    + "_"
                    + str(dataloader_idx): sum(
                        step_output["loss"]
                        for step_output in validation_step_outputs
                    )
                    / len(validation_step_outputs),
                    "metrics/loss_alt"
                    + "_"
                    + str(dataloader_idx): sum(
                        step_output["loss_alt"]
                        for step_output in validation_step_outputs
                    ) if self.hparams.task_type == "mixed" else 0.0
                    / len(validation_step_outputs),
                    "metrics/precision"
                    + "_"
                    + str(dataloader_idx): best_precision,
                    "metrics/recall" + "_" + str(dataloader_idx): best_recall,
                    "metrics/f1" + "_" + str(dataloader_idx): best_f1,
                    "metrics/precision_alt"
                    + "_"
                    + str(dataloader_idx): best_precision_alt,
                    "metrics/recall_alt"
                    + "_"
                    + str(dataloader_idx): best_recall_alt,
                    "metrics/f1_alt" + "_" + str(dataloader_idx): best_f1_alt,
                    "metrics/recall_at_10"
                    + "_"
                    + str(dataloader_idx): recall_at_10.mean(),
                    "metrics/recall_alt_at_10"
                    + "_"
                    + str(dataloader_idx): recall_at_10_alt.mean() if self.hparams.task_type == "mixed" else 0.0,
                }

            # add to the average metrics
            for key in avg_metrics.keys():
                if key + "_" + str(dataloader_idx) in metrics:
                    div = len(self.validation_step_outputs)
                    if is_expanded:
                        div = div // 2
                    if len(self.trainer.val_dataloaders) == 1:
                        div = 1
                    avg_metrics[key] += (
                        metrics[key + "_" + str(dataloader_idx)] / div
                    )

            l_idx = dataloader_idx // 2
            if is_expanded:
                l_idx = l_idx // 2
            for key in metrics.keys():
                if (
                    re.sub("_\d+$", "", key) + f"_l{l_idx}"
                    in language_metrics[f"l{l_idx}"]
                ):
                    if is_expanded:
                        div = 2
                    else:
                        div = 4
                    if len(self.trainer.val_dataloaders) == 1:
                        div = 1
                    language_metrics[f"l{l_idx}"][
                        re.sub("_\d+$", "", key) + f"_l{l_idx}"
                    ] += (metrics[key] / div)

            try:
                # log metrics that can be averaged across processes
                self.log_dict(metrics, sync_dist=True)
            except urllib3.exceptions.NewConnectionError:
                print("unable to log.")

        print("Best thresholds: ", best_thresholds)

        try:
            path_thresdict = "/".join(self.trainer.ckpt_path.split("/")[:-1])

            path_thresdict += "/thresdict.json"

            print("Saving thresholds at: ", path_thresdict)

            with open(
                path_thresdict,
                "w",
            ) as f:
                json.dump(best_thresholds, f)
        except Exception:
            print("Not saving thresholds.")

        # log average metrics
        try:
            self.log_dict(avg_metrics, sync_dist=True)
            for l_idx, metrics in language_metrics.items():
                self.log_dict(metrics, sync_dist=True)
        except urllib3.exceptions.NewConnectionError:
            print("unable to log.")

        # empty validation step outputs list
        self.validation_step_outputs = []

    def find_best_idx(self, precision, recall, thresholds):
        num = (5) * (precision * recall)  # Tensor([n_thresholds + 1])

        den = 4 * precision + recall  # Tensor([n_thresholds + 1])

        # compute f-score for each beta and each threshold
        f1_scores = num / den  # Tensor([n_thresholds + 1])

        # remove nans
        f1_scores[torch.isnan(f1_scores)] = 0.0

        idx = f1_scores.argmax()

        return idx

    def configure_optimizers(self):
        list_parameters = [
            dict(
                params=[
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad
                ],
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(self.hparams.beta_1, self.hparams.beta_2),
            )
        ]

        if self.hparams.proj_mlp:
            list_parameters += [
                dict(
                    params=[
                        p
                        for n, p in self.projector.named_parameters()
                        if p.requires_grad
                    ],
                    lr=self.hparams.learning_rate_sru,
                    weight_decay=self.hparams.weight_decay,
                    betas=(self.hparams.beta_1, self.hparams.beta_2),
                )
            ]

            if self.hparams.frames_conv:
                list_parameters += [
                    dict(
                        params=[
                            p
                            for n, p in self.time_projector.named_parameters()
                            if p.requires_grad
                        ],
                        lr=self.hparams.learning_rate_sru,
                        weight_decay=self.hparams.weight_decay,
                        betas=(self.hparams.beta_1, self.hparams.beta_2),
                    )
                ]

        optimizer = AdamW(
            list_parameters,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta_1, self.hparams.beta_2),
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_test_epoch_start(self):
        # list for storing the outputs of each test step
        # in the past this was done automatically using the test_epoch_end hook
        # but https://github.com/Lightning-AI/lightning/pull/16520
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        for k in ("kwd", "kwd_mask"):
            batch[k] = [
                torch.stack(batch[k][i]) for i in range(len(batch[k]))
            ]

        n_groups = len(batch["kwd"])

        input_features = []
        for i in range(n_groups):
            input_features.append(
                {
                    "utt_features": batch["utt"].unsqueeze(0),
                    "kwd_features": batch["kwd"][i],
                    "utt_mask": batch["utt_mask"].unsqueeze(0),
                    "kwd_mask": batch["kwd_mask"][i],
                }
            )

        # forward each group through the model
        output = []
        for input_features_, labels in zip(
            input_features, batch["hotword_labels"]
        ):
            output.append(
                self.forward(
                    kwd_features=input_features_["kwd_features"],
                    utt_features=input_features_["utt_features"],
                    labels=labels,
                    kwd_mask=input_features_.get("kwd_mask", None),
                    utt_mask=input_features_.get("utt_mask", None),
                )
            )

        # join predicted class probabilities and target class indices
        if batch.get("hotword_mask", None) != None:
            preds = torch.cat(
                [
                    out.logits.softmax(dim=-1)[:, 1].detach() * mask
                    for out, mask in zip(output, batch["hotword_mask"])
                ],
                dim=0,
            )
        else:
            preds = torch.cat(
                [out.logits.softmax(dim=-1)[:, 1].detach() for out in output],
                dim=0,
            )

        targets = torch.cat(
            [labels for labels in batch["hotword_labels"]], dim=0
        )
        self.test_step_outputs.append(
            {"preds": preds, "targets": targets, "speaker": batch["speaker"]}
        )

    def on_test_epoch_end(self):

        def f_precision(labels, samples, samples2=None):
            # compute precision-recall curve
            precision, _, thresholds = self.pr_curve(samples, labels)
            # get precision at the configured operating threshold
            idx = torch.where(thresholds - self.hparams.threshold < 0)[0].size(
                dim=0
            )
            return precision[idx].item()

        def f_recall(labels, samples, samples2=None):
            # compute precision-recall curve
            _, recall, thresholds = self.pr_curve(samples, labels)
            # get recall at the configured operating threshold
            idx = torch.where(thresholds - self.hparams.threshold < 0)[0].size(
                dim=0
            )
            return recall[idx].item()

        def f_f1(labels, samples, samples2=None):
            # compute precision-recall curve
            precision, recall, thresholds = self.pr_curve(samples, labels)

            # get precision and recall at the configured operating threshold
            idx = torch.where(thresholds - self.hparams.threshold < 0)[0].size(
                dim=0
            )
            return (
                2
                * precision[idx].item()
                * recall[idx].item()
                / (precision[idx].item() + recall[idx].item())
                if (precision[idx].item() != 0 and recall[idx].item() != 0)
                else 0
            )

        # set conditions based on speaker
        speakers = [
            step_output["speaker"]
            for step_output in self.test_step_outputs
            for i_ in range(len(step_output["preds"]))
        ]
        speaker2id = {
            speaker: speaker_id
            for speaker_id, speaker in enumerate(set(speakers))
        }
        conditions = [speaker2id[speaker] for speaker in speakers]

        samples = torch.cat(
            [step_output["preds"] for step_output in self.test_step_outputs],
            dim=0,
        )
        labels = torch.cat(
            [step_output["targets"] for step_output in self.test_step_outputs],
            dim=0,
        )
        precision = evaluate_with_conf_int(
            samples,
            f_precision,
            labels,
            conditions,
            num_bootstraps=1000,
            alpha=5,
        )
        recall = evaluate_with_conf_int(
            samples, f_recall, labels, conditions, num_bootstraps=1000, alpha=5
        )
        f1 = evaluate_with_conf_int(
            samples, f_f1, labels, conditions, num_bootstraps=1000, alpha=5
        )

        metrics = {
            "Precision": precision[0],
            "Precision_LB": precision[1][0],
            "Precision_UB": precision[1][1],
            "Recall": recall[0],
            "Recall_LB": recall[1][0],
            "Recall_UB": recall[1][1],
            "F1": f1[0],
            "F1_LB": f1[1][0],
            "F1_UB": f1[1][1],
        }

        # display results using pandas DataFrame
        m_names = list(metrics.keys())
        results = pd.DataFrame(
            np.array([[metrics[m_] for m_ in m_names]]), columns=m_names
        )
        print(results)

        # save precision-recall curve data with all data points for plotting
        preds = [
            step_output["preds"] for step_output in self.test_step_outputs
        ]
        golds = [
            step_output["targets"] for step_output in self.test_step_outputs
        ]
        precision, recall, thresholds = self.pr_curve(
            torch.cat(preds, dim=0), torch.cat(golds, dim=0)
        )
        pr_data = {
            "precision": precision.cpu().numpy().tolist(),
            "recall": recall.cpu().numpy().tolist(),
            "thresholds": thresholds.cpu().numpy().tolist(),
        }
        try:
            path_prdata = "/".join(self.trainer.ckpt_path.split("/")[:-1])

            if "ACL6060" in self.trainer.test_dataloaders.dataset.root:
                path_prdata += "/pr_data_acl6060.json"
            else:
                path_prdata += "/pr_data_aishell.json"

            print("Saving precision-recall curve data at: ", path_prdata)

            with open(
                path_prdata,
                "w",
            ) as f:
                json.dump(pr_data, f)
        except Exception:
            print("Not saving precision-recall curve data.")

        # empty test step outputs list
        self.test_step_outputs = []

    def on_load_checkpoint(self, checkpoint):
        # because of code changes, early experiment checkpoints have the `model.resnet` attribute
        # the following code is to adapt the state dict to the current version of the code
        resnet_regex = re.compile("resnet.")
        feature_extractor_regex = re.compile("(model.embedder|model.encoder)")
        state_dict_keys = list(checkpoint["state_dict"].keys())
        if any(
            [
                resnet_regex.search(state_dict_key) != None
                for state_dict_key in state_dict_keys
            ]
        ):
            for state_dict_key in state_dict_keys:
                new_state_dict_key = resnet_regex.sub("", state_dict_key)
                if match := feature_extractor_regex.search(new_state_dict_key):
                    new_state_dict_key = (
                        new_state_dict_key[:6]
                        + "feature_extractor."
                        + new_state_dict_key[6:]
                    )
                weights = checkpoint["state_dict"].pop(state_dict_key)
                checkpoint["state_dict"][new_state_dict_key] = weights
