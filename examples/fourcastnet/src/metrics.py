# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from typing import Tuple
from torchmetrics.classification import AUROC, AveragePrecision, F1Score, ROC, Accuracy


class Metrics:
    """Class used for computing performance related metrics. Expects predictions /
    targets to be of shape [C, H, W] where H is latitude dimension and W is longitude
    dimension. Metrics are computed for each channel separately.

    Parameters
    ----------
    img_shape : Tuple[int]
        Shape of input image (resolution for fourcastnet)
    clim_mean_path : str, optional
        Path to total climate mean data, needed for ACC. By default "/era5/stats/time_means.npy"
    device : torch.device, optional
        Pytorch device model is on, by default 'cpu'
    """

    def __init__(
        self,
        img_shape: Tuple[int],
        num_classes: int = 2,
        clim_mean_path: str = "/era5/stats/time_means.npy",
        device: torch.device = "cpu",
    ):
        self.img_shape = tuple(img_shape)
        self.device = device
        self.num_classes = num_classes

        # Load climate mean value
        self.clim_mean = torch.as_tensor(np.load(clim_mean_path))

        # compute latitude weighting
        nlat = img_shape[0]
        lat = torch.linspace(90, -90, nlat)
        lat_weight = torch.cos(torch.pi * (lat / 180))
        lat_weight = nlat * lat_weight / lat_weight.sum()
        self.lat_weight = lat_weight.view(1, nlat, 1)
        # place on device
        if self.device is not None:
            self.lat_weight = self.lat_weight.to(self.device)
            self.clim_mean = self.clim_mean.to(self.device)

    def _check_shape(self, *args):
        # checks for shape [C, H, W]
        for x in args:
            assert x.ndim == 3
            assert tuple(x.shape[1:]) == self.img_shape

    def metrics_celled(self, all_targets, all_preds):
        if self.num_classes > 2:
            acc_table = torch.zeros(self.img_shape[0], self.img_shape[1])
            acc = Accuracy(
                task="multiclass",
                num_classes=self.num_classes,
                top_k=1,
                average="micro",
            ).to(self.device)
            acc_table = torch.tensor(
                [
                    [
                        acc(all_preds[:, :, x, y], all_targets[:, x, y])
                        for x in range(self.img_shape[0])
                    ]
                    for y in range(self.img_shape[1])
                ]
            )
            acc_table = torch.nan_to_num(acc_table, nan=0.0)
            rocauc_table_macro = torch.zeros(self.img_shape[0], self.img_shape[1])
            rocauc_table_weighted = torch.zeros(self.img_shape[0], self.img_shape[1])
            rocauc = AUROC(
                task="multiclass",
                num_classes=self.num_classes,
                average="macro",
                thresholds=20,
            )
            rocauc_table_macro = torch.tensor(
                [
                    [
                        rocauc(all_preds[:, :, x, y], all_targets[:, x, y])
                        for x in range(self.img_shape[0])
                    ]
                    for y in range(self.img_shape[1])
                ]
            )
            rocauc = AUROC(
                task="multiclass",
                num_classes=self.num_classes,
                average="weighted",
                thresholds=20,
            )
            rocauc_table_weighted = torch.tensor(
                [
                    [
                        rocauc(all_preds[:, :, x, y], all_targets[:, x, y])
                        for x in range(self.img_shape[0])
                    ]
                    for y in range(self.img_shape[1])
                ]
            )
            rocauc_table_macro = torch.nan_to_num(rocauc_table_macro, nan=0.0)
            rocauc_table_weighted = torch.nan_to_num(rocauc_table_weighted, nan=0.0)
            thresholds = torch.zeros(self.img_shape[0], self.img_shape[1])

            return acc_table, rocauc_table_macro, rocauc_table_weighted, thresholds

        else:
            rocauc_table = torch.zeros(self.img_shape[0], self.img_shape[1])
            rocauc = AUROC(task="binary", num_classes=2)
            print("table", rocauc_table.shape)
            print("preds", all_preds.shape)
            print("targets", all_targets.shape)
            rocauc_table = torch.tensor(
                [
                    [
                        rocauc(all_preds[:, :, x, y], all_targets[:, x, y])
                        for x in range(self.img_shape[0])
                    ]
                    for y in range(self.img_shape[1])
                ]
            )
            rocauc_table = torch.nan_to_num(rocauc_table, nan=0.0)

            ap_table = torch.zeros(self.img_shape[0], self.img_shape[1])
            f1_table = torch.zeros(self.img_shape[0], self.img_shape[1])
            thresholds = torch.zeros(self.img_shape[0], self.img_shape[1])

            ap = AveragePrecision(task="binary")
            roc = ROC(task="binary")
            for x in range(self.img_shape[0]):
                for y in range(self.img_shape[1]):
                    ap_table[x][y] = ap(all_preds[:, :, x, y], all_targets[:, x, y])
                    fpr, tpr, thr = roc(all_preds[:, :, x, y], all_targets[:, x, y])
                    j_stat = tpr - fpr
                    ind = torch.argmax(j_stat).item()
                    thresholds[x][y] = thr[ind]
                    f1 = F1Score(task="binary", threshold=thresholds[x][y]).to(
                        self.device
                    )
                    f1_table = f1(all_preds[:, :, x, y], all_targets[:, x, y])

            ap_table = torch.nan_to_num(ap_table, nan=0.0)
            f1_table = torch.nan_to_num(f1_table, nan=0.0)

            return rocauc_table, ap_table, f1_table, thresholds

    def weighted_acc(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the anomaly correlation coefficient (ACC). The ACC calculation is
        weighted based on the latitude.

        Parameters
        ----------
        pred : torch.Tensor
            [C, H, W] Predicted tensor
        target : torch.Tensor
            [C, H, W] Target tensor

        Returns
        -------
        torch.Tensor
            [C] ACC values for each channel
        """

        self._check_shape(pred, target)

        # subtract climate means
        (n_chans, img_x, img_y) = pred.shape
        clim_mean = self.clim_mean[0, 0:n_chans, 0:img_x]
        pred_hat = pred - clim_mean
        target_hat = target - clim_mean

        # Weighted mean
        pred_bar = torch.sum(
            self.lat_weight * pred_hat, dim=(1, 2), keepdim=True
        ) / torch.sum(
            self.lat_weight * torch.ones_like(pred_hat), dim=(1, 2), keepdim=True
        )
        target_bar = torch.sum(
            self.lat_weight * target_hat, dim=(1, 2), keepdim=True
        ) / torch.sum(
            self.lat_weight * torch.ones_like(target_hat), dim=(1, 2), keepdim=True
        )
        pred_diff = pred_hat - pred_bar
        target_diff = target_hat - target_bar

        # compute weighted acc
        # Ref: https://www.atmos.albany.edu/daes/atmclasses/atm401/spring_2016/ppts_pdfs/ECMWF_ACC_definition.pdf
        p1 = torch.sum(self.lat_weight * pred_diff * target_diff, dim=(1, 2))
        p2 = torch.sum(self.lat_weight * pred_diff * pred_diff, dim=(1, 2))
        p3 = torch.sum(self.lat_weight * target_diff * target_diff, dim=(1, 2))
        m = p1 / torch.sqrt(p2 * p3)

        return m

    def weighted_rmse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes RMSE weighted based on latitude

        Parameters
        ----------
        pred : torch.Tensor
            [C, H, W] Predicted tensor
        target : torch.Tensor
            [C, H, W] Target tensor

        Returns
        -------
        torch.Tensor
            [C] Weighted RSME values for each channel
        """
        self._check_shape(pred, target)

        # compute weighted rmse
        m = torch.sqrt(torch.mean(self.lat_weight * (pred - target) ** 2, dim=(1, 2)))

        return m
