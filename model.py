import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from unet import UNet
import hci4d


class Net(nn.Module):
    """
    Network class, containing the model.
    Because this can only run on a pixel path of limited size, the a
    processing.Scene is required to run the model.
    """

    def init_feature_net(self):
        """
        Init network for feature extraction
        """
        c = 64
        v = self.hp['views']

        conv1_list = []
        conv2_list = []
        bn_list = []
        for i in range(5):
            if i == 0:
                conv1_list.append(nn.Conv2d(3 * v, c, (3, 3), padding=(1, 1)))
            else:
                conv1_list.append(nn.Conv2d(c, c, (3, 3), padding=(1, 1)))

            conv2_list.append(nn.Conv2d(c, c, (3, 3), padding=(1, 1)))
            bn_list.append(nn.BatchNorm2d(c))

        self.feature_conv1_list = torch.nn.ModuleList(conv1_list)
        self.feature_conv2_list = torch.nn.ModuleList(conv2_list)
        self.feature_bn_list = torch.nn.ModuleList(bn_list)

    def __init__(self, hp):
        """
        Init PatchNet

        :param hp: hyper parameters
        :type hp: dict
        """
        super(Net, self).__init__()

        self.hp = hp

        self.init_feature_net()

        # train network for regression and uncertainty
        n = 2
        if self.hp['multi']:
            n *= 3

        self.unet = UNet(64 * n + 3, 4, wf=6, batch_norm=True, padding=True)

    def forward_feature_net(self, shifts):
        """
        Forward pass for FeatureNet

        :param shifts: shifted EPIs
        :type shifts: torch.Tensor of size (b, 3n, h, w)

        :returns: features
        """
        x = shifts

        for i in range(len(self.feature_conv1_list)):
            x = self.feature_conv1_list[i](x)
            x = F.relu(x)
            x = self.feature_conv2_list[i](x)
            x = self.feature_bn_list[i](x)
            x = F.relu(x)

        return x

    def forward_data_net(self, shifts):
        """
        Forward pass for DataNet

        :param shifts: shifted EPIs
        :type shifts: torch.Tensor of size (b, samples, 3 * views, h, w)

        :returns: (disparity, uncertainty, label prediction, label uncertainty)
        """
        x = shifts

        y = self.unet(x)

        disp = y[:, 0]
        uncert = y[:, 1]
        pred = y[:, 2]
        pred_uncert = y[:, 3]

        return disp, uncert, pred, pred_uncert

    def forward(self, h_views, v_views):
        """
        Forward network in an end-to-end fashion

        :param h_views: horizontal view stack
        :type h_views: torch.Tensor of shape (b, n, 3, h, w)

        :param v_views: vertical view stack
        :type v_views: torch.Tensor of shape (b, n, 3, h, w)

        :returns: (disparity, uncertainty, range prediction)
        """
        # reshape input to combine view and color dimension
        b, n, c, h, w = h_views.shape
        center = h_views[:, int(h_views.shape[1] / 2)]
        h_views = h_views.view(b, n * c, h, w)
        v_views = v_views.view(b, n * c, h, w)

        # compute multi views
        shift_m = hci4d.Shift(-1)
        shift_p = hci4d.Shift(+1)

        h_views_m, v_views_m = shift_m((h_views.clone(), v_views.clone()))
        h_views_p, v_views_p = shift_p((h_views.clone(), v_views.clone()))

        # extract features
        # swap dimensions of horizontal stack
        h_views = h_views.permute(0, 1, 3, 2)

        h_features = self.forward_feature_net(h_views)

        # again swap image dimensions to concatenate with vertical EPI
        h_features = h_features.permute(0, 1, 3, 2)

        v_features = self.forward_feature_net(v_views)

        # concatenate features and compute disparity
        features = torch.cat([h_features, v_features, center], 1)

        if self.hp['multi']:
            # multi view input activated
            # swap dimensions of horizontal stack
            h_views_m = h_views_m.permute(0, 1, 3, 2)
            h_views_p = h_views_p.permute(0, 1, 3, 2)

            h_features_m = self.forward_feature_net(h_views_m)
            h_features_p = self.forward_feature_net(h_views_p)

            # again swap image dimensions to concatenate with vertical EPI
            h_features_m = h_features_m.permute(0, 1, 3, 2)
            h_features_p = h_features_p.permute(0, 1, 3, 2)

            v_features_m = self.forward_feature_net(v_views_m)
            v_features_p = self.forward_feature_net(v_views_p)

            # concatenate features and compute disparity
            features = torch.cat([h_features, v_features, h_features_m,
                                  v_features_m, h_features_p, v_features_p, center], 1)

        disp, uncert, pred, pred_uncert = self.forward_data_net(features)

        return disp.squeeze(1), uncert.squeeze(1), pred.squeeze(1), \
            pred_uncert.squeeze(1)
