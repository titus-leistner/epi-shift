import time

import torch

import hci4d
import loss_fns
import dl_utils


class Process:
    """
    Callable to process one lightfield scene
    """

    def __init__(self, net, device, hp):
        """
        :param net: the network object
        :type net: model.Net

        :param device: device for tensors
        :type device: str

        :param hp: hyper parameters
        :type hp: dict
        """

        self.net = net
        self.device = device
        self.disp_range = hp['drng']
        self.train_range = hp['trng']
        self.pad = hp['pad']
        self.uncert = hp['uncrt']
        self.reg_weight = hp['rgw']
        self.triangle = hp['tri']
        self.stack = hp['stack']
        self.mine = hp['mine']

        assert(isinstance(self.net, torch.nn.Module))
        assert(isinstance(self.disp_range, tuple) and len(self.disp_range) == 2)
        assert(isinstance(self.train_range, float))
        assert(isinstance(self.stack, bool))
        assert(isinstance(self.pad, int))
        assert(isinstance(self.reg_weight, float))
        assert(isinstance(self.uncert, bool))
        assert(isinstance(self.triangle, bool))
        assert(isinstance(self.mine, bool))

        # decide for loss functions
        if hp['uncrt']:
            if hp['rgl1']:
                self.reg_loss = loss_fns.l1_uncert
            else:
                self.reg_loss = loss_fns.mse_uncert

            if hp['cll1']:
                self.class_loss = loss_fns.l1_uncert
            else:
                self.class_loss = loss_fns.mse_uncert
        else:
            if hp['rgl1']:
                self.reg_loss = loss_fns.l1
            else:
                self.reg_loss = loss_fns.mse

            if hp['cll1']:
                self.class_loss = loss_fns.l1
            else:
                self.class_loss = loss_fns.mse

    def stack_shifts(self, data):
        """
        Shift a scene multiple times according to disp_range and stack shifts

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: (h_views, v_views, center, gt, index)
        """
        h_views, v_views, center, gt, index = data
        h_stack = []
        v_stack = []
        gt_stack = []

        for disp in range(self.disp_range[0], self.disp_range[1] + 1):
            shift = hci4d.Shift(disp)

            data = (h_views.clone(), v_views.clone(),
                    center, gt.clone(), index)
            h_shift, v_shift, _, gt_shift, _ = shift(data)

            h_stack.append(h_shift)
            v_stack.append(v_shift)
            gt_stack.append(gt_shift)

        h_views = torch.cat(h_stack, 0)
        v_views = torch.cat(v_stack, 0)
        gt = torch.cat(gt_stack, 0)

        return h_views, v_views, center, gt, index

    def reduce_shifts(self, disp, uncert, pred):
        """
        Reduce a batch of shifts to a single disparity output

        :param disp: batch of disparity maps 
        :type disp: torch.Tensor

        :param uncert: batch of uncertainties
        :type uncert: torch.Tensor

        :param pred: the range prediction output of the network
        :type pred: torch.Tensor

        :returns: (reduced disparity, corresponding uncertainty)
        """
        label = torch.max(pred, 0)[1]

        out_disp = disp[0:1].clone()
        out_uncert = uncert[0:1].clone()

        for y in range(out_disp.shape[1]):
            for x in range(out_disp.shape[2]):
                out_disp[0, y, x] = disp[label[y, x], y, x]
                out_uncert[0, y, x] = uncert[label[y, x], y, x]

        out_disp += label.float().unsqueeze(0) + self.disp_range[0]
        #out_disp = label.float().unsqueeze(0) + self.disp_range[0]

        return out_disp, out_uncert

    def __call__(self, data):
        """
        Runs the network and returns the results and the loss

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns (disp, uncert, pred, loss, runtime)
        """
        t_start = time.time()

        batch_net = self.net

        gt_red = data[3]
        if self.stack:
            # stack shifts
            data = self.stack_shifts(data)
            # batch_net = dl_utils.BatchIter(self.net)

        # copy to GPU
        h_views, v_views, center, gt, index = data
        h_views = h_views.to(self.device)
        v_views = v_views.to(self.device)
        gt = gt.to(self.device)
        gt_red = gt_red.to(self.device)

        # execute network
        disp, uncert, pred, pred_uncert = batch_net(
            h_views, v_views)

        disp_red = disp.clone()
        uncert_red = uncert.clone()
        if self.stack:
            disp_red, uncert_red = self.reduce_shifts(
                disp, uncert, pred)

        # compute loss
        # crop the results
        w = disp.shape[-1]
        h = disp.shape[-2]
        w -= 2 * self.pad
        h -= 2 * self.pad

        crop = hci4d.CenterCrop((h, w))

        disp_crop, uncert_crop, pred_crop, pred_uncert_crop, gt_crop, disp_red_crop, uncert_red_crop, gt_red_crop = crop(
            (disp, uncert, pred, pred_uncert, gt, disp_red, uncert_red, gt_red))

        # regression loss
        loss = self.reg_weight * self.reg_loss(
            disp_crop, uncert_crop, gt_crop, None, self.train_range + 0.1)

        if not self.stack:
            loss = self.reg_weight * self.reg_loss(
                disp_crop, uncert_crop, gt_crop)

        if self.stack:
            # MSE over the image
            mse = None

            if self.mine:
                mse = torch.pow(disp_red_crop - gt_red_crop, 2.0).detach()
                dl_utils.save_img('mse.png', mse.squeeze(
                    0).detach().cpu().numpy())
                mse += 1.0

            # classification loss
            class_gt = None

            # triangular or rectangular function?
            if self.triangle:
                class_gt = self.train_range - torch.abs(gt_crop)
                class_gt = torch.clamp(class_gt, min=0.0)
            else:
                class_gt = (torch.abs(gt_crop) <= self.train_range).float()

            loss += self.class_loss(pred_crop,
                                    pred_uncert_crop, class_gt, mse)

        t_end = time.time()
        runtime = t_end - t_start

        if self.stack:
            disp = disp_red
            uncert = uncert_red

        return disp, uncert, pred, loss, runtime
