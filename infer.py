import torch
from torchvision import transforms
import numpy as np

import sys
import os
import click

import dl_utils
import dl_plot
import hci4d
import loss_fns

import model
import process


@click.command()
@click.option('--out', default='out/infer', help='Output Directory')
@click.option('--lr', default=0.00001, help='Learning Rate.')
@click.option('--gpus', default=1, help='Number of GPUs to use.')
@click.option('--prms', default='params.pt', help='PT-file with pretrained network parameters.')
@click.option('--dset', default='../lf-dataset/boxes', help='Location of the inference dataset.')
@click.option('--drng', default=3, help='Disparity range, defined as [-drange, drange].')
@click.option('--stack/--nostack', default=True, help='Use a prediction stack to improve generalisation for large baselines?')
@click.option('--multi/--nomulti', default=True, help='Input multiple views to the architecture?')
def main(**hp):
    # add hyper parameters
    hp['drng'] = (-hp['drng'], hp['drng'])
    #hp['drng'] = (0, 12)
    hp['trng'] = 1.0
    hp['views'] = 9
    hp['pad'] = 12
    hp['uncrt'] = True
    hp['rgw'] = 1.0
    hp['rgl1'] = False
    hp['cll1'] = False
    hp['tri'] = False
    hp['mine'] = False

    dataset = hci4d.HCI4D(hp['dset'], nviews=(hp['views'], hp['views']),
                          cache=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False, num_workers=1)

    # create output directory
    if not os.path.exists(hp['out']):
        os.makedirs(hp['out'])

    # 1st device for all parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create net and load parameters
    net = model.Net(hp)

    dl_utils.load_model_params(net, hp['prms'])

    for p in net.parameters():
        p.requires_grad = False

    net.eval()

    # parallelize for multiple gpus
    if hp['gpus'] > 1:
        net = torch.nn.DataParallel(net, list(range(hp['gpus'])))
    net.to(device)

    # loop over the dataset
    for j, data in enumerate(dataloader):
        print(f'Processing Scene {j}...')
        # process scene
        proc = process.Process(net, device, hp)

        disp, uncert, pred, loss, runtime = proc(data)

        # save results
        dataset.save_batch(hp['out'], data[-1], disp.cpu().numpy(),
                           uncert.cpu().numpy(), runtime)

    return 0


if __name__ == '__main__':
    sys.exit(main())
