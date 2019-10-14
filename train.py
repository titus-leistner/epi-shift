import torch
from torchvision import transforms
import numpy as np

import sys
import os
import datetime
import click

import dl_utils
import dl_plot
import hci4d
import loss_fns

import model
import process


@click.command()
@click.option('--out', default='out/train', help='Output Directory')
@click.option('--lr', default=0.00001, help='Learning Rate.')
@click.option('--bsz', default=1, help='Batch size for training')
@click.option('--psz', default=32, help='Size of testing images.')
@click.option('--gpus', default=1, help='Number of GPUs to use.')
@click.option('--prms', default='', help='PT-file with pretrained network parameters.')
@click.option('--tset', default='../lf-dataset/additional', help='Location of the training dataset.')
@click.option('--vset', default='../lf-dataset/boxes', help='Location of the validation dataset.')
@click.option('--drng', default=3, help='Disparity range, defined as [-drange, drange].')
@click.option('--trng', default=0.5, help='Disparity range for training, defined as [-drange, drange].')
@click.option('--stack/--nostack', default=True, help='Use a prediction stack to improve generalisation for large baselines?')
@click.option('--fix/--train', default=False, help='Fix or train BatchNorm layers?')
@click.option('--uncrt/--nouncrt', default=False, help='Use uncertainty?')
@click.option('--rgl1/--rgmse', default=False, help='Use L1 or L2 loss for regression?')
@click.option('--cll1/--clmse', default=False, help='Use L1 or L2 loss for classification?')
@click.option('--rgw', default=1.0, help='Weight for regression loss')
@click.option('--tri/--rect', default=False, help='Triangular or rectangular function for classification?')
@click.option('--mine/--nomine', default=False, help='Hardmining for wrong classifications using MSE weighting?')
@click.option('--multi/--nomulti', default=True, help='Input multiple views to the architecture?')
def main(**hp):
    # add hyper parameters
    hp['drng'] = (-hp['drng'], hp['drng'])
    hp['views'] = 9
    hp['pad'] = 12

    # create output directory
    if not os.path.exists(hp['out']):
        os.makedirs(hp['out'])
    param_dir = os.path.join(hp['out'], 'params', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(param_dir)

    # output hps
    dl_utils.save_hyper_params(hp, param_dir)

    # init plotting
    plot = dl_plot.Plot(os.path.join(hp['out'], 'plots'), True)

    transform = transforms.Compose([
        hci4d.RandomCrop(512 - 32, pad=16),
        #hci4d.RandomShift(hp['drng']),
        hci4d.RandomZoom(0.5, 1.0),
        hci4d.CenterCrop(hp['psz']),
        hci4d.RandomRotate(),
        hci4d.RedistColor(),
        hci4d.Brightness(),
        hci4d.Contrast()])

    # import datasets
    trainset = hci4d.HCI4D(hp['tset'], transform=transform, length=4096)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=hp['bsz'],
                                              shuffle=True, num_workers=4)

    testset = hci4d.HCI4D(hp['vset'])
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=1,
                                             shuffle=False, num_workers=1)

    # 1st device for all parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load parameters, if necessary
    net = model.Net(hp)

    if hp['prms']:
        dl_utils.load_model_params(net, hp['prms'])

    # parallelize for multiple gpus
    if hp['gpus'] > 1:
        net = torch.nn.DataParallel(net, list(range(hp['gpus'])))
    net.to(device)

    # init optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=hp['lr'])

    i = 0
    while True:
        try:
            # loop over the dataset multiple times
            for j, data in enumerate(trainloader):
                if j == 10000:
                    print("Lower learning rate and fix BatchNorm...")
                    optimizer = torch.optim.Adam(net.parameters(), lr=hp['lr'] / 10.0)
                    hp['fix'] = True

                # prepare training
                for p in net.parameters():
                    p.requires_grad = True

                net.train()

                if hp['fix']:
                    net.eval()

                optimizer.zero_grad()

                # process scene
                proc = process.Process(net, device, hp)

                disp, uncert, pred, train_loss, runtime = proc(data)

                train_loss.backward()
                optimizer.step()

                # process test data
                if i % 25 == 0:
                    #print('Processing Test Data...')
                    k, data = next(enumerate(testloader))
                    
                    # evaluate
                    for p in net.parameters():
                        p.requires_grad = False

                    net.eval()

                    proc = process.Process(net, device, hp)

                    disp, uncert, pred, test_loss, runtime = proc(data)

                    # plot
                    weights, biases = dl_plot.extract_params_pytorch(net)

                    uncert_img = uncert.detach().cpu().numpy()[0]
                    pred_img = pred.detach().cpu().numpy()[0]

                    if hp['stack']:
                        pred_img = pred.detach().cpu().numpy()[3]

                    gt_img = data[3].detach().cpu().numpy()[0]
                    #gt_img = np.round(gt_img)
                    disp_img = disp.detach().cpu().numpy()[0]
                    norm = (np.amin(gt_img), np.amax(gt_img))

                    plot.plot(i, train_loss, test_loss, weights, biases,
                              pred_img, gt_img, disp_img,
                              title1='Range Label',
                              img_norms=((0.0, 1.0), norm, norm))

                    # save model
                    dl_utils.save_model_params(i, net, param_dir)

                    # save results
                    testset.save_batch(
                        hp['out'], data[-1], disp.cpu().numpy(), uncert.cpu().numpy(), runtime)


                i += 1

        except (KeyboardInterrupt, SystemExit):
            break

    plot.exit()
    return 0


if __name__ == '__main__':
    sys.exit(main())
