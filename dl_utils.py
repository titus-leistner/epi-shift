import torch
import numpy as np

import os
import datetime
import json
import warnings

def save_hyper_params(hp, param_dir):
    """
    Output the hyper parameters as JSON-file to
    param_dir/hyper_params.json

    :param hp: hyper-parameters as dictionary
    :type hp: dict

    :param param_dir: directory to save the json file to
    :type param_dir: str
    """
    f = open(os.path.join(param_dir, 'hyper_params.json'), 'w')
    json.dump(hp, f, ensure_ascii=False, indent=4)
    f.close()


def load_model_params(net, params):
    """
    Load model parameters from .pt file

    :param net: the network
    :type net: torch.nn.Module

    :param params: the filename of the parameter file
    :type params: str
    """
    print('Loading parameters from "{}"...'.format(params))
    loaded = torch.load(params)
    old = loaded.state_dict()

    if hasattr(loaded, 'module'):
        old = loaded.module.state_dict()

    new = net.state_dict()
    new = {k: new[k] for k in new.keys() if k not in old.keys()}
    new.update(old)

    net.load_state_dict(new)
    for p in net.parameters():
        p.requires_grad = False
        p[torch.isnan(p)] = 0.0
        p.requires_grad = True


def save_model_params(i, net, param_dir):
    """
    Output the model parameters to param_dir/params_i.pt

    :param i: training iteration
    :type i: int

    :param net: the network model
    :type net: torch.nn.Module

    :param param_dir: directory to save the pt-file to
    :type param_dir: str
    """
    torch.save(net, os.path.join(param_dir, 'params_%04d.pt' % i))


def save_img(fname, arr):
    """
    Save the numpy-array as image with the given filename

    :param fname: the filename to save to
    :type fname: str

    :param arr: the image
    :type arr: numpy.ndarray with shape (3, h, w) -> rgb or (h, w) -> greyscale
    """
    import skimage.io

    # convert, if necessary
    if not isinstance(arr, np.ndarray):
        arr = arr.detach().cpu().numpy()

    # normalize, if necessary
    a_min = np.min(arr)
    a_max = np.max(arr)

    if(a_min < 0.0 or a_max > 1.0):
        arr = (arr - a_min) / (a_max - a_min)

    if len(arr.shape) == 3:
        arr = np.transpose(arr, (1, 2, 0))

    # save image
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        skimage.io.imsave(fname, arr)


class BatchIter:
    """
    Process each image in a batch iteratively
    """

    def __init__(self, net):
        """
        :param net: the pytorch network
        :type net: torch.nn.Module

        :param args: tuple with arguments
        :type args: tuple
        """
        assert(isinstance(net, torch.nn.Module))

        self.net = net

    def __call__(self, *args):
        """
        Run the network once for each image in the batch

        :param args: tuple with arguments of type torch.tensor
        :type args: tuple
        """
        for arg in args:
            assert(isinstance(arg, torch.Tensor))
            
        
        results = []
        
        b = args[0].shape[0]
        for i in range(b):
            net_args = []
            for arg in args:
                net_args.append(arg[i:i+1])

            results.append(self.net(*net_args))
        
        out = []
        for j in range(len(results[0])):
            tensor = []
            for i in range(b):
                tensor.append(results[i][j])
            tensor = torch.cat(tensor, 0)
        
            out.append(tensor) 

        return out
