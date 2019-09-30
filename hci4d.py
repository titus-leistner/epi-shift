from torch.utils.data.dataset import Dataset
from torchvision import transforms

import numpy as np

import os
import random
import copy

import pfm
import utils
import dl_utils
import lf_utils


class HCI4D(Dataset):
    """
    A class for the synthetic HCI 4D Light Field Dataset

    http://hci-lightfield.iwr.uni-heidelberg.de/
    """

    def __init__(self, root, nviews=(9, 9), transform=None, cache=False, length=0):
        """
        Loads the dataset

        :param root: root folder e.g. training in

              training
              |
              |---boxes
              |   |
              |   |---input_Cam00.png
              |   |---input_Cam01.png
              |   |---...
              |   |---gt_disp_lowres.pfm
              |---cotton
              |---...

        :type root: str

        :param nviews: number of views as (w_views, h_views) (default (9, 9))
        :type nviews: tuple(int, int)

        :param transform: optional transform to be applied
        :type transform: callable

        :param cache: cache all scenes to RAM first?
        :param cache: bool

        :param length: predefined length or 0 to use real length
        :type length: int
        """
        self.name = os.path.basename(root)
        self.scenes_names = [f.name for f in os.scandir(root) if f.is_dir()]
        self.scenes = [f.path for f in os.scandir(root) if f.is_dir()]
        self.nviews = nviews
        self.transform = transform
        self.length = length

        self.cache = cache
        if cache:
            self.data = []
            self.cache_scenes()

    def load_scene(self, index):
        """
        Loads one scene

        :param index: scene index in range(0, len(dataset))
        :type index: int
        """
        import skimage.io

        scene = self.scenes[index]
        files = [f.name for f in os.scandir(scene)]
        imgs = [f for f in files if (f.endswith('.png') or f.endswith(
            '.jpg') or f.endswith('.jpeg')) and 'normals' not in f and
            'mask' not in f and 'objectids' not in f and 'unused' not in f]
        imgs.sort()

        # compute indices of cross setup
        w, h = self.nviews
        us = [int(h / 2) * w + i for i in range(h)]
        vs = [int(w / 2) + w * i for i in range(h)]

        h_views = []
        for i in us:
            substr = str(i).zfill(3)
            fname = imgs[i]
            fname = os.path.join(scene, fname)
            h_views.append(skimage.img_as_float(
                skimage.io.imread(fname)).astype(np.float32))
        h_views = np.stack(h_views)
        h_views = h_views.transpose((0, 3, 1, 2))

        v_views = []
        for i in vs:
            substr = str(i).zfill(3)
            fname = imgs[i]
            fname = os.path.join(scene, fname)
            v_views.append(skimage.img_as_float(
                skimage.io.imread(fname)).astype(np.float32))
        v_views = np.stack(v_views)
        v_views = v_views.transpose((0, 3, 1, 2))

        # extract center view
        center = v_views[int(h / 2)].copy()

        # try to find the ground truth disparity pfm file
        pfms = [f for f in files if f.endswith('.pfm')]

        if len(pfms) > 1:
            # only load files with 'disp' in the name
            pfms = [f for f in pfms if 'disp' in f]
        if len(pfms) > 1:
            # only load lowres file
            pfms = [f for f in pfms if 'lowres' in f]
        if len(pfms) > 1:
            # only load center view
            pfms = [f for f in pfms if str(us[int(w / 2)]).zfill(3) in f]

        # load ground truth disparity
        gt = np.zeros_like(center[0])
        if len(pfms) > 0:
            gt = pfm.load(os.path.join(scene, pfms[0]))
            gt = np.flip(gt, 0).copy()

        index = np.atleast_1d(index)

        return h_views, v_views, center, gt, index

    def cache_scenes(self):
        """
        Loads all scenes to RAM
        """
        print('Caching dataset "{}"...'.format(self.name))
        for i, scene in enumerate(self.scenes):
            self.data.append(self.load_scene(i))

            try:
                utils.print_progress((i + 1) / len(self.scenes) * 100.0)
            except Exception:
                pass

    def __len__(self):
        # TODO: dirty hack to create bigger batches after augmentation
        # return 4096
        if self.length == 0:
            return len(self.scenes)

        return self.length

    def __getitem__(self, index):
        """
        Loads the next scene and returns it as
        (h_views, v_views, center, gt, index)
        where the views are tensors of shape (w or h, 3, h_image, w_image),
        center is the center view and gt is the ground truth
        of shape (h_img, w_img) or zeroes if the dataset does not provide it.
        Index is just a scalar list index as numpy.ndarray.

        :param index: scene index in range(0, len(dataset))
        :type index: int
        """
        index = index % len(self.scenes)

        data = None
        if self.cache:
            data = self.data[index]
        else:
            data = self.load_scene(index)

        if self.transform:
            data = copy.deepcopy(data)
            data = self.transform(data)

        #print(np.min(data[3]), np.max(data[3]))
        return data

    def save_batch(self, path, index, result=None, uncert=None, runtime=None):
        """
        Save the scene batch, ground truth and result to disk.
        Creates one one subdirectory in 'scenes/' for each scene in the batch.
        The results can be saved to 'ours/disp_maps/scene.pfm'.
        The runtime can also be saved to 'ours/runtimes/scene.txt'.

        :param path: the path to save scenes to
        :type path: str

        :param index: indices of the batch
        :type index: np.ndarray of shape (b, 1)

        :param result: batch of results
        :type result: np.ndarray of shape (b, h, w)

        :param uncert: batch of uncertainties
        :type uncert: np.ndarray of shape (b, h, w)

        :param runtime: runtime for the batch
        :type runtime: float
        """
        # create directories
        scenes = os.path.join(path, 'scenes')
        ours = os.path.join(path, 'ours')

        if not os.path.exists(scenes):
            os.makedirs(scenes)

        if not os.path.exists(ours):
            os.makedirs(ours)

        disp_maps = os.path.join(ours, 'disp_maps')
        if not os.path.exists(disp_maps):
            os.makedirs(disp_maps)

        runtimes = os.path.join(ours, 'runtimes')
        if not os.path.exists(runtimes):
            os.makedirs(runtimes)

        # for each scene
        for arr_i, i in enumerate(index.squeeze(1).tolist()):
            i = int(i)
            scene = self.scenes_names[i]

            # create directory
            scene_dir = os.path.join(scenes, scene)

            # get scene images
            h_views, v_views, center, gt, _ = self.__getitem__(i)

            lf_utils.save_views(scene_dir, h_views, v_views)
            dl_utils.save_img(os.path.join(scene_dir, 'center.png'), center)
            dl_utils.save_img(os.path.join(scene_dir, 'gt.png'), gt)

            # save results, uncertainties and/or runtimes
            if result is not None:
                dl_utils.save_img(os.path.join(
                    scene_dir, 'result.png'), result[arr_i])

                res_out = np.flip(result[arr_i], 0)
                pfm.save(os.path.join(scene_dir, 'result.pfm'), res_out)
                pfm.save(os.path.join(disp_maps, f'{scene}.pfm'), res_out)

            if uncert is not None:
                dl_utils.save_img(os.path.join(
                    scene_dir, 'uncert.png'), uncert[arr_i])

            if runtime is not None:
                # devide runtime by batchsize and output
                b = float(index.shape[0])
                with open(os.path.join(runtimes, f'{scene}.txt'), 'w') as f:
                    f.write(str(runtime / b))


class Zoom:
    """
    Rescale the input lighfield according to some factor
    """

    def __init__(self, factor):
        """
        :param factor: desired zoom factor (e.g. 0.5 for half the image size) 
        :type factor: float
        """
        assert isinstance(factor, float)
        self.factor = factor

    def __call__(self, data):
        """
        Rescale the lightfield data.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
                     or any other sequence of images or image stacks
        :type data: tuple

        :returns: the scaled lightfield data
        """
        from scipy import ndimage

        data = list(data)
        for i in range(len(data)):
            shape = data[i].shape
            if len(shape) < 2 or shape[-1] <= 1 or shape[-2] <= 1:
                continue
            zoom = [1.0] * len(data[i].shape)
            zoom[-2] = zoom[-1] = self.factor
            data[i] = ndimage.zoom(data[i], zoom, order=0)

        # correct ground truth
        if len(data) > 3:
            data[3] *= float(self.factor)

        return tuple(data)


class RandomZoom:
    """
    Rescale the lightfield randomly
    """

    def __init__(self, min_scale=0.5, max_scale=1.0):
        """
        :param min_scale: minumum possible scale
        :type min_scale: float

        :param max_scale: minumum possible scale
        :type max_scale: float
        """
        self.interval = (min_scale, max_scale)

    def __call__(self, data):
        factor = random.uniform(self.interval[0], self.interval[1])

        zoom = Zoom(factor)

        return zoom(data)


class Crop:
    """
    Crop the input lightfield to a given size with a given position
    """

    def __init__(self, size, pos):
        """
        :param size: output size. Tuple (height, width) or int for square size
        :type size: tuple(h, w) or int

        :param pos: crop position(s) tuple (y, x)
        :type pos: tuple(y, x)
        """
        assert isinstance(size, int) or (
            isinstance(size, tuple) and len(size) == 2)
        assert isinstance(pos, tuple)

        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

        self.pos = pos

    def __call__(self, data):
        """
        Crop the lightfield data.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
                     or any other sequence of images or image stacks
        :type data: tuple

        :returns: the cropped lightfield data
        """
        data = list(data)
        h, w = self.size
        y, x = self.pos

        for i in range(len(data)):
            shape = data[i].shape
            if len(shape) < 2 or shape[-1] <= 1 or shape[-2] <= 1:
                continue

            data[i] = data[i][..., y:y+h, x:x+w]

        return tuple(data)


class CenterCrop:
    """
    Crop by cutting off equal margins input lightfield to a given size
    """

    def __init__(self, size):
        """
        :param size: output size. Tuple (height, width) or int for square size
        :type size: tuple(h, w) or int
        """
        assert isinstance(size, int) or (
            isinstance(size, tuple) and len(size) == 2)

        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

    def __call__(self, data):
        """
        Crop the lightfield data.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
                     or any other sequence of images or image stacks
        :type data: tuple

        :returns: the cropped lightfield data
        """
        h = data[0].shape[-2]
        w = data[0].shape[-1]

        y = int((h - self.size[0]) / 2)
        x = int((w - self.size[1]) / 2)

        assert y >= 0 and x >= 0

        crop = Crop(self.size, (y, x))

        return crop(data)


class RandomCrop:
    """
    Crop patches randomly to a given size
    """

    def __init__(self, size, pad=0):
        """
        :param size: output size. Tuple (height, width) or int for square size
        :type size: tuple(h, w) or int

        :param pad: optional padding to not choose samples from
        :tyoe pad: int
        """
        assert isinstance(size, int) or (
            isinstance(size, tuple) and len(size) == 2)

        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

        assert isinstance(pad, int)
        self.pad = pad

    def __call__(self, data):
        """
        Crop the lightfield data.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
                     or any other sequence of images or image stacks
        :type data: tuple

        :returns: the cropped lightfield data
        """
        h = data[0].shape[-2]
        w = data[0].shape[-1]

        assert h > self.size[0]
        assert w > self.size[1]

        y = random.randint(self.pad, h - self.size[0] - self.pad)
        x = random.randint(self.pad, w - self.size[1] - self.pad)

        crop = Crop(self.size, (y, x))

        return crop(data)


class RedistColor:
    """
    Randomly redistribute color
    """

    def __call__(self, data):
        """
        Redistribute the lightfield color data.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the recolored lightfield data
        """
        # create redistribution matrix
        mat = np.zeros((3, 3))
        mat[0, 0] = random.uniform(0.0, 1.0)
        mat[0, 1] = random.uniform(0.0, 1.0 - mat[0, 0])
        mat[1, 0] = random.uniform(0.0, 1.0 - mat[0, 0])
        mat[1, 1] = random.uniform(0.0, 1.0 - max(mat[0, 1], mat[1, 0]))

        mat[0, 2] = 1.0 - mat[0, 0] - mat[0, 1]
        mat[1, 2] = 1.0 - mat[1, 0] - mat[1, 1]
        mat[2, 0] = 1.0 - mat[0, 0] - mat[1, 0]
        mat[2, 1] = 1.0 - mat[0, 1] - mat[1, 1]
        mat[2, 2] = mat[0, 0] + mat[0, 1] + mat[1, 0] + mat[1, 1] - 1.0

        for i in range(min(3, len(data))):
            stack = None
            if isinstance(data[i], np.ndarray):
                stack = data[i].copy()
            else:
                stack = data[i].clone()

            assert stack.shape[-3] == 3

            data[i][..., 0, :, :] = mat[0, 0] * stack[..., 0, :, :]
            data[i][..., 0, :, :] += mat[0, 1] * stack[..., 1, :, :]
            data[i][..., 0, :, :] += mat[0, 2] * stack[..., 2, :, :]

            data[i][..., 1, :, :] = mat[1, 0] * stack[..., 0, :, :]
            data[i][..., 1, :, :] += mat[1, 1] * stack[..., 1, :, :]
            data[i][..., 1, :, :] += mat[1, 2] * stack[..., 2, :, :]

            data[i][..., 2, :, :] = mat[2, 0] * stack[..., 0, :, :]
            data[i][..., 2, :, :] += mat[2, 1] * stack[..., 1, :, :]
            data[i][..., 2, :, :] += mat[2, 2] * stack[..., 2, :, :]

        return tuple(data)


class Contrast:
    """
    Randomly change Contrast
    """

    def __init__(self, level=0.9):
        """
        :param level: level of change
        :type level: float
        """
        assert isinstance(level, float)
        self.level = level

    def __call__(self, data):
        """
        Change the lightfields contrast.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the recolored lightfield data
        """
        alpha = random.uniform(-self.level, self.level) + 1.0
        mean = data[0].mean()

        data = list(data)
        for i in range(min(3, len(data))):
            data[i] = data[i] * alpha + mean * (1.0 - alpha)

        return tuple(data)


class Brightness:
    """
    Randomly change Brightness
    """

    def __init__(self, level=0.9):
        """
        :param level: level of change
        :type level: float
        """
        assert isinstance(level, float)
        self.level = level

    def __call__(self, data):
        """
        Change the lightfields brightness.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the recolored lightfield data
        """
        alpha = random.uniform(-self.level, self.level) + 1.0

        data = list(data)
        for i in range(min(3, len(data))):
            data[i] = data[i] * alpha

        return tuple(data)


class Noise:
    """
    Add random Gaussian noise
    """

    def __init__(self, stdev=0.01):
        """
        :param stdev: standard deviation of noise
        :type stdev: float
        """
        assert isinstance(stdev, float)
        self.stdev = stdev

    def __call__(self, data):
        """
        Add random Gaussian noise.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the recolored lightfield data
        """
        data = list(data)
        for i in range(min(3, len(data))):
            noise = np.random.normal(scale=self.stdev, size=data[i].shape)
            data[i] += noise

        return tuple(data)


class Shift:
    """
    Shift the lightfield according to some disparity
    """

    def __init__(self, disp):
        """
        :param disp: discrete disparity that should be zero afterwards
        :type disp: int
        """
        assert isinstance(disp, int)
        self.disp = disp

    def __call__(self, data):
        """
        Shift the lightfield

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the shifted lightfield data
        """
        data = list(data)
        # test if numpy or pytorch
        cat = np.concatenate
        if not isinstance(data[0], np.ndarray):
            from torch import cat as torch_cat
            cat = torch_cat

        h_views = data[0]
        v_views = data[1]

        w = h_views.shape[-4]
        h = v_views.shape[-4]
        hw = int(w / 2)
        hh = int(h / 2)

        l = h_views.shape[-1]
        for i in range(w):
            shift = self.disp * (i - hw)
            h_views[..., i, :, :, :] = cat(
                [h_views[..., i, :, :, -shift:],
                 h_views[..., i, :, :, :-shift]], -1)

        l = v_views.shape[-2]
        for i in range(h):
            shift = self.disp * (i - hh)
            v_views[..., i, :, :, :] = cat(
                [v_views[..., i, :, -shift:, :],
                 v_views[..., i, :, :-shift, :]], -2)

        # correct ground truth
        if len(data) > 3:
            data[3] -= float(self.disp)

        return tuple(data)


class RandomShift:
    """
    Randomly shift the lightfield an correct the ground truth accordingly
    """

    def __init__(self, disp_range):
        """
        :param disp_range: interval of disparities for shifts. tuple(min, max)
        or positive int for a range of (-disp_range, +disp_range)
        :type disp_range: tuple(int, int) or int
        """
        assert isinstance(disp_range, int) or (
            isinstance(disp_range, tuple) and len(disp_range) == 2)

        self.disp_range = disp_range

        if isinstance(disp_range, int):
            assert disp_range > 0
            self.disp_range = (-disp_range, disp_range)

    def __call__(self, data):
        """
        Shift the lightfield randomly and correct the ground truth accordingly

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the shifted lightfield data
        """
        # shift randomly
        disp = random.randint(self.disp_range[0], self.disp_range[1])

        shift = Shift(disp)
        data = shift(data)

        return data


class Rotate90:
    """
    Rotate the lightfield by 90 degrees
    """

    def __init__(self):
        pass

    def __call__(self, data):
        """
        Rotate the lightfield by 90 degrees

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the rotated lightfield data
        """
        view = np.transpose
        flip = np.flip
        if not isinstance(data[0], np.ndarray):
            from torch import flip as torch_flip
            def view(t, s): return t.permute(s)
            def flip(t, a): return torch_flip(t, (a,))

        data = list(data)

        for i in range(min(4, len(data))):
            axis = list(range(len(data[i].shape)))
            axis[-1], axis[-2] = axis[-2], axis[-1]
            data[i] = flip(view(data[i], axis), -2).copy()

        if len(data) > 1:
            data[0], data[1] = data[1], data[0]
            data[1] = flip(data[1], -4)

        return tuple(data)


class RandomRotate:
    """
    Rotate the lightfield by 90 degrees
    """

    def __init__(self):
        self.rot = Rotate90()

    def __call__(self, data):
        r = random.randint(0, 3)

        for i in range(r):
            data = self.rot(data)

        return data
