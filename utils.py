import os
import torch


def print_progress(percent, bar_len=None):
    """
    Print a progress bar to stdout

    :param percent: progress
    :type percent: float

    :param bar_len: (optional) length of the bar, commandline length if None
    :tyoe bar_len: int
    """
    if bar_len is None:
        bar_len = os.get_terminal_size().columns - 10

    percent /= 100.0
    progress = ''
    for i in range(bar_len):
        if i < int(bar_len * percent):
            progress += "="
        else:
            progress += " "
    print('\r[%s] %.2f%%' % (progress, percent * 100), end='')

    if percent >= 1.0:
        print('')


def print_img(img):
    """
    Helper function showing an image in the terminal

    :param img: the image
    :type img: numpy.ndarray with shape (h, w)
    """
    from skimage import img_as_ubyte
    from skimage.transform import rescale

    # normalize between 0 and 1
    img -= np.min(img)
    img /= np.max(img)

    # convert grayscale to rgb
    if(len(img.shape) < 3 or img.shape[2] == 1):
        # convert from grayscale
        img = np.stack([img, img, img], axis=2)

    # rescale to terminal size
    width = os.get_terminal_size().columns
    scale = float(width) / float(img.shape[1])

    with warnings.catch_warnings(record=True) as caught_warnings:
        img = rescale(img, scale, mode='constant', multichannel=True)

        # convert img to uint
        img = img_as_ubyte(img)

    output = ''
    for y in range(0, img.shape[0] - 1, 2):
        for x in range(img.shape[1]):
            tr, tg, tb = img[y, x]
            br, bg, bb = img[y + 1, x]

            output += '\033[38;2;{};{};{}m\033[48;2;{};{};{}m\u2580\033[0m'.format(
                tr, tg, tb, br, bg, bb)

        output += '\n'

    print(output)
