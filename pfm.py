import numpy as np
import re
import sys


def load(filename):
    """
    Load a PFM file into a Numpy array. 

    :param filename: path to pfm file
    :type filename: str

    :returns: the image as numpy.ndarray
    Note that it will have
    a shape of H x W, not W x H. Returns a tuple containing the
    loaded image and the scale factor from the file.
    """
    color = None
    width = None
    height = None
    scale = None
    endian = None

    file = open(filename, 'rb')
    if not file:
        raise Exception('Unable to open ' + filename)

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape)


def save(filename, image, scale=1.0):
    """
    Save a Numpy array to a PFM file.

    :param filename: path to pfm file
    :type filename: str

    :param image: the image
    :type image: numpy.ndarray

    :param scale: (optional) pfm-internal scale parameter (default 1.0)
    :type scale: float
    """
    f = open(filename, 'wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    # greyscale
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    f.write(b'PF\n' if color else b'Pf\n')
    f.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    f.write(b'%f\n' % scale)

    image.tofile(f)
