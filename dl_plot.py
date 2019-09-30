import os
import shutil
import warnings
import datetime
import numpy as np
import matplotlib


def extract_params_pytorch(net):
    """
    Extract weights and biases from a pytorch network.
    Returns a tuple(weights, biases) of numpy arrays

    :param net: pytorch network
    :type net: torch.nn.Module

    Returns a tuple(weights, biases)
    """
    # extracting weights, biases
    weights = np.concatenate([p.cpu().data.numpy().flatten()
                              for n, p in net.named_parameters() if 'weight' in n])
    biases = np.concatenate([p.cpu().data.numpy().flatten()
                             for n, p in net.named_parameters() if 'bias' in n])
    return weights, biases


class Plot():
    """
    Helper class to plot training progress of neural networks
    """

    def __init__(self, path='', landscape=False, latex=False):
        """
        Helper class to plot training progress

        :param path: (optional) path for plots or '' for windowed mode
        :type path: str

        :param landscape: (optional) boolean landscape mode?
        :type landscape: bool

        :param latex: (optional) use latex for pdf plots?
        :type latex: bool
        """
        if path != '':
            matplotlib.use('Agg')

            # Activate LaTex plotting, if possible
            if latex and shutil.which('pdflatex') is not None and shutil.which('dvipng') is not None:
                matplotlib.rcParams['text.usetex'] = True
                matplotlib.rcParams['font.family'] = 'serif'
                matplotlib.rcParams['font.size'] = 12.0

        # initialize matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt

        # initialize members
        self.path = path
        self.loss_train = np.empty([0])
        self.loss_test = np.empty([0])

        self.perc_w = np.empty([0, 9])
        self.perc_b = np.empty([0, 9])

        self.steps = np.empty([0])

        self.index = 0

        # initialize matplotlib
        plt.rc('axes', axisbelow=True)
        plt.rc('axes', facecolor='lightgray')

        # setup rows and columns for plot
        if landscape:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12.8, 7.2))
            self.loss = axes[0, 0]
            self.ws = axes[0, 1]
            self.bs = axes[0, 2]

            self.img1 = axes[1, 0]
            self.img2 = axes[1, 1]
            self.img3 = axes[1, 2]

        else:
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7.2, 12.8))
            self.loss = axes[0, 0]
            self.ws = axes[1, 0]
            self.bs = axes[2, 0]

            self.img1 = axes[0, 1]
            self.img2 = axes[1, 1]
            self.img3 = axes[2, 1]

        fig.canvas.set_window_title('Training Progress')

        self.plot_loss()
        self.plot_ws()
        self.plot_bs()

        plt.tight_layout()
        if path == '':
            plt.show(block=False)
        else:
            # create directory for figure plots
            self.plot_dir = os.path.join(
                path, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            os.makedirs(self.plot_dir)

            # create progress csv
            with open(os.path.join(self.plot_dir, 'progress.csv'), 'w') as f:
                f.write(
                    'i, train loss, test loss, weights mean, weights stdev, biases mean, biases stdev\n')

    def percentiles(self, data):
        """
        Computes 9 percentiles of the data

        :param data: the distribution values
        :type data: numpy.ndarray
        """
        ps = np.empty([9])
        for i in range(9):
            ps[i] = np.percentile(data, 10.0 * i + 10.0)

        return ps

    def plot_loss(self):
        """
        Plot the loss axis
        """
        self.loss.clear()
        self.loss.set_title('Loss')
        self.loss.grid(linestyle='dotted', color='white')

        self.loss.plot(self.steps, self.loss_train, label='training loss',
                       color='red')
        self.loss.plot(self.steps, self.loss_test, label='test loss',
                       color='blue')
        self.loss.legend()

    def plot_ws(self):
        """
        Plot the weights axis
        """
        self.ws.clear()
        self.ws.set_title('Weight Distribution')
        self.ws.grid(linestyle='dotted', color='white')

        for i in range(5):
            self.ws.fill_between(self.steps, self.perc_w.T[i],
                                 self.perc_w.T[8 - i], color='red', alpha=0.2)

    def plot_bs(self):
        """
        Plot the biases axis
        """
        self.bs.clear()
        self.bs.set_title('Bias Distribution')
        self.bs.grid(linestyle='dotted', color='white')

        for i in range(5):
            self.bs.fill_between(self.steps, self.perc_b.T[i],
                                 self.perc_b.T[8 - i], color='red', alpha=0.2)

    def plot(self, i, loss_train, loss_test, weights, biases,
             img1=None, img2=None, img3=None,
             title1='Input', title2='Ground Truth', title3='Result',
             img_norms=(None, None, None)):
        """
        Add new values for a given iteration

        :param i: iteration
        :type i: int

        :param loss_train: new training loss
        :type loss_train: float

        :param loss_test: new test loss
        :type loss_test: float

        :param weights: weights
        :type weights: numpy.ndarray

        :param biases: biases
        :type biases: numpy.ndarray

        :param img1: (optional) first image
        :type img1: numpy.ndarray

        :param img2: (optional) second image
        :type img2: numpy.ndarray

        :param img3: (optional) third image
        :type img3: numpy.ndarray

        :param title1: (optional) title for first image
        :type title1: str

        :param title2: (optional) title for second image
        :type title2: str

        :param title3: (optional) title for third image
        :type title3: str

        img_norms:  (optional) tuple of 3 tuples (vmin, vmax) to norm imgs
                    or None values
        """
        # update members
        self.steps = np.append(self.steps, [i])
        self.loss_train = np.append(self.loss_train, [loss_train])
        self.loss_test = np.append(self.loss_test, [loss_test])
        self.perc_w = np.vstack([self.perc_w, self.percentiles(weights)])
        self.perc_b = np.vstack([self.perc_b, self.percentiles(biases)])

        # plot
        self.plot_loss()
        self.plot_ws()
        self.plot_bs()

        # init norms
        plt_norms = []
        for interval in img_norms:
            if interval is None:
                plt_norms.append(None)
            else:
                vmin, vmax = interval
                plt_norms.append(
                    matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))

        # show images
        self.img1.clear()
        self.img1.set_title(title1)
        if img1 is not None:
            self.img1.imshow(img1, norm=plt_norms[0])
            # print_img(img1)
            # print(title1 + '\n')

        self.img2.clear()
        self.img2.set_title(title2)
        if img2 is not None:
            self.img2.imshow(img2, norm=plt_norms[1])
            # print_img(img2)
            # print(title2 + '\n')

        self.img3.clear()
        self.img3.set_title(title3)
        if img3 is not None:
            self.img3.imshow(img3, norm=plt_norms[2])
            # print_img(img3)
            # print(title3 + '\n')

        self.plt.tight_layout()

        if self.path == '':
            self.plt.pause(0.05)
        else:
            try:
                self.plt.savefig(os.path.join(self.path, 'train_prog'))
                self.plt.savefig(os.path.join(
                    self.plot_dir, 'train_prog_' + format(self.index, '04d')))
                self.plt.savefig(os.path.join(
                    self.plot_dir, 'train_prog_' + format(self.index, '04d') + '.pdf'))
                self.plt.savefig(os.path.join(
                    self.plot_dir, 'train_prog_' + format(self.index, '04d') + '.svg'))
            except Exception as e:
                print('Error during plotting:')
                print(e)

        self.index += 1

        # print values
        print('Iteration %i' % (i))
        print('Loss:     train data: %f,    \ttest data: %f' %
              (loss_train, loss_test))
        print('Weights:  mean:       %f,    \tstdev:     %f' %
              (np.mean(weights), np.std(weights)))
        print('Biases:   mean:       %f,    \tstdev:     %f' %
              (np.mean(biases), np.std(biases)))
        print('-----------------------------------------------------------')

        # write to csv-file
        with open(os.path.join(self.plot_dir, 'progress.csv'), 'a') as f:
            f.write('{},{},{},{},{},{},{}\n'.format(i, loss_train, loss_test, np.mean(
                weights), np.std(weights), np.mean(biases), np.mean(biases)))

    def exit(self):
        """
        Keep plotting window open in window mode or render video in image mode
        """
        if self.path == '':
            self.plt.show()
        else:
            print('Rendering training video...')
            os.system('ffmpeg -framerate 15 -r 15 -i ' +
                      os.path.join(self.plot_dir, 'train_prog_%04d.png') +
                      ' -c:v libx264 -crf 0 -preset veryslow -c:a libmp3lame -b:a 320k ' +
                      os.path.join(self.plot_dir, 'train_prog.mp4'))
