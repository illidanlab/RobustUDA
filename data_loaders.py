from batchup.datasets import cifar10, stl
from skimage.transform import downscale_local_mean, resize
import numpy as np
def load_cifar10(range_01=False, val=False):
    #
    #
    # Load CIFAR-10 for adaptation with STL
    #
    #

    print('Loading CIFAR-10...')
    if val:
        d_cifar = cifar10.CIFAR10(n_val=5000)
    else:
        d_cifar = cifar10.CIFAR10(n_val=0)

    d_cifar.train_X = d_cifar.train_X[:]
    d_cifar.val_X = d_cifar.val_X[:]
    d_cifar.test_X = d_cifar.test_X[:]
    d_cifar.train_y = d_cifar.train_y[:]
    d_cifar.val_y = d_cifar.val_y[:]
    d_cifar.test_y = d_cifar.test_y[:]

    # Remap class indices so that the frog class (6) has an index of -1 as it does not appear int the STL dataset
    cls_mapping = np.array([0, 1, 2, 3, 4, 5, -1, 6, 7, 8])
    d_cifar.train_y = cls_mapping[d_cifar.train_y]
    d_cifar.val_y = cls_mapping[d_cifar.val_y]
    d_cifar.test_y = cls_mapping[d_cifar.test_y]

    # Remove all samples from skipped classes
    train_mask = d_cifar.train_y != -1
    val_mask = d_cifar.val_y != -1
    test_mask = d_cifar.test_y != -1

    d_cifar.train_X = d_cifar.train_X[train_mask]
    d_cifar.train_y = d_cifar.train_y[train_mask]
    d_cifar.val_X = d_cifar.val_X[val_mask]
    d_cifar.val_y = d_cifar.val_y[val_mask]
    d_cifar.test_X = d_cifar.test_X[test_mask]
    d_cifar.test_y = d_cifar.test_y[test_mask]

    if range_01:
        d_cifar.train_X = d_cifar.train_X * 2.0 - 1.0
        d_cifar.val_X = d_cifar.val_X * 2.0 - 1.0
        d_cifar.test_X = d_cifar.test_X * 2.0 - 1.0

    print('CIFAR-10: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
        d_cifar.train_X.shape, d_cifar.train_y.shape, d_cifar.val_X.shape, d_cifar.val_y.shape, d_cifar.test_X.shape,
        d_cifar.test_y.shape))

    print('CIFAR-10: train: X.min={}, X.max={}'.format(
        d_cifar.train_X.min(), d_cifar.train_X.max()))

    d_cifar.n_classes = 9

    return d_cifar


def load_stl(zero_centre=False, val=False):
    #
    #
    # Load STL for adaptation with CIFAR-10
    #
    #

    print('Loading STL...')
    if val:
        d_stl = stl.STL()
    else:
        d_stl = stl.STL(n_val_folds=0)

    d_stl.train_X = d_stl.train_X[:]
    d_stl.val_X = d_stl.val_X[:]
    d_stl.test_X = d_stl.test_X[:]
    d_stl.train_y = d_stl.train_y[:]
    d_stl.val_y = d_stl.val_y[:]
    d_stl.test_y = d_stl.test_y[:]

    # Remap class indices to match CIFAR-10:
    cls_mapping = np.array([0, 2, 1, 3, 4, 5, 6, -1, 7, 8])
    d_stl.train_y = cls_mapping[d_stl.train_y]
    d_stl.val_y = cls_mapping[d_stl.val_y]
    d_stl.test_y = cls_mapping[d_stl.test_y]

    d_stl.train_X = d_stl.train_X[:]
    d_stl.val_X = d_stl.val_X[:]
    d_stl.test_X = d_stl.test_X[:]

    # Remove all samples from class -1 (monkey) as it does not appear int the CIFAR-10 dataset
    train_mask = d_stl.train_y != -1
    val_mask = d_stl.val_y != -1
    test_mask = d_stl.test_y != -1

    d_stl.train_X = d_stl.train_X[train_mask]
    d_stl.train_y = d_stl.train_y[train_mask]
    d_stl.val_X = d_stl.val_X[val_mask]
    d_stl.val_y = d_stl.val_y[val_mask]
    d_stl.test_X = d_stl.test_X[test_mask]
    d_stl.test_y = d_stl.test_y[test_mask]

    # Downsample images from 96x96 to 32x32
    d_stl.train_X = downscale_local_mean(d_stl.train_X, (1, 1, 3, 3))
    d_stl.val_X = downscale_local_mean(d_stl.val_X, (1, 1, 3, 3))
    d_stl.test_X = downscale_local_mean(d_stl.test_X, (1, 1, 3, 3))

    if zero_centre:
        d_stl.train_X = d_stl.train_X * 2.0 - 1.0
        d_stl.val_X = d_stl.val_X * 2.0 - 1.0
        d_stl.test_X = d_stl.test_X * 2.0 - 1.0

    print('STL: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
        d_stl.train_X.shape, d_stl.train_y.shape, d_stl.val_X.shape, d_stl.val_y.shape, d_stl.test_X.shape,
        d_stl.test_y.shape))

    print('STL: train: X.min={}, X.max={}'.format(
        d_stl.train_X.min(), d_stl.train_X.max()))

    d_stl.n_classes = 9

    return d_stl