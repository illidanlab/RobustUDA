# Copyright(c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import numpy as np
import os
import pickle
import scipy.stats
from scipy.stats import wasserstein_distance

def distance_metric(test_label_distribution, target_label_distribution):

    kld2 = scipy.stats.entropy(test_label_distribution, target_label_distribution)
    print("kld test to target : {}".format(kld2))

    mixture_2 = (test_label_distribution + target_label_distribution) / 2
    jsd_2 = (scipy.stats.entropy(test_label_distribution, qk=mixture_2)
           + scipy.stats.entropy(target_label_distribution, qk=mixture_2)) / 2
    print("JSD test to target : {}".format(jsd_2))

    w_distance2 = wasserstein_distance(test_label_distribution, target_label_distribution)
    print("w_distance test to target : {}".format(w_distance2))

    BC = np.sum(np.sqrt(test_label_distribution*target_label_distribution))
    h = np.sqrt(1 - BC)
    print("Hellinger distance test to target : {}".format(h))

    # Bhattacharyya distance：
    b = -np.log(BC)
    print("Bhattacharyya distance test to target : {}".format(b))









