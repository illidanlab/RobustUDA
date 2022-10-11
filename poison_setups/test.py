import pickle
from torchvision import datasets, transforms

with open("cifar10_transfer_learning.pickle", "rb") as handle:
    setup_dicts = pickle.load(handle)

# Which set up to do in this run?
setup = setup_dicts[0]  # this can be changed to choose a different setup

# get set up for this trial
target_class = setup["target class"]
target_img_idx = setup["target index"]
poisoned_label = setup["base class"]
base_indices = setup["base indices"]
print("indice 1", base_indices[1])
num_poisons = len(base_indices)
print('target_class', target_class)
print('target_img_idx', target_img_idx)
print('poisoned_label', poisoned_label)
print('base_indices', base_indices)
print('num_poisons', num_poisons)