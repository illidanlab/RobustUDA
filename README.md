# Robust_UDA
The code is for paper "Robust Unsupervised Domain Adaptation from A Corrupted Source".

We conduct two kind of unsuerpvised domain ataptation (UDA) tasks:
digits tasks (USPS <-> MNIST), and image tasks (CIFAR10 <-> STL)

We utilize two kinds of backdoor attacks to corrupt the source training data:
BadNet and CLBD
## Arguments
we provide a detailed description on key arguments.
| Arguments   | Discription  |
|  ----  | ----  |
|method| UDA method, such as DANN, CDAN|
|task| the argument is only for digits domain adaptation tasks, including minist2usps and usps2mnist|
|corrupt| corruption method for source data|
|poison_ratio| The ratio of poison samples|
|block| the number of blocks for MoM algorithm, if block=1, no MoM algorithm is applied|
|cls_par| this argument is for fine-tune, if cls_par = 0.0, only IM is used, if cls_par > 0, both IM and PL are used|
|s_dset, t_dset|these two arguments are only for image tasks, and s_dset indicates the source domain, while t_dset indicates the target domain|
|poison| binary arguments, 1 for poison, and 0 for clean|

## Sample command
To conduct the MoM method for digits tasks:

``python digits_MOM_poison.py CDAN --block 15 --poison 1 --poison_ratio 0.02 --corrupt badnet``

To conduct the MoM method for digits tasks with fine-tune:

``python digits_MOM_SHOT.py CDAN --block 15 --poison 1 --poison_ratio 0.02 --corrupt badnet --cls_par 0.0``

To conduct the MoM method for image tasks:

``python cifar_MOM_poison.py CDAN --corrupt clbd --block 10 --poison_ratio 0.5 --num_iterations 20000``

To conduct the MoM method for image tasks with fine-tune:

``python cifar_MOM_SHOT.py DANN --corrupt clbd --block 10 --cls_par 0.0 --poison_ratio 0.5 --num_iterations 20000``

