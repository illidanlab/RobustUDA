import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
def plot1():
    block = [1, 10, 20, 30, 40]
    clean_acc = [63.97222, 64.41667, 63.94444, 62.72222, 62]
    poison_acc = [10.66667, 35.68056, 49.18056, 53.91667, 55.76389]
    attack_rate = [87.31944, 55.65278, 33.54167, 23.56944, 19.27778]
    plt.plot(block, clean_acc, label='Clean acc', marker="*", linewidth=3)
    plt.plot(block, poison_acc, label='Poison acc', marker="*", linewidth=3)
    plt.plot(block, attack_rate, label='Success rate', marker="*", linewidth=3)

    plt.xlabel('Block size', fontsize=15)
    plt.ylabel('Accuravy(%)', fontsize=15)
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.legend(fontsize=15)
    plt.show()
def plot2():
    block = [1, 5,10, 20, 30, 40]
    clean_acc = [68.1976, 67.89773, 67.8346, 65.23043, 64.17298, 63.98359]
    poison_acc = [12.24747, 35.03788, 41.01957, 46.10164, 47.39583, 48.97412]
    attack_rate = [84.01199, 52.99874, 42.7399, 33.44381, 29.11932, 25.96275]
    plt.plot(block, clean_acc, label='Clean acc', marker="*", linewidth=3)
    plt.plot(block, poison_acc, label='Poison acc', marker="*", linewidth=3)
    plt.plot(block, attack_rate, label='Success rate', marker="*", linewidth=3)

    plt.xlabel('Block size', fontsize=15)
    plt.ylabel('Accuravy(%)', fontsize=15)
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.legend(fontsize=15)
    plt.show()
#DANN badnet
def plot3():
    block = [1,10, 20, 30, 40]
    clean_acc = [67.88, 65.93, 63.74, 63.49, 62.83]
    poison_acc = [8.75,  29.61, 47.53, 54.43, 55.71]
    attack_rate = [95.38, 67.14, 37.51, 24.00, 20.83]
    plt.plot(block, clean_acc, label='Clean acc', marker="*", linewidth = 3)
    plt.plot(block, poison_acc, label='Poison acc', marker="*", linewidth = 3)
    plt.plot(block, attack_rate, label='Success rate', marker="*", linewidth = 3)

    plt.xlabel('Block size', fontsize=15)
    plt.ylabel('Accuravy(%)', fontsize=15)
    plt.yticks([10,20,30,40,50,60,70,80,90,100])
    plt.legend(fontsize=15)
    plt.show()
def plot4():
    block = [1,10, 20, 30, 40]
    clean_acc = [69.19, 66.95076, 65.48, 65.01, 63.70]
    poison_acc = [13.30, 39.50, 43.88, 47.29, 49.10]
    attack_rate = [82.09, 46.09, 36.76, 30.76, 25.71]
    plt.plot(block, clean_acc, label='Clean acc', marker="*", linewidth=3)
    plt.plot(block, poison_acc, label='Poison acc', marker="*", linewidth=3)
    plt.plot(block, attack_rate, label='Success rate', marker="*", linewidth=3)

    plt.xlabel('Block size', fontsize=15)
    plt.ylabel('Accuravy(%)', fontsize=15)
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.legend(fontsize=15)
    plt.show()

def plot_bar():
    #block 1
    ratio = np.array([0.01, 0.02, 0.03])
    clean_acc = [88.79, 89.34, 88.44]
    poison_acc = [8.57,8.77, 8.62]
    attack_rate = [92.33, 97.06, 95.17]
    clean_acc1 = [86.25, 85.00, 85.70]
    poison_acc1 = [12.21, 9.22, 8.77]
    attack_rate1 = [8.77, 59.99, 81.56]


    bar_width = 0.004
    x_major_locator = MultipleLocator(0.01)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.bar(ratio-bar_width/2, clean_acc, bar_width, align="center", label="block number = 1")
    plt.bar(ratio+bar_width/2, clean_acc1, bar_width, align="center", label="block number = 10")
    plt.ylabel("Clean test accuracy (%)", fontsize=15)
    plt.xlabel("Poison ratio", fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    plt.close()
    x_major_locator = MultipleLocator(0.01)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.bar(ratio - bar_width / 2, poison_acc, bar_width, align="center", label="block number = 1")
    plt.bar(ratio + bar_width / 2, poison_acc1, bar_width, align="center", label="block number = 10")
    plt.ylabel("Poison test accuracy (%)", fontsize=15)
    plt.xlabel("Poison ratio", fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    plt.close()
    x_major_locator = MultipleLocator(0.01)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.bar(ratio - bar_width / 2, attack_rate, bar_width, align="center", label="block number = 1")
    plt.bar(ratio + bar_width / 2, attack_rate1, bar_width, align="center", label="block number = 10")
    plt.ylabel("Attack success rate (%)", fontsize=15)
    plt.xlabel("Poison ratio", fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    plt.close()



plot_bar()

