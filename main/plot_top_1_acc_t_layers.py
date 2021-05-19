import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# file = '../results_extra_samples.txt'
# labels = []
# values = []
# ordered_labels = []
# ordered_values = []

# file_variable = open(file)
# all_lines_variable = file_variable.readlines()
# print(type(all_lines_variable))
# print(len(all_lines_variable))

colors = ("red", "green", "blue", "orange", "purple", "pink", "olive", "brown", "cyan")
groups = ("MobileNet", "ResNet50", "SqueezeNet", "VGG16", "AlexNet", "GoogLeNet", "DenseNet121", "InceptionV3", "ShuffleNet")

x = (4495, 3032, 1030, 4802, 663, 2223, 2693, 3660, 538)
y = (71.8, 76.1, 58, 71.5, 56.5, 69.7, 74.4, 77.2, 69.3)
layers = (53, 50, 18, 16, 8, 22, 121, 42, 50)

fig, ax = plt.subplots()
for x, y, color, group, layer in zip(x, y, colors, groups, layers):
    n = 750
    ax.scatter(x, y, c=color, s=9, label=group, linewidths=(layer/8))

ax.legend()
ax.grid(True)

plt.xlabel("time (s)")
plt.ylabel("top-1 acc (%)")
plt.ylim([0, 100])
plt.xlim([0, 5000])
plt.show()
plt.savefig('../../top_1_acc_t_layers', dpi=150)