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

x = (3.5, 25, 1.2, 138, 61, 6.6, 7.9, 27, 2.2)
y = (71.8, 76.1, 58, 71.5, 56.5, 69.7, 74.4, 77.2, 69.3)

x1 = (3.5, 25, 1.2, 138, 61, 6.6, 7.9, 27, 2.2)
y1 = (86, 84, 83, 86, 72, 82, 83, 90, 87)

fig, ax = plt.subplots()
for x, y, color, group in zip(x, y, colors, groups):
    n = 750
    ax.scatter(x, y, c=color, s=9, label=group, linewidths=3)

for x, y, color, group in zip(x1, y1, colors, groups):
    n = 750
    ax.scatter(x, y, c=color, s=9, linewidths=3)

ax.legend()
ax.grid(True)

plt.xlabel("params (M)")
plt.ylabel("top-1 acc (%)")
plt.ylim([0, 100])
plt.xlim([0, 140])
plt.show()
plt.savefig('../../top_1_acc_params', dpi=150)