import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

file = '../results_main.txt'
labels = []
values = []

file_variable = open(file)
all_lines_variable = file_variable.readlines()
# print(type(all_lines_variable))
# print(len(all_lines_variable))
for line in all_lines_variable:
    n = 0
    if line.startswith('----- '):
        file_line = line.split(' ')
        net_name = file_line[1]
        labels.append(net_name)
        n += 1

    if line.startswith(labels[n-1] + ' top 1 accuracy: '):
        accuracy_line = line.split(' ')
        accuracy_value = accuracy_line[len(accuracy_line) - 1]
        accuracy_value.replace('\n', '')
        value = float(accuracy_value)
        values.append(value)

print(labels)
print(values)


# y_pos = np.arange(len(labels))

# fig, ax = plt.subplots()

# hbars = ax.barh(y_pos, values, align='center')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(labels)
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Accuracy (%)')
# ax.set_title('Imagenet 2012 Dataset Top 1 accuracy')

# ax.set_xlim(right=100)  # adjust xlim to fit labels

# plt.tight_layout()
# plt.show()
# plt.savefig('../../Accuracy', dpi=150)