import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

file = '../results_extra_samples.txt'
labels = []
values = []
ordered_labels = []
ordered_values = []

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

    if line.startswith(labels[n-1] + ' top 5 accuracy: '):
        accuracy_line = line.split(' ')
        accuracy_value = accuracy_line[len(accuracy_line) - 1]
        accuracy_value.replace('\n', '')
        value = float(accuracy_value)
        values.append(value)

def order_lists():
    for i in range(len(values)):
        max_index = values.index(max(values))
        ordered_values.append(values.pop(max_index))
        ordered_labels.append(labels.pop(max_index))

# print(labels)
# print(values)
order_lists()
# print(ordered_labels)
# print(ordered_values)

width = 0.8
y_pos = np.arange(len(ordered_labels))

fig, ax = plt.subplots()

rect = ax.barh(y_pos, ordered_values, width, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(ordered_labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Accuracy (%)')
ax.set_title('Personal Dataset Top 5 Accuracy (100 samples)')

ax.set_xlim(right=100)  # adjust xlim to fit labels

def autolabel(rects):
    for rect in rects:
        width = rect.get_width()
        ax.annotate('{0:.2f}'.format(width),
                    xy=(width - 5, rect.get_y() + rect.get_height() - 0.1),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rect)

plt.tight_layout()
plt.show()
plt.savefig('../../Accuracy_top_5_extra_samples', dpi=150)