#!/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

font = {'family': 'normal', 'size' : 26}
matplotlib.rc('font', **font)


def read_data(name):
    result = []

    with open(name) as file:
        for line in file.readlines():
            result.append(float(line))

    return result


fig, ax = plt.subplots(1)
fig.set_size_inches(22.5, 10.5)
ax.tick_params(color='#D8DEE9', labelcolor='#D8DEE9')
ax.xaxis.label.set_color('#D8DEE9')
ax.yaxis.label.set_color('#D8DEE9')

for spine in ax.spines.values():
    spine.set_edgecolor('#D8DEE9')


for fn in [
        #["volatile_global_block_sleep_single_gpu", "#BF616A"],
        ["volatile_global_multi_gpu", "#81A1C1"],
        ["volatile_global_block_multi_gpu", "#BF616A"],
        ["volatile_global_block_sleep_multi_gpu", "#A3BE8C"]]:
    data = read_data(fn[0])

    # Draw the density plot
    sns.distplot(data, hist=False, kde=True,
                 kde_kws={'linewidth': 3}, rug=True,
                 color=fn[1],
                 label=fn[0])

# Plot formatting
plt.legend(prop={'size': 16}, title='Airline')
plt.xlabel('Elapsed [ms]')
plt.ylabel('Density')
plt.ylim(0,185)
plt.xlim(0.039,0.11)

l = plt.legend(facecolor='white', framealpha=0.1)
for text in l.get_texts():
    text.set_color('#D8DEE9')

#plt.show()
fig.savefig('multi_gpu_density.pdf', transparent=True, dpi=300)
