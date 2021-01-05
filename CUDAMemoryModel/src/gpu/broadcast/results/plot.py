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
        ["volatile_global_block_sleep", "#BF616A"],
        #["volatile_global", "#A3BE8C"],
        ["atomic_wait", "#EBCB8B"], # +
        ["atomic_wait_with_ret", "#81A1C1"], # +
        #["atomic_wait_with_ret_polling_count_1", "#A3BE8C"],
        ["atomic_wait_with_ret_polling_count_0", "#A3BE8C"]]: # +
        #["basic", "#EBCB8B"]]:
        #["volatile_global_block", "#81A1C1"]]:
        #["volatile_global_block_sleep", "#BF616A"]]:
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
plt.ylim(0,450)
plt.xlim(0.039,0.10)

l = plt.legend(facecolor='white', framealpha=0.1)
for text in l.get_texts():
    text.set_color('#D8DEE9')

#plt.show()
fig.savefig('density.pdf', transparent=True, dpi=300)
