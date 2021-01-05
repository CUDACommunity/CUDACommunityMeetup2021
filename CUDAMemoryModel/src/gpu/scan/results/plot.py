#!/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import csv

font = {'family': 'normal', 'size' : 26}
matplotlib.rc('font', **font)

def load(filename):
    x = []
    y = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            x.append(int(row[0]))
            y.append(float(row[1]))
    return (x, y)

fig, ax = plt.subplots(1)
ax.set_ylabel('Time [ms]')
ax.set_xlabel('Size of input array')
ax.tick_params(color='#D8DEE9', labelcolor='#D8DEE9')
for spine in ax.spines.values():
    spine.set_edgecolor('#D8DEE9')
fig.set_size_inches(22.5, 10.5)

for f in [['decoupled_fence_rtx3090', '#EBCB8B'], ['three_kernels_rtx3090', '#A3BE8C'], ['decoupled_rtx3090', '#81A1C1']]:
    x, y = load(f[0])
    plt.plot(x, y, color=f[1], label=f[0])

ax.xaxis.label.set_color('#D8DEE9')
ax.yaxis.label.set_color('#D8DEE9')
plt.ylim(0,0.07)
plt.grid(color='#D8DEE9')
l = plt.legend(facecolor='white', framealpha=0.1)
for text in l.get_texts():
        text.set_color('#D8DEE9')
#plt.show()
fig.savefig('decoupled_fence.pdf', transparent=True, dpi=300)
