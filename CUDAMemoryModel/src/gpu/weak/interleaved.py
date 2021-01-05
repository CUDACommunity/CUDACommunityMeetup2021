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
ax.set_xlabel('Number of blocks')
plt.xlim(0, 91000)
plt.ylim(0, 0.21)
ax.tick_params(color='#D8DEE9', labelcolor='#D8DEE9')
for spine in ax.spines.values():
    spine.set_edgecolor('#D8DEE9')
fig.set_size_inches(22.5, 10.5)
wof_x, wof_y = load('interleaved_wo_fences')
wf_x, w_y = load('interleaved_w_fences')

plt.plot(wof_x, wof_y, color='#A3BE8C', linewidth=2, label='without fences')
plt.plot(wf_x, w_y, color='#81A1C1', linewidth=2, label='with fences')
ax.xaxis.label.set_color('#D8DEE9')
ax.yaxis.label.set_color('#D8DEE9')
plt.grid(color='#D8DEE9')
l = plt.legend(facecolor='white', framealpha=0.1)
for text in l.get_texts():
        text.set_color('#D8DEE9')
#plt.show()
fig.savefig('interleaved_fence_perf.pdf', transparent=True, dpi=300)
