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
ax.set_xlabel('Blocks count')
ax.tick_params(color='#D8DEE9', labelcolor='#D8DEE9')
for spine in ax.spines.values():
    spine.set_edgecolor('#D8DEE9')

fig.set_size_inches(22.5, 10.5)
rtx3090_x, rtx3090_y = load('rtx3090_2')
rtx2080_x, rtx2080_y = load('rtx2080')
gtx560_x, gtx560_y = load('gtx560')
gtx730_x, gtx730_y = load('gtx730')
gtx970_x, gtx970_y = load('gtx970')
gtx1080_x, gtx1080_y = load('gtx1080')

plt.plot(rtx3090_x, rtx3090_y, color='#EBCB8B', label='rtx3090')
plt.plot(rtx2080_x, rtx2080_y, color='#A3BE8C', label='rtx2080')
plt.plot(gtx560_x, gtx560_y, color='#81A1C1', label='gtx560')
plt.plot(gtx730_x, gtx730_y, color='#E5E9F0', label='gtx730')
plt.plot(gtx970_x, gtx970_y, color='#88C0D0', label='gtx970')
plt.plot(gtx1080_x, gtx1080_y, color='#BF616A', label='gtx1080')
ax.xaxis.label.set_color('#D8DEE9')
ax.yaxis.label.set_color('#D8DEE9')
plt.grid(color='#D8DEE9')
l = plt.legend(facecolor='white', framealpha=0.1)
for text in l.get_texts():
        text.set_color('#D8DEE9')
#plt.show()
fig.savefig('mp_elap.pdf', transparent=True, dpi=300)
