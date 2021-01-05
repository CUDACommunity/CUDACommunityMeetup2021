#!/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

font = {'family' : 'normal',
        'size'   : 26}

matplotlib.rc('font', **font)

y = []
x = []

with open('cmake-build-debug/fin', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        x.append(len(y))
        y.append(row[0])

# the histogram of the data
fig,ax = plt.subplots(1)
fig.set_size_inches(22.5, 10.5)
#plt.yticks(np.arange(min(x), max(x)+1, 200.0))
plt.bar(x, y, width=1.0, facecolor='#A3BE8C')

plt.xlabel('Block number')
plt.ylabel('Iterations count')
ax.xaxis.label.set_color('#D8DEE9')
ax.yaxis.label.set_color('#D8DEE9')
#ax.set_yticklabels([])
ax.tick_params(color='#D8DEE9', labelcolor='#D8DEE9')
for spine in ax.spines.values():
    spine.set_edgecolor('#D8DEE9')

#plt.ylim(0,1000)
ax.grid(True, color='#D8DEE9')
#plt.show()
fig.savefig('decoupled_hist.pdf', transparent=True, dpi=300)
