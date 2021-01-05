#!/bin/env python3
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'size'   : 26}

matplotlib.rc('font', **font)

y = [ 0, 55, 31, 47, 44, 66, 61, 0 ]
x = range(1, len(y) + 1)

# the histogram of the data
fig,ax = plt.subplots(1)
fig.set_size_inches(22.5, 10.5)
plt.bar(x, y, facecolor='#A3BE8C')

plt.xlabel('Data size')
plt.ylabel('Violations count')
ax.xaxis.label.set_color('#D8DEE9')
ax.yaxis.label.set_color('#D8DEE9')
#ax.set_yticklabels([])
ax.tick_params(color='#D8DEE9', labelcolor='#D8DEE9')
for spine in ax.spines.values():
    spine.set_edgecolor('#D8DEE9')

#plt.ylim(0,1000)
ax.grid(True, color='#D8DEE9')
#plt.show()
fig.savefig('mp_violations_hist.pdf', transparent=True, dpi=300)
