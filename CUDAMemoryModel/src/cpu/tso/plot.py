#!/bin/env python3
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'size'   : 26}

matplotlib.rc('font', **font)

x = []
with open('result') as f:
    for line in f:
        x.append(int(line))

print(min(x))
print(sum(x)/len(x))
print(max(x))

# the histogram of the data
fig,ax = plt.subplots(1)
fig.set_size_inches(22.5, 10.5)
n, bins, patches = plt.hist(x, 150, density=True, facecolor='#A3BE8C')

plt.xlim(0, 100000)
plt.ylim(0, 0.000029)
#plt.xlabel('Load after Store violations count')
ax.set_yticklabels([])
ax.tick_params(color='#D8DEE9', labelcolor='#D8DEE9')
#ax.ticklabel_format(style='plain')
for spine in ax.spines.values():
    spine.set_edgecolor('#D8DEE9')

ax.xaxis.grid(True, color='#D8DEE9')
#plt.show()
fig.savefig('sc_violations_hist.pdf', transparent=True, dpi=300)
