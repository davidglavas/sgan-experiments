import matplotlib.pyplot as plt
import numpy as np

# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

ethPooling8 = [0.4, 1.3, 4.5, 7.4, 20.5, 40.3, 63, 85.4]
ethPooling12 = [1.1, 3.4, 7.7, 10.2, 17.4, 40.9, 68.3, 97.1]
hotelPooling8 = [0.9, 2.6, 6.7, 12.1, 30.7, 51.8, 72.7, 89.8]
hotelPooling12 = [1, 2.7, 7.5, 13, 29.4, 54, 73.3, 85.4]
univPooling8 = [15.5, 41.1, 74.5, 87.4, 97.4, 99.3, 100, 100]
univPooling12 = [24.7, 53.8, 81.6, 90.8, 97.3, 99.2, 99.8, 99.9]
zara1Pooling8 = [0.3, 0.8, 2.4, 5.9, 23.3, 51.7, 80.8, 91.9]
zara1Pooling12 = [0.8, 2.4, 7.3, 13.2, 29.5, 51.5, 83, 90.3]
zara2Pooling8 = [1.4, 4.4, 13.1, 23.7, 69.2, 86.8, 92.1, 94.4]
zara2Pooling12 = [6.9, 19.2, 41.6, 54.9, 75.8, 87, 91.9, 94]

ethNoPooling12 = [0.5, 1.6, 4.8, 7.9, 10.1, 32.1, 51.5, 97.1]
ethNoPooling8 = [0.4, 1.4, 4, 7.7, 18.8, 36.5, 59.9, 85.4]
hotelNoPooling12 = [1.4, 2.7, 6.1, 11.6, 28.2, 53.6, 70.3, 85.8]
hotelNoPooling8 = [0.6, 1.9, 5.1, 9.7, 29.2, 51.9, 70.6, 90.1]
univNoPooling12 = [18.9, 45.6, 75.4, 86.7, 96.4, 99, 99.8, 99.9]
univNoPooling8 = [12.4, 35.3, 68.3, 83.9, 96.6, 99.5, 100, 100]
zara1NoPooling12 = [0.3, 1.3, 4.0, 7.8, 21.1, 40.9, 79.6, 90.2]
zara1NoPooling8 = [0.2, 0.8, 2.9, 5.5, 19.2, 42.4, 79, 91.6]
zara2NoPooling12 = [2.1, 6.2, 14.9, 24.9, 66.4, 86.7, 91.6, 93.8]
zara2NoPooling8 = [1.3, 4.0, 10.8, 20.9, 66.1, 86.9, 91.9, 94.3]

thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#fig.suptitle('Collisions')

#ax1.plot(x, y, '.-')
ax1.plot(thresholds, ethNoPooling8, '.-')
ax1.plot(thresholds, hotelNoPooling8, '.-')
ax1.plot(thresholds, univNoPooling8, '.-')
ax1.plot(thresholds, zara1NoPooling8, '.-')
ax1.plot(thresholds, zara2NoPooling8, '.-')

#ax2.plot(x, y**2, 'tab:orange')
ax2.plot(thresholds, ethPooling8, '.-')
ax2.plot(thresholds, hotelPooling8, '.-')
ax2.plot(thresholds, univPooling8, '.-')
ax2.plot(thresholds, zara1Pooling8, '.-')
ax2.plot(thresholds, zara2Pooling8, '.-')

#ax3.plot(x, -y, 'tab:green')
ax3.plot(thresholds, ethNoPooling12, '.-')
ax3.plot(thresholds, hotelNoPooling12, '.-')
ax3.plot(thresholds, univNoPooling12, '.-')
ax3.plot(thresholds, zara1NoPooling12, '.-')
ax3.plot(thresholds, zara2NoPooling12, '.-')

#ax4.plot(x, -y**2, 'tab:red')
ax4.plot(thresholds, ethPooling12, '.-')
ax4.plot(thresholds, hotelPooling12, '.-')
ax4.plot(thresholds, univPooling12, '.-')
ax4.plot(thresholds, zara1Pooling12, '.-')
ax4.plot(thresholds, zara2Pooling12, '.-')


for ax in fig.get_axes():
    ax.label_outer()
    ax.set_xticks([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2])
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=4)
    #plt.xticks(fontsize=14, rotation=90)


#fig.autofmt_xdate()

ax1.set_title('Without Pooling')
ax2.set_title('With Pooling')
ax1.set(ylabel='Pred_Length 8')
ax3.set(ylabel='Pred_Length 12')

#plt.figlegend([ax1, ax2, ax3, ax4], ['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2'], loc='lower center', ncol=5, labelspacing=0.)

import matplotlib.patches as mpatches
eth = mpatches.Patch(color='blue', label='ETH')
hotel = mpatches.Patch(color='orange', label='HOTEL')
univ = mpatches.Patch(color='green', label='UNIV')
zara1 = mpatches.Patch(color='red', label='ZARA1')
zara2 = mpatches.Patch(color='purple', label='ZARA2    ')
lgd = plt.figlegend(handles=[eth, hotel, univ, zara1, zara2], bbox_to_anchor=(1.1, 0.9))


plt.show()
fig.savefig('thresholdsFigure.png', dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
