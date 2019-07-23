import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from cycler import cycler

fig, ax = plt.subplots()
xdata, ydata = [], []
xdata2, ydata2 = [], []
ln1, = plt.plot([], [], 'ro')
ln2, = plt.plot([], [], 'bo')


myX = list(range(15))
coordinates1 = [(x, y) for x, y in zip(range(0, 20, 2), range(0, 10, 1))]
coordinates2 = [(x, y) for x, y in zip(range(0, 10, 1), range(0, 20, 2))]

def init():

    ax.set_xlim(-1, 20)
    ax.set_ylim(-1, 20)
    #return ln1, ln2

def update(frame):
    coordinates1 = frame[0]
    coordinates2 = frame[1]

    xdata.append(coordinates1[0])
    ydata.append(coordinates1[1])

    xdata2.append(coordinates2[0])
    ydata2.append(coordinates2[1])

    ln1.set_data(xdata, ydata)
    #ln1.set_color('r')
    ln2.set_data(xdata2, ydata2)
    #ln2.set_color('g')

    return ln1, ln2

def update2(frame):
    print('frame:', frame)

    global xdata
    global ydata

    if (len(xdata) == 4):
        xdata = []
        ydata = []

    xdata.append(frame)
    ydata.append(frame)

    ln1.set_data(xdata, ydata)

    #ln1.set_color('g')
    #return ln1,


numPed = 3

# i-th entry is data for the i-th pedestrian, length is numPed*2
xdata = []
ydata = []

# i-th entry is line for the i-th pedestrian, length is numped*2
# you can set the colors of the lines during initialization, also the shapes
lines = []
#ln1, = plt.plot([], [], color='r', linestyle='dashed',  marker='o', markerfacecolor='r', markersize=4)
ln1, = plt.plot([], [], color='g', linestyle=':')

# obs+gt
for i in range(numPed):
    lines.append(plt.plot([], [], color='g', linestyle='-'))

# fake
for i in range(numPed):
    lines.append(plt.plot([], [], color='g', linestyle='--'))


def update3(frame):

    for idx, currPed in enumerate(frame):
        xdata[idx].append(currPed[0])
        ydata[idx].append(currPed[1])

    # set colors, each pedestrian unique and consistent color


    # set shapes, one shape for obs and gt, another shape for fake


def init3():

    ax.set_xlim(-1, 20)
    ax.set_ylim(-1, 20)

    #return ln1, ln2




# make update work with an arbitrary number of pedestrians
# unique colors for pedestrians, based on (modulo) index
# different shapes for gt and fake predictions (based on index: change shape when index > numPed)

#ani = FuncAnimation(fig, update, frames=zip(coordinates1, coordinates2), init_func=init, blit=False)
ani = FuncAnimation(fig, update2, frames=list(range(20)), init_func=init, blit=False)



plt.show()