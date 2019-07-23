import numpy as np
import matplotlib.pyplot as plt

import pickle

def compareTwoResults():
    # load evaluation results
    fileName = 'paper.pkl'
    with open(fileName, 'rb') as f:
        maxADE8, maxFDE8, maxADE12, maxFDE12 = pickle.load(f)

    fileName = 'naiveRandom.pkl'
    with open(fileName, 'rb') as f:
        avgADE8, avgFDE8, avgADE12, avgFDE12 = pickle.load(f)

    # one hist for ADE and one for FDE
    # x axis are 5 datasets and AVG
    # y axis are evaluation results for given dataset and error metric


    N = 5
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # the first argument is the margin of the x-axis, the second of the y-axis
    ax.margins(0.1, 0.25)

    # more lists -> more rectangles within a block
    # more values in lists -> more separate blocks of rectangles

    # TODO compute average

    yvals = [4, 9, 2, 10, 4]  # maxADE8
    yvals = list(maxADE8.values())
    rects1 = ax.bar(ind, yvals, width, color='r')

    zvals = [1, 2, 3, 11, 5]  # avgADE8
    zvals = list(avgADE8.values())
    rects2 = ax.bar(ind+width, zvals, width, color='g')

    kvals = [11, 12, 13, 12, 2]  # maxADE12
    kvals = list(maxADE12.values())
    rects3 = ax.bar(ind+width*2, kvals, width, color='r')

    pvals = [1, 2, 3, 13, 1]  # avgADE12
    pvals = list(avgADE12.values())
    rects4 = ax.bar(ind+width*3, pvals, width, color='g')


    ax.set_ylabel('Error')  # ADE or FDE
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2', 'AVG') )
    #ax.legend( (rects1[0], rects2[0], rects3[0]), ('y', 'z', 'k') )
    ax.legend( (rects1[0], rects2[0]), ('Paper', 'github pretrained') )


    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.2f'%float(h),
                    ha='center', va='bottom')

    autolabel(rects1)
    #autolabel(rects2)
    autolabel(rects3)
    #autolabel(rects4)

    plt.show()


# 20VP-20 results
def createPaperResultsPKl():
    ADE8, FDE8, ADE12, FDE12 = {}, {}, {}, {}

    ADE8['ETH'] = 0.60
    ADE8['HOTEL'] = 0.52
    ADE8['UNIV'] = 0.44
    ADE8['ZARA1'] = 0.22
    ADE8['ZARA2'] = 0.29

    FDE8['ETH'] = 1.19
    FDE8['HOTEL'] = 1.02
    FDE8['UNIV'] = 0.84
    FDE8['ZARA1'] = 0.43
    FDE8['ZARA2'] = 0.58

    ADE12['ETH'] = 0.87
    ADE12['HOTEL'] = 0.67
    ADE12['UNIV'] = 0.76
    ADE12['ZARA1'] = 0.35
    ADE12['ZARA2'] = 0.42

    FDE12['ETH'] = 1.62
    FDE12['HOTEL'] = 1.37
    FDE12['UNIV'] = 1.52
    FDE12['ZARA1'] = 0.68
    FDE12['ZARA2'] = 0.84

    # pickle evaluation results
    destination = 'paper.pkl'
    with open(destination, 'wb') as f:
        pickle.dump((ADE8, FDE8, ADE12, FDE12), f)

compareTwoResults()
