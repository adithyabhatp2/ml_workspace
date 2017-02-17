#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *

import matplotlib
import matplotlib.ticker as ticker
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True

matplotlib.rcParams.update({'figure.autolayout':True})
matplotlib.rcParams.update({'font.size': 11})

def Plot1Sub1(data, figName, legend, labs,x_lab):
    (x, y1, y2, xlabel, ylabel) = data
    fig, ax = plt.subplots(figsize=(5,3.4))
    x_val = index =[2, 3, 4]
    bar_width = 0.35
    opacity = 0.4
    index2 = [ i+bar_width for i in index ]
    rects1 = plt.bar(index, y1, bar_width, color = 'g', hatch = '//')
    rects2 = plt.bar(index2, y2, bar_width, label = legend, color='y', hatch = '\\\\')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    x_val = [2.2, 3.2, 4.2]
    plt.xticks(x_val,x_lab)
    plt.title("Impact of Hidden Units", fontsize = 12)
    plt.ylim([0, 90])
    #plt.yticks()
    plt.legend(labs,loc='best', fontsize = 12)
    plt.tight_layout()
    subplots_adjust(bottom=0.2, left=0.15)
    pp = PdfPages(figName)
    #pp.savefig(bbox_inches='tight')
    pp.savefig()
    pp.close()


def Plot_exp1c():

    y11 = []
    y12 = []

    for eta in [10, 100, 1000]:
        filename = "sigmoid_hu_" + str(eta) + ".csv"
        fp = open(filename, "r")
        lines = fp.read().split("\n")
        lines = [line for line in lines if line != ""]
        tuning_accuracy = 0
        for line in lines[2:-6]:
            tuning_accuracy = float(line.split(",")[3]) * 100.0
        y11.append(tuning_accuracy)

    for eta in [10, 100, 1000]:
        filename = "relu_hu_" + str(eta) + ".csv"
        fp = open(filename, "r")
        lines = fp.read().split("\n")
        lines = [line for line in lines if line != ""]
        tuning_accuracy = 0
        for line in lines[2:-6]:
            tuning_accuracy = float(line.split(",")[3]) * 100.0
        y12.append(tuning_accuracy)

    x=  ["10", "100", "1000"]
    xlabel = "Number of Hidden Units"
    ylabel = "Test Set Accuracy (%)"
    sub1 = (x, y11, y12, xlabel, ylabel)
    legend = ["Sigmoid", "ReLU"]
    labs = legend
    x_lab = x
    
    Plot1Sub1(sub1, "sigmoidHUs.pdf",legend,labs,x_lab)

def main():
    Plot_exp1c()
    #Plot_exp4()
    #Plot_exp7()

main()

