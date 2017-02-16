#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *

import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['axes.unicode_minus']=False
matplotlib.rcParams.update({'figure.autolayout':True})
matplotlib.rcParams.update({'font.size': 11})


def Plot1Sub1(data, figName, legend, labs,x_lab):
    (x, y1, y2, xlabel, ylabel) = data
    fig, ax = plt.subplots(figsize=(5,3))



    line1 = plt.plot(x, y1, 'bs--', label = legend[0])
    line2 = plt.plot(x, y2, 'k^-',  label = legend[1])

    plt.xlabel(xlabel)#, fontsize=15)
    plt.ylabel(ylabel)#, fontsize=15)
    plt.xticks(x, x_lab, fontsize=10)
    plt.yticks(fontsize=12)
    plt.title("Weight Decay Parameter (lambda) Tuning")
    ax.tick_params(axis='x', pad=7)
    plt.legend(labs,loc='upper center')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')
    plt.tight_layout()
    subplots_adjust(bottom=0.2, left=0.15)
    pp = PdfPages(figName)
    pp.savefig()
    pp.close()

def Plot_exp6():

    y11 = []
    y12 = []

    for eta in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]:
        filename = "sigmoid_lambda_" + str(eta) + ".csv"
        fp = open(filename, "r")
        lines = fp.read().split("\n")
        lines = [line for line in lines if line != ""]
        tuning_accuracy = 0
        for line in lines[2:-6]:
            tuning_accuracy = float(line.split(",")[2]) * 100.0
        y11.append(tuning_accuracy)

    for eta in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]:
        filename = "relu_lambda_" + str(eta) + ".csv"
        fp = open(filename, "r")
        lines = fp.read().split("\n")
        lines = [line for line in lines if line != ""]
        tuning_accuracy = 0
        for line in lines[2:-6]:
            tuning_accuracy = float(line.split(",")[2]) * 100.0
        y12.append(tuning_accuracy)

    x= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ylabel = "Tuning Set Accuracy (%)"
    xlabel = "Learning Rate"


    sub1 = (x, y11, y12, xlabel, ylabel)
    legend = ["Sigmoid", "ReLU"]

    labs = legend
    x_lab = ["10e-5", "10e-4", "10e-3", "10e-2", "10e-1", "1", "10e1", "10e2", "10e3", "10e4", "10e5"]
    Plot1Sub1(sub1, "weightDecay.pdf",legend,labs,x_lab)

Plot_exp6()