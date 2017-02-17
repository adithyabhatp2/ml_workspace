#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os;


def readSeriesFromFile(seriesFilePath):
    series1File = open(seriesFilePath, 'r')
    yData = []
    xData = []
    lines = series1File.read().split("\n")
    lines = [line for line in lines if line != ""]
    yDataPoint = 0
    xDataPoint = 0
    for line in lines[2:-7]:
        yDataPoint = float(line.split(",")[2]) * 100.0
        xDataPoint = line.split(",")[0]
        yData.append(yDataPoint)
        xData.append(xDataPoint)
    return yData, xData


def drawSeriesAsLine(seriesPath, plt):

    yData, xData = readSeriesFromFile(seriesPath)

    plt.plot(xData, yData)

    plt.autoscale(enable=True)

    plt.legend(seriesPath)


def drawMultiSeriesLinePlot():

    series0Path = "/u/a/d/adbhat/private/gitRepository/ml_workspace/NeuralNetworks/out/results/learningRateVariation/" + "sigmoid_eta_1e-05.csv"
    series1Path = "/u/a/d/adbhat/private/gitRepository/ml_workspace/NeuralNetworks/out/results/learningRateVariation/" + "relu_eta_0.0001.csv"
    series2Path = "/u/a/d/adbhat/private/gitRepository/ml_workspace/NeuralNetworks/out/results/learningRateVariation/" + "relu_eta_0.001.csv"
    # series3Path = "/u/a/d/adbhat/private/gitRepository/ml_workspace/NeuralNetworks/out/results/learningRateVariation/" + "relu_eta_0.01.csv"
    series4Path = "/u/a/d/adbhat/private/gitRepository/ml_workspace/NeuralNetworks/out/results/learningRateVariation/" + "relu_eta_0.1.csv"

    legends = []
    legends.append('0.00001')
    legends.append('0.0001')
    legends.append('0.001')
    # legends.append('0.01')
    legends.append('0.1')

    drawSeriesAsLine(series0Path, plt)
    drawSeriesAsLine(series1Path, plt)
    drawSeriesAsLine(series2Path, plt)
    # drawSeriesAsLine(series3Path, plt)
    drawSeriesAsLine(series4Path, plt)

    plt.legend(legends, loc='lower right')

    plt.title("Learning Rate (eta) Tuning - ReLU")
    plt.xlabel("Epochs")
    plt.ylabel("Tuning Set Accuracy")
    
    # plt.show()
    plt.savefig("temp.png")
    print "Done"


def main():
    drawMultiSeriesLinePlot()


if __name__ == '__main__':
    main()
