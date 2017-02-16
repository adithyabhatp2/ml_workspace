import os 
import sys

experiment_no = int(sys.argv[1]) 

#java Lab2 <fileName> <numHiddenUnits> <sigmoid/relu> <numEpochs> <learningRate> <momentumRate> <weightDecayRate>

if experiment_no == 1: 
	cmd_prefix = "java -jar NeuralNetworks.jar ../../../filename.txt 10 sigmoid 100 "
	cmd_postfix = " 0 0"

	for eta in [0.00001, 0.0001, 0.001, 0.01, 0.1]:

		cmd = cmd_prefix + str(eta) + cmd_postfix
		filename = "sigmoid_eta_" + str(eta) + ".csv"
		cmd += " > " + filename
elif experiment_no == 2:
	cmd_prefix = "java -jar NeuralNetworks.jar ../../../filename.txt 10 relu 100 "
	cmd_postfix = " 0 0"

	for eta in [0.00001, 0.0001, 0.001, 0.01, 0.1]:

		cmd = cmd_prefix + str(eta) + cmd_postfix
		filename = "relu_eta_" + str(eta) + ".csv"
		cmd += " > " + filename