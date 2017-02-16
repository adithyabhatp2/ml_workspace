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
		os.system(cmd)
elif experiment_no == 2:
	cmd_prefix = "java -jar NeuralNetworks.jar ../../../filename.txt 10 relu 100 "
	cmd_postfix = " 0 0"

	for eta in [0.00001, 0.0001, 0.001, 0.01, 0.1]:

		cmd = cmd_prefix + str(eta) + cmd_postfix
		filename = "relu_eta_" + str(eta) + ".csv"
		cmd += " > " + filename
		os.system(cmd)
elif experiment_no == 3:
	cmd_prefix = "java -jar NeuralNetworks.jar ../../../filename.txt "
	cmd_postfix = " sigmoid 1000 0.001 0 0"

	for HUs in [10, 100, 1000]:
		print ("Number of hidden units is " + str(HUs))
		cmd = cmd_prefix + str(HUs) + cmd_postfix
		filename = "sigmoid_hu_" + str(HUs) + ".csv"
		cmd += " > " + filename
		os.system(cmd)
elif experiment_no == 4:
	cmd_prefix = "java -jar NeuralNetworks.jar ../../../filename.txt "
	cmd_postfix = " relu 1000 0.01 0 0"

	for HUs in [10, 100, 1000]:
		print ("Number of hidden units is " + str(HUs))
		cmd = cmd_prefix + str(HUs) + cmd_postfix
		filename = "relu_hu_" + str(HUs) + ".csv"
		cmd += " > " + filename
		os.system(cmd)
elif experiment_no == 5:
	cmd_prefix = "java -jar NeuralNetworks.jar ../../../filename.txt 10 sigmoid 100 0.001 0 "
	for lambda1 in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]:
		cmd = cmd_prefix + str(lambda1)
		filename = "sigmoid_lambda_" + str(lambda1) + ".csv"
		cmd += " > " + filename
		os.system(cmd)	
elif experiment_no == 6:
	cmd_prefix = "java -jar NeuralNetworks.jar ../../../filename.txt 10 relu 100 0.01 0 "

	for lambda1 in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]:
		cmd = cmd_prefix + str(lambda1)
		filename = "relu_lambda_" + str(lambda1) + ".csv"
		cmd += " > " + filename
		os.system(cmd)
elif experiment_no == 7:
	cmd_prefix = "java -jar NeuralNetworks.jar ../../../filename.txt "
	cmd_postfix = " sigmoid 1000 0.001 0.9 0.1"

	for HUs in [10, 100, 1000]:
		print ("Number of hidden units is " + str(HUs))
		cmd = cmd_prefix + str(HUs) + cmd_postfix
		filename = "all_sigmoid_hu_" + str(HUs) + ".csv"
		cmd += " > " + filename
		os.system(cmd)
elif experiment_no == 8:
	cmd_prefix = "java -jar NeuralNetworks.jar ../../../filename.txt "
	cmd_postfix = " relu 1000 0.01 0.9 0.01"

	for HUs in [10, 100, 1000]:
		print ("Number of hidden units is " + str(HUs))
		cmd = cmd_prefix + str(HUs) + cmd_postfix
		filename = "all_relu_hu_" + str(HUs) + ".csv"
		cmd += " > " + filename
		os.system(cmd)	