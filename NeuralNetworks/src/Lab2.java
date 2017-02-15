import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * Created by adbhat on 2/12/17.
 */
public class Lab2 {

    static int debugLevel = 3;

    public static void main(String args[]) {

        // Sanity check
        NeuralNetwork myNN = new NeuralNetwork(4, 3, 0.1);
        myNN.addHiddenLayer(2, ActivationFunctions.SIGMOID);
        myNN.intialize();

        double[] input = {1, 2.5, 5, 10};
        double[] outputLabels = {1, 2, 3};
        double[] prediction = myNN.predictAndTrain(input, false, outputLabels);

        // Training check
        // learnOr();
        // learnAnd();

        // Parse Dataset
        if (args.length != 1) {
            System.err.println("Usage : java Lab2 <fileName>");
            System.exit(1);
        }

        String fileName = args[0];
        DataParser.generateDataset(fileName);

        List<double[]> trainInputs = DataParser.train_set_input;
        List<double[]> trainOutputs = DataParser.train_set_output;

        List<double[]> tuneInputs = DataParser.tune_set_input;
        List<double[]> tuneOutputs = DataParser.tune_set_output;

        List<double[]> testInputs = DataParser.test_set_input;
        List<double[]> testOutputs = DataParser.test_set_output;

        // Create and initialize NN
        NeuralNetwork trainNN = new NeuralNetwork(trainInputs.get(0).length, trainOutputs.get(0).length, 0.1);

        int numHiddenLayers = 1;
        int numHiddenUnits = 10;
        trainNN.addHiddenLayer(10, ActivationFunctions.SIGMOID);

        trainNN.intialize();

        // Pick best model

        int numEpochs = 50;
        // Variables used to select and store best mode using tuning set.
        List<double[][]> bestWeights = null;
        double maxTuneAccuracy = 0.0;
        int bestEpoch = -1;

        for (int i = 0; i < numEpochs; i++) {
            if (debugLevel <= 3) {
                System.out.println("Epoch : " + i);
            }
            train(trainInputs, trainOutputs, trainNN, false, false);
            double acc = test(tuneInputs, tuneOutputs, trainNN, false, false);
            if (acc > maxTuneAccuracy) {
                bestWeights = trainNN.crossLayerWts;
                maxTuneAccuracy = acc;
                bestEpoch = i;
            }
        }
        trainNN.crossLayerWts = bestWeights;
        if (debugLevel <= 3) {
            System.out.println("Best epoch : " + bestEpoch);
        }

        // Test
        test(testInputs, testOutputs, trainNN, false, true);

        System.out.println("Done");
    }



    public static double train(List<double[]> inputs, List<double[]> labels, NeuralNetwork network, boolean printPredictedClass, boolean printPredictionSummary) {

        int[][] confusionMatrix = new int[labels.size()][labels.size()];

        for (int i = 0; i < inputs.size(); i++) {
            double[] instance = inputs.get(i);
            double[] label = labels.get(i);
            double[] predictions = network.predictAndTrain(instance, false, label);

            int predictedClass = argMax(predictions);
            int actualClass = argMax(label);

            confusionMatrix[actualClass][predictedClass]++;

            if (debugLevel <= 1) {
                System.out.println("Predicted: " + predictedClass + " Actual: " + actualClass + " Confidence: " + predictions[predictedClass] + "\n");
            }

            if (printPredictedClass) {
                System.out.println(predictedClass);
            }
        }

        double accuracy = getAccuracyFromConfusionMatrix(confusionMatrix);

        if (debugLevel <= 2 || printPredictionSummary) {
            System.out.println("Train Accuracy : " + accuracy);
            System.out.println("Confusion Matrix: \n" + Arrays.deepToString(confusionMatrix));
        }
        return accuracy;
    }


    public static double test(List<double[]> inputs, List<double[]> labels, NeuralNetwork network, boolean printPredictedClass, boolean printPredictionSummary) {

        int[][] confusionMatrix = new int[labels.get(0).length][labels.get(0).length];

        for (int i = 0; i < inputs.size(); i++) {
            double[] instance = inputs.get(i);
            double[] label = labels.get(i);
            double[] predictions = network.predictAndTrain(instance, true, null);

            int predictedClass = argMax(predictions);
            int actualClass = argMax(label);

            confusionMatrix[actualClass][predictedClass]++;

            if (debugLevel <= 1) {
                System.out.println("Predicted: " + predictedClass + " Actual: " + actualClass + " Confidence: " + predictions[predictedClass] + "\n");
            }

            if (printPredictedClass) {
                System.out.println(predictedClass);
            }
        }

        double accuracy = getAccuracyFromConfusionMatrix(confusionMatrix);

        if (debugLevel <= 2 || printPredictionSummary) {
            System.out.println("Train Accuracy : " + accuracy);
            System.out.println("Confusion Matrix: \n" + Arrays.deepToString(confusionMatrix));
        }
        return accuracy;
    }


    private static double getAccuracyFromConfusionMatrix(int[][] confusionMatrix) {
        double accuracy = 0;
        int correct = 0;
        int total = 0;

        for (int i = 0; i < confusionMatrix.length; i++) {
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                total += confusionMatrix[i][j];
                if (i == j) {
                    correct += confusionMatrix[i][j];
                }
            }
        }
        return (double) correct / (double) total;
    }


    public static int argMax(double[] array) {
        double maxVal = 0;
        int maxInd = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxInd = i;
            }
        }
        return maxInd;
    }

    public static void learnAnd() {

        NeuralNetwork myNN = new NeuralNetwork(2, 1, 1);
        myNN.intialize();

        double[][] crossLayerWt = myNN.crossLayerWts.get(0);
        crossLayerWt[0][0] = 10.0;
        crossLayerWt[0][1] = 40.0;
        crossLayerWt[0][2] = -20.0;

        System.out.println("Before training");
        myNN.printWeightArrays();

        Random random = new Random();

        for (int i = 0; i < 1000000; i++) {
            boolean a = random.nextBoolean();
            boolean b = random.nextBoolean();

            boolean op = a && b;

            double input[] = new double[2];
            input[0] = a == true ? 1.0 : 0.0;
            input[1] = b == true ? 1.0 : 0.0;

            double output[] = new double[1];
            output[0] = op == true ? 1.0 : 0.0;

            double preds[] = myNN.predictAndTrain(input, false, output);

            if (i % 10000 == 0) {
                System.out.println("\nEpoch: " + i);
                myNN.printWeightArrays();
                System.out.println("Input: " + Arrays.toString(input));
                System.out.println("Prediction : " + Arrays.toString(preds));
            }
        }
    }

    public static void learnOr() {

        NeuralNetwork orNN = new NeuralNetwork(2, 1, 0.1);
        orNN.intialize();

        System.out.println("Before training");
        orNN.printWeightArrays();

        Random random = new Random();

        for (int i = 0; i < 1000; i++) {
            boolean a = random.nextBoolean();
            boolean b = random.nextBoolean();

            boolean op = a || b;

            double input[] = new double[2];
            input[0] = a == true ? 1.0 : 0.0;
            input[1] = b == true ? 1.0 : 0.0;

            double output[] = new double[1];
            output[0] = op == true ? 1.0 : 0.0;

            orNN.predictAndTrain(input, false, output);

            if (i % 100 == 0) {
                System.out.println("Epoch: " + i);
                orNN.printWeightArrays();
            }
        }
    }


}
