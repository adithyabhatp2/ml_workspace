import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by adbhat on 2/12/17.
 */
public class Lab2 {

    static int debugLevel = 3;

    public static void main(String args[]) {

        // // Sanity check
        // NeuralNetwork myNN = new NeuralNetwork(4, 3, 0.1);
        // myNN.addHiddenLayer(2, ActivationFunctions.SIGMOID);
        // myNN.intialize();
        //
        // double[] input = {1, 2.5, 5, 10};
        // double[] outputLabels = {1, 2, 3};
        // double[] prediction = myNN.predictAndTrain(input, false, outputLabels);

        // // Training check
        // learnAndOr();
        // learnAndOrXor();

        // Parse Dataset
        if (args.length != 7) {
            System.err.println("Usage : java Lab2 <fileName> <numHiddenUnits> <sigmoid/relu> <numEpochs> <learningRate> <momentumRate> <weightDecayRate>");
            System.exit(1);
        }

        System.out.println("Lab2 <fileName> <numHiddenUnits> <sigmoid/relu> <numEpochs> <learningRate> <momentumRate> <weightDecayRate>".replaceAll(" ", "\t"));
        System.out.println(Arrays.toString(args).replaceAll(", ", "\t").replaceFirst("\\[", "").replaceFirst("\\]", ""));

        String fileName = args[0];
        int numHiddenLayers = 1;
        int numHiddenUnits = Integer.parseInt(args[1]);
        String activationFunction = args[2].toUpperCase();
        int numEpochs = Integer.parseInt(args[3]);

        double learningRate = Double.parseDouble(args[4]);
        double momentumRate = Double.parseDouble(args[5]);
        double weightDecayRate = Double.parseDouble(args[6]);

        DataParser.generateDataset(fileName);

        List<double[]> trainInputs = DataParser.train_set_input;
        List<double[]> trainOutputs = DataParser.train_set_output;

        List<double[]> tuneInputs = DataParser.tune_set_input;
        List<double[]> tuneOutputs = DataParser.tune_set_output;

        List<double[]> testInputs = DataParser.test_set_input;
        List<double[]> testOutputs = DataParser.test_set_output;

        // Create and initialize NN
        NeuralNetwork trainNN = new NeuralNetwork(trainInputs.get(0).length, trainOutputs.get(0).length, learningRate, momentumRate, weightDecayRate);

        trainNN.addHiddenLayer(numHiddenUnits, activationFunction);

        trainNN.intialize();

        // Pick best model


        // Variables used to select and store best mode using tuning set.
        List<double[][]> bestWeights = null;
        double maxTuneAccuracy = 0.0;
        int bestEpoch = -1;

        List<InputPair> inputPairs = createInputPairCollection(trainInputs, trainOutputs);

        for (int i = 0; i < numEpochs; i++) {
            if (debugLevel <= 3) {
                System.out.println("\nEpoch : " + i);
            }

            Collections.shuffle(inputPairs);

            trainInputs = inputPairs.stream().map(InputPair::getInput).collect(Collectors.toList());
            trainOutputs = inputPairs.stream().map(InputPair::getOutput).collect(Collectors.toList());

            train(trainInputs, trainOutputs, trainNN, false, true);
            double acc = test(tuneInputs, tuneOutputs, trainNN, false, true, true);
            if (acc > maxTuneAccuracy) {
                bestWeights = trainNN.crossLayerWts;
                maxTuneAccuracy = acc;
                bestEpoch = i;
            }
            test(testInputs, testOutputs, trainNN, false, true, false);
        }
        trainNN.crossLayerWts = bestWeights;
        if (debugLevel <= 3) {
            System.out.println("\nBest epoch : " + bestEpoch);
        }

        // Test
        test(testInputs, testOutputs, trainNN, false, true, false);

        System.out.println("Done");
    }



    public static double train(List<double[]> inputs, List<double[]> labels, NeuralNetwork network, boolean printPredictedClass, boolean printPredictionSummary) {

        int[][] confusionMatrix = new int[labels.get(0).length][labels.get(0).length];

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
            System.out.println("Confusion Matrix: \n" + Arrays.deepToString(confusionMatrix).replaceAll("], ", "],\n "));
        }
        return accuracy;
    }


    public static double test(List<double[]> inputs, List<double[]> labels, NeuralNetwork network, boolean printPredictedClass, boolean printPredictionSummary, boolean isTune) {

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
            String testType = isTune == true ? "Tune" : "Test";
            System.out.println(testType + " Accuracy : " + accuracy);
            System.out.println("Confusion Matrix: \n" + Arrays.deepToString(confusionMatrix).replaceAll("], ", "],\n "));
        }
        return accuracy;
    }



    private static List<InputPair> createInputPairCollection(List<double[]> inputs, List<double[]> labels) {
        Iterator<double[]> inputItr = inputs.iterator();
        Iterator<double[]> labelItr = labels.iterator();

        List<InputPair> inputPaircollection = new LinkedList<>();

        while(inputItr.hasNext()) {
            inputPaircollection.add(new InputPair(inputItr.next(), labelItr.next()));
        }

        return inputPaircollection;
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



    public static void learnAndOr() {

        NeuralNetwork myNN = new NeuralNetwork(2, 2, 0.1, 0, 0);
        myNN.intialize();

        System.out.println("Before training");
        myNN.printWeightArrays();

        Random random = new Random();

        for (int i = 0; i < 10000; i++) {
            boolean a = random.nextBoolean();
            boolean b = random.nextBoolean();

            double input[] = new double[2];
            input[0] = a == true ? 1.0 : 0.0;
            input[1] = b == true ? 1.0 : 0.0;

            double output[] = new double[2];
            output[0] = (a && b) ? 1.0 : 0.0;
            output[1] = (a || b) ? 1 : 0;

            double preds[] = myNN.predictAndTrain(input, false, output);

            if (i % 1000 == 0) {
                System.out.println("\nEpoch: " + i);
                myNN.printWeightArrays();
                System.out.println("Input: " + Arrays.toString(input));
                System.out.println("Prediction : " + Arrays.toString(preds));
            }
        }
    }

    public static void learnAndOrXor() {

        NeuralNetwork myNN = new NeuralNetwork(2, 3, 0.1, 0, 0);
        myNN.addHiddenLayer(2, ActivationFunctions.SIGMOID);
        myNN.intialize();

        System.out.println("Before training");
        myNN.printWeightArrays();

        Random random = new Random();

        for (int i = 0; i < 10000; i++) {
            boolean a = random.nextBoolean();
            boolean b = random.nextBoolean();

            double input[] = new double[2];
            input[0] = a == true ? 1 : 0;
            input[1] = b == true ? 1 : 0;

            double output[] = new double[3];
            output[0] = (a && b) ? 1 : 0;
            output[1] = (a || b) ? 1 : 0;
            output[2] = (a ^ b) ? 1 : 0;

            double preds[] = myNN.predictAndTrain(input, false, output);

            if (i % 1000 == 0) {
                System.out.println("\nEpoch: " + i);
                myNN.printWeightArrays();
                myNN.printOutputVectors();
                System.out.println("Input: " + Arrays.toString(input));
                System.out.println("Prediction : " + Arrays.toString(preds));
            }
        }
    }


}
