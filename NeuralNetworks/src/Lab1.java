import java.io.IOException;
import java.io.PrintStream;
import java.util.*;

import weka.core.Instance;
import weka.core.Instances;

public class Lab1 {

    static int debugLevel = 2;
    static LinkedList<String> predictStats;
    static HashMap<Integer, String> outputLines;

    public static void main(String[] args) throws IOException {

        if (args.length != 3) {
            System.out.println("Usage: <train-file> <tune-file> <test-file>");
            System.exit(0);
        }

        String trainFilePath = args[0];
        String tuneFilePath = args[1];
        String testFilePath = args[2];

        double learnRate = 0.1;
        int numEpochs = 1000;

        Instances trainInstances = UciDatasetInstanceParser.parseFromUciDataFile(trainFilePath);
        Instances tuneInstances = UciDatasetInstanceParser.parseFromUciDataFile(tuneFilePath);
        Instances testInstances = UciDatasetInstanceParser.parseFromUciDataFile(testFilePath);

        int numAttributes = trainInstances.numAttributes();
        ANNode singleNode = new ANNode(trainInstances.numAttributes() - 1, learnRate);

        // Variables used to select and store best mode using tuning set.
        ANNode bestNode = null;
        double maxTuneAccuracy = 0.0;
        int bestEpoch = -1;

        // Train using all instances numEpoch times, pick best model using tuning set.
        for (int i = 0; i < numEpochs; i++) {
            if (debugLevel <= 2) {
                System.out.println("Epoch : " + i);
            }
            train(trainInstances, singleNode);
            double acc = test(tuneInstances, singleNode, false, false);
            if (acc > maxTuneAccuracy) {
                bestNode = new ANNode(singleNode);
                maxTuneAccuracy = acc;
                bestEpoch = i;
            }
        }

        System.out.println("Best epoch : " + bestEpoch);
        test(testInstances, bestNode, true, true);

    }

    private static void writeOutputToStream(PrintStream out) {
        if (debugLevel <= 4) {
            System.out.println("Output Size: " + outputLines.size());
        }
        for (int i = 0; i < outputLines.size(); i++) {
            out.println(outputLines.get(new Integer(i)));
        }
    }

    /**
     * Trains the ANNode using the instaces in an online manner.
     *
     * @param instances
     * @param singleNode
     * @return
     */
    public static double train(Instances instances, ANNode singleNode) {
        double accuracy = 0.0;
        int tp = 0, tn = 0, fp = 0, fn = 0;

        Enumeration<Instance> instEnum = instances.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = instEnum.nextElement();
            double predictedClassValue = singleNode.predictAndLearn(inst);
            double actualClassValue = inst.classValue();

            String predictedClass = inst.classAttribute().value((int) predictedClassValue);
            String actualClass = inst.stringValue(inst.classAttribute());

            if (predictedClassValue == actualClassValue && actualClassValue == 0)
                tn++;
            else if (predictedClassValue == actualClassValue && actualClassValue == 1)
                tp++;
            else if (predictedClassValue != actualClassValue && predictedClassValue == 0)
                fn++;
            else if (predictedClassValue != actualClassValue && predictedClassValue == 1)
                fp++;

            if (debugLevel <= 1) {
                System.out.println("Predicted: " + predictedClass + " Actual: " + actualClass + " Confidence: " + singleNode.computeNet() + "\n");
            }

        }
        accuracy = (double) (tp + tn) / (double) (tp + tn + fp + fn);
        if (debugLevel <= 2) {
            System.out.println("Train Accuracy : " + accuracy + " TP: " + tp + " FP: " + fp + " TN: " + tn + " FN: " + fn);
        }
//            predictStats.add("TestIndex:\t" + testFoldIndex + "\tEpoch:\t" + epochNum + "\tTrain Accuracy :\t" + accuracy + "\tTP:\t" + tp + "\tFP:\t" + fp + "\tTN:\t" + tn + "\tFN:\t" + fn);

        return accuracy;
    }



    /**
     * Tests the instances against the given ANNode.
     *
     * @param instances The test instances.
     * @param singleNode The perceptron used for prediction.
     * @param printPredictedClass Print the class prediction or not.
     * @param printPredictionSummary Print the prediction summary statistics.
     * @return The accuracy.
     */
    public static double test(Instances instances, ANNode singleNode, boolean printPredictedClass, boolean printPredictionSummary) {
        double accuracy = 0.0;
        int tp = 0, tn = 0, fp = 0, fn = 0;

        Enumeration<Instance> instEnum = instances.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = instEnum.nextElement();
            double predictedClassValue = singleNode.predictOnly(inst);
            double actualClassValue = inst.classValue();

            String predictedClass = inst.classAttribute().value((int) predictedClassValue);
            String actualClass = inst.stringValue(inst.classAttribute());

            if (predictedClassValue == actualClassValue && actualClassValue == 0)
                tn++;
            else if (predictedClassValue == actualClassValue && actualClassValue == 1)
                tp++;
            else if (predictedClassValue != actualClassValue && predictedClassValue == 0)
                fn++;
            else if (predictedClassValue != actualClassValue && predictedClassValue == 1)
                fp++;

            if (debugLevel <= 1) {
                System.out.println("Predicted: " + predictedClass + " Actual: " + actualClass + " Confidence: " + singleNode.computeNet() + "\n");
            }

            if (printPredictedClass) {
                System.out.println(predictedClass);
            }

        }

        accuracy = (double) (tp + tn) / (double) (tp + tn + fp + fn);
        if (debugLevel <= 2 || printPredictionSummary) {
            System.out.println("Test Accuracy : " + accuracy + " TP: " + tp + " FP: " + fp + " TN: " + tn + " FN: " + fn);
        }

        return accuracy;
    }


}
