import java.io.IOException;
import java.io.PrintStream;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public class Main {

    static int debugLevel = 2;
    static LinkedList<String> predictStats;
    static HashMap<Integer, String> outputLines;

    public static void main(String[] args) throws IOException {

        if (args.length != 4) {
            System.out.println("Usage: <train-file> <test-file> learningRate epochs");
            System.exit(0);
        }

        String trainFilePath = args[0];
        String testFilePath = args[1];
        double learnRate = Double.parseDouble(args[2]);
        int numEpochs = Integer.parseInt(args[3]);


        Instances trainInstances = UciDatasetInstanceParser.parseFromUciDataFile(trainFilePath);
        Instances testInstances = UciDatasetInstanceParser.parseFromUciDataFile(testFilePath);

        int numAttributes = trainInstances.numAttributes();
        ANNode singleNode = new ANNode(trainInstances.numAttributes() - 1, learnRate);

        for (int i = 0; i < numEpochs; i++) {
            if(debugLevel <= 2) {
                System.out.println("Epoch : " + i);
            }
            train(trainInstances, singleNode);
            test(testInstances, singleNode);
        }

//		if(args.length == 4)
//			{
//			// TODO : input type checking
//			String arffTrainFilePath = args[0];
//			int numFolds = Integer.parseInt(args[1]);
//			double learnRate = Double.parseDouble(args[2]);
//			int numEpochs = Integer.parseInt(args[3]);
//
//			predictStats = new LinkedList<String>();
//			outputLines = new HashMap<Integer, String>();
//
//			Instances allInstances = ArffHelper.readInputFromArff(arffTrainFilePath);
//
//			ANNode singleNode = new ANNode(allInstances.numAttributes() - 1, learnRate);
//
//			int numAttributes = allInstances.numAttributes();
//			// List<Instances> folds =
//			// StatisticsHelper.getStratifiedFolds(allInstances, numFolds);
//			List<LinkedList<Instance>> folds = StatisticsHelper.getStratifiedFolds_Lists(allInstances, numFolds);
//
//			// System.out.println("Num entries in instpos map: " +
//			// StatisticsHelper.originalInstPosMap.keySet().size());
//			for (int testFoldIndex = 0;testFoldIndex < numFolds;testFoldIndex++)
//				{
//
//				singleNode = new ANNode(numAttributes - 1, learnRate);
//				// trainFolds over entire set epoch times
//				double trainAccuracy = -1;
//				for (int epochNum = 1;epochNum <= numEpochs;epochNum++)
//					{
//					trainAccuracy = trainFolds(folds, testFoldIndex, singleNode, epochNum);
//					}
//				// testFolds after epoch trainings
//				double testAccuracy = testFolds(folds, testFoldIndex, singleNode, numEpochs);
//				if(debugLevel <= 3)
//					{
//					System.out.println("Epoch : \t" + numEpochs + "\t TrainAccuracy: \t" + trainAccuracy + "\t TestAccuracy: \t" + testAccuracy);
//					}
//				}
//
//			writeOutputToStream(System.out);
//
//			}
//
//		else
//			{
//			System.out.println("Usage: neuralnet <data-set-file> numFolds learningRate epochs");
//			}
//
//		if(args.length == 0)
//			{
//		outputLines = new HashMap<Integer, String>();
//			System.out.println("Usage: neuralnet <data-set-file> nunFolds learningRate epochs");
//			String dataSet = "sonar";
//			String arffTrainFilePath = "input/" + dataSet + ".arff";
//
//			double learnRate = 0.1;
//			int numFolds = 10;
//			int numEpochs = 1000;
//
//			predictStats = new LinkedList<String>();
//
//			Instances allInstances = ArffHelper.readInputFromArff(arffTrainFilePath);
//			int numAttributes = allInstances.numAttributes();
//			ANNode singleNode = new ANNode(allInstances.numAttributes() - 1, learnRate);
//
//			List<LinkedList<Instance>> folds = StatisticsHelper.getStratifiedFolds_Lists(allInstances, numFolds);
//
//			for (int testFoldIndex = 0;testFoldIndex < numFolds;testFoldIndex++)
//				{
//				singleNode = new ANNode(numAttributes - 1, learnRate);
//				for (int epochNum = 1;epochNum <= numEpochs;epochNum++)
//					{
//					double trainAccuracy = trainFolds(folds, testFoldIndex, singleNode, epochNum);
//					double testAccuracy = testFolds(folds, testFoldIndex, singleNode, epochNum);
//					if(debugLevel <= 3)
//						{
//						if(epochNum == 1 || epochNum == 10 || epochNum == 100 || epochNum == 1000)
//							System.out.println("Epoch : \t" + epochNum + "\t TrainAccuracy:\t" + trainAccuracy + "\t TestAccuracy: \t" + testAccuracy);
//						}
//					}
//				}
//
//			if(debugLevel <= 5)
//				{
//				System.out.print(String.join("\n", predictStats));
//				}

        // Single Run thru all instances in order - for training

        // for (int i = 0;i < allInstances.numInstances();i++)
        // {
        // Instance inst = allInstances.instance(i);
        // double predictedClassValue = singleNode.predictAndLearn(inst);
        //
        // String predictedClass = inst.classAttribute().value((int)
        // predictedClassValue);
        // String actualClass = inst.stringValue(inst.classAttribute());
        //
        // System.out.println("Predicted: " + predictedClass + " Actual: " +
        // actualClass + " Confidence: " + singleNode.computeNet() + "\n");
        // }
//			}
    }

    private static void writeOutputToStream(PrintStream out) {
        if (debugLevel <= 4) {
            System.out.println("Output Size: " + outputLines.size());
        }
        for (int i = 0; i < outputLines.size(); i++) {
            out.println(outputLines.get(new Integer(i)));
        }

    }

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


    public static double trainFolds(List<LinkedList<Instance>> folds, int testFoldIndex, ANNode singleNode, int epochNum) {
        double accuracy = 0.0;
        int numFolds = folds.size();
        int tp = 0, tn = 0, fp = 0, fn = 0;

        for (int foldIndex = 0; foldIndex < numFolds; foldIndex++) {
            List<Instance> instanceList = folds.get(foldIndex);
            if (foldIndex != testFoldIndex) {
                for (int i = 0; i < instanceList.size(); i++) {
                    Instance inst = instanceList.get(i);
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
            }
        }
        accuracy = (double) (tp + tn) / (double) (tp + tn + fp + fn);
        if (debugLevel <= 2) {
            System.out.println("Train Accuracy : " + accuracy + " TP: " + tp + " FP: " + fp + " TN: " + tn + " FN: " + fn);
        }
        if (epochNum == 1 || epochNum == 10 || epochNum == 100 || epochNum == 1000)
            predictStats.add("TestIndex:\t" + testFoldIndex + "\tEpoch:\t" + epochNum + "\tTrain Accuracy :\t" + accuracy + "\tTP:\t" + tp + "\tFP:\t" + fp + "\tTN:\t" + tn + "\tFN:\t" + fn);

        return accuracy;
    }

    public static double test(Instances instances, ANNode singleNode) {
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
        }

        accuracy = (double) (tp + tn) / (double) (tp + tn + fp + fn);
        if (debugLevel <= 2) {
            System.out.println("Train Accuracy : " + accuracy + " TP: " + tp + " FP: " + fp + " TN: " + tn + " FN: " + fn);
        }

        return accuracy;
    }


    public static double testFolds(List<LinkedList<Instance>> folds, int testFoldIndex, ANNode singleNode, int epochNum) {
        double accuracy = 0.0;
        int tp = 0, tn = 0, fp = 0, fn = 0;

        LinkedList<Instance> instanceList = folds.get(testFoldIndex);

        for (int i = 0; i < instanceList.size(); i++) {
            Instance inst = instanceList.get(i);
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

            // Lookup was failing in Instances..
            Integer opKey = StatisticsHelper.originalInstPosMap.get(inst);
            if (opKey == null) {
                System.out.println("\n\n\nNot able to retireve from originalInstPosMAp!!!!\n\n");
            }
            // if(outputLines.get(opKey) != null)
            // {
            // String oldVal = outputLines.get(opKey);
            // System.out.println("\n\n\nPre-existing val in outputLines!!!\n");
            // }

            String op = (testFoldIndex + 1) + "\t" + predictedClass + "\t" + actualClass + "\t" + singleNode.computeOutput(singleNode.computeNet());
            outputLines.put(opKey, op);
            if (debugLevel <= 4) {
                System.out.println("opKey (original index): " + opKey + "\tLine: " + op);
            }

        }

        accuracy = (double) (tp + tn) / (double) (tp + tn + fp + fn);
        if (debugLevel <= 2) {
            System.out.println("Test Accuracy : " + accuracy + " TP: " + tp + " FP: " + fp + " TN: " + tn + " FN: " + fn);
        }
        if (epochNum == 1 || epochNum == 10 || epochNum == 100 || epochNum == 1000)
            predictStats.add("TestIndex:\t" + testFoldIndex + "\tEpoch:\t" + epochNum + "\tTest Accuracy :\t" + accuracy + "\tTP:\t" + tp + "\tFP:\t" + fp + "\tTN:\t" + tn + "\tFN:\t" + fn);

        return accuracy;
    }

}
