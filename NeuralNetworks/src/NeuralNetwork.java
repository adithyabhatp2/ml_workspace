import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.*;

/**
 * Created by adbhat on 2/12/17.
 * Note that inputs are also counted as a layer.
 */
public class NeuralNetwork {

    double learningRate = 0.1;

    int debugLevel = 2;

    public int numInputs, numOutputs, numHiddenLayers = 0, numLayers;

    ArrayList<Integer> numUnitsPerLayer;

    List<double[][]> crossLayerWts;
    List<String> activationFunctions;


    public NeuralNetwork(int numInputs, int numOutputs, double learningRate) {

        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.learningRate = learningRate;

        this.numUnitsPerLayer = new ArrayList<>();
        this.numUnitsPerLayer.add(numInputs);

        // length = numLayers -1 for the below two.
        this.crossLayerWts = new LinkedList<>();
        this.activationFunctions = new LinkedList<>();
    }

    /**
     * Add more specifications if necessary, such as activation function
     *
     * @param numUnits
     * @return
     */
    public int addHiddenLayer(int numUnits, String activationFunction) {
        numHiddenLayers++;
        this.numUnitsPerLayer.add(numUnits);
        this.activationFunctions.add(activationFunction);
        return numHiddenLayers;
    }

    /**
     * Create the weight arrays, in form wij | i=this layer's unit, j = prev layer outputs.
     */
    public void intialize() {

        // Output Layer stuff
        this.numUnitsPerLayer.add(numOutputs);
        this.activationFunctions.add(ActivationFunctions.SIGMOID);

        // Initialize weight arrays
        this.numLayers = this.numHiddenLayers + 2;
        for (int i = 1; i < this.numLayers; i++) {
            // one extra wt for the bias unit at the end of the w[j][i] row..
            double[][] crossLayerWt = new double[this.numUnitsPerLayer.get(i)][this.numUnitsPerLayer.get(i - 1) + 1];
            initializeToSmalldoubles(crossLayerWt);
            this.crossLayerWts.add(crossLayerWt);
        }

        if (debugLevel <= 2) {
            for (double layerWt[][] : this.crossLayerWts) {
                System.out.println("CrossLayerWt shape : (" + layerWt.length + "," + layerWt[0].length + ")");
            }
        }
    }


    public double[] predictAndTrain(double input[], boolean predictOnly, double[] labelVector) {

        if (input.length != this.numInputs) {
            throw new IllegalArgumentException("Incorrect number of inputs, expected " + this.numInputs);
        }

        // Predict
        List<double[]> outputVectors = this.computeOutputVectorsAtEachLayer(input);

        if (predictOnly) {
            return outputVectors.get(outputVectors.size()-1);
        }

        //Train and backprop
        if (labelVector.length != numOutputs) {
            throw new IllegalArgumentException("Incorrect length of labelVector for output, expected " + this.numOutputs);
        }


        //// Compute Errors
        List<double[]> errors = this.computeErrorsAtEachLayer(labelVector, outputVectors);

        //// Compute deltas
        List<double[][]> deltas = this.computeWeightDeltasForEachWeightMatrix(errors, outputVectors);

        //// Compute momentums / dropout


        //// Update weights
        List<double[][]> newCrossLayerWts = new LinkedList<>();
        for (int i = 1; i < this.numLayers; i++) {
            RealMatrix weightMatrix = MatrixUtils.createRealMatrix(this.crossLayerWts.get(i - 1));
            RealMatrix deltaMatrix = MatrixUtils.createRealMatrix(deltas.get(i - 1));

            RealMatrix newWtMatrix = weightMatrix.add(deltaMatrix); // TODO: add momentum term, etc
            newCrossLayerWts.add(newWtMatrix.getData());
        }
        // sanity check
        if (newCrossLayerWts.size() != crossLayerWts.size()) {
            throw new IllegalStateException("new cross layer wt array size != old");
        }


        //TODO verify end to end
        this.crossLayerWts = newCrossLayerWts;
        return outputVectors.get(outputVectors.size()-1);
    }


    private List<double[]> computeOutputVectorsAtEachLayer(double[] input) {

        List<double[]> outputVectors = new LinkedList<>();
        outputVectors.add(input); // Layer 0 - input

        double[] outputVector = input;

        for (int i = 1; i < this.numLayers; i++) {

            double[] featureVector = Arrays.copyOf(outputVector, outputVector.length + 1);
            featureVector[featureVector.length - 1] = 1; // bias

            double[][] weightArray = this.crossLayerWts.get(i - 1);
            RealMatrix weightMatrix = MatrixUtils.createRealMatrix(weightArray);

            double[] outputNet = weightMatrix.operate(featureVector);

            String activationFunction = this.activationFunctions.get(i - 1);

            // need to transform outputNet to outputVector
            if (activationFunction.equalsIgnoreCase(ActivationFunctions.SIGMOID)) {
                outputVector = ActivationFunctions.sigmoidOnVector(outputNet);
            }
            else if (activationFunction.equalsIgnoreCase(ActivationFunctions.RELU)) {
                outputVector = ActivationFunctions.reluOnVector(outputNet);
            }

            outputVectors.add(outputVector);
        }
        return outputVectors;
    }


    private List<double[]> computeErrorsAtEachLayer(double[] labelVector, List<double[]> outputVectors) {

        List<double[]> errors = new LinkedList<>();
        double[] outputVector = outputVectors.get(outputVectors.size()-1);

        // Output Layer
        errors.add(new ArrayRealVector(labelVector).subtract(new ArrayRealVector(outputVector)).toArray());
        RealVector nextError = new ArrayRealVector(errors.get(0));

        // Hidden Layers
        for (int i = this.numLayers - 2; i >= 1; i--) {
            RealVector featureVector = new ArrayRealVector(outputVectors.get(i)).append(1.0); // for bias input
            RealMatrix weightMatrix = MatrixUtils.createRealMatrix(this.crossLayerWts.get(i)); // [numOutputs x numFeatures]
            String activationFunction = this.activationFunctions.get(i - 1);
            RealVector error = null;

            RealMatrix wTranspMatrix = weightMatrix.transpose(); // each row - contributions of one feature to each of the outputs of the next layer. [numFeatures x numOutputs]
            RealVector errorContributionsByFeature = wTranspMatrix.operate(nextError);

            if (activationFunction.equalsIgnoreCase(ActivationFunctions.SIGMOID)) {
                error = featureVector.mapMultiply(-1).mapAdd(1).ebeMultiply(featureVector); // o(-o+1)
                error = error.ebeMultiply(errorContributionsByFeature);
            }
            else if (activationFunction.equalsIgnoreCase(ActivationFunctions.RELU)) {
                error = errorContributionsByFeature;
                double errArray[] = error.toArray();
                for(int j=0;i<errArray.length;j++) {
                    if(featureVector.getEntry(j) <0 ) {
                        errArray[j]=0;
                    }
                }
                error = new ArrayRealVector(errArray);
                //TODO : unsure.. check if this is correct for ReLU!
            }

            error = error.getSubVector(0, error.getDimension() - 1); // remove bias
            nextError = error;
            errors.add(0, error.toArray()); // so that finally is in same order as weights
        }
        errors.add(0, null); // dummy for input layer 0.
        return errors;
    }


    private List<double[][]> computeWeightDeltasForEachWeightMatrix(List<double[]> errors, List<double[]> outputVectors ){
        List<double[][]> wtDeltas = new LinkedList<>();
        // Hidden and output layers
        for (int i = 1; i < this.numLayers; i++) {
            RealVector featureVector = new ArrayRealVector(outputVectors.get(i - 1)).append(1.0);
            RealMatrix weightMatrix = MatrixUtils.createRealMatrix(this.crossLayerWts.get(i - 1));
            RealVector error = new ArrayRealVector(errors.get(i));

            // errors[i] x featureVectors [i] = op[i-1] - cross prod of these 2 vectors - gives us a matrix
            RealMatrix deltaMatrix = error.outerProduct(featureVector).scalarMultiply(this.learningRate); // cross product
            // sanity check
            if (deltaMatrix.getColumnDimension() != weightMatrix.getColumnDimension() || deltaMatrix.getRowDimension() != weightMatrix.getRowDimension()) {
                throw new IllegalArgumentException("mismatch between deltaMatrix and weight matrix shapes at index : " + i);
            }

            wtDeltas.add(deltaMatrix.getData());
        }
        return wtDeltas;
    }


    private void initializeToSmalldoubles(double[][] double2dArray) {
        Random randObj = new Random();
        for (double[] row : double2dArray) {
            for (int i = 0; i < row.length; i++) {
                row[i] = Math.random();
                // row[i] = 0.1; // Temp for testing
            }
        }
    }


    public void printWeightArrays() {
        for(int i=1;i<numLayers;i++) {
            System.out.println("Layer: "+(i-1)+"to Layer: "+i);
            System.out.println(Arrays.deepToString(this.crossLayerWts.get(i-1)));
        }
    }

}
