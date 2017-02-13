import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.*;

/**
 * Created by adbhat on 2/12/17.
 * Note that inputs are also counted as a layer.
 */
public class NeuralNetwork {

    int debugLevel = 2;

    public int numInputs, numOutputs, numHiddenLayers=0, numLayers;

    ArrayList<Integer> numUnitsPerLayer;

    List<double [][]> crossLayerWts;
    List<String> activationFunctions;


    public NeuralNetwork(int numInputs, int numOutputs) {
        this.numHiddenLayers = numHiddenLayers;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;

        this.numUnitsPerLayer = new ArrayList<>();
        this.numUnitsPerLayer.add(numInputs);

        this.crossLayerWts = new LinkedList<>();
        this.activationFunctions = new LinkedList<>();
    }

    /**
     * Add more specifications if necessary, such as activation function
     * @param numUnits
     * @return
     */
    public int addHiddenLayer(int numUnits) {
        numHiddenLayers++;
        this.numUnitsPerLayer.add(numUnits);
        this.activationFunctions.add(ActivationFunctions.SIGMOID);
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
        this.numLayers = this.numHiddenLayers+2;
        for(int i=1;i<this.numLayers;i++) {
            // one extra wt for the bias unit at the end of the w[j][i] row..
            double[][] crossLayerWt = new double [this.numUnitsPerLayer.get(i)][this.numUnitsPerLayer.get(i-1)+1];
            initializeToSmalldoubles(crossLayerWt);
            this.crossLayerWts.add(crossLayerWt);
        }

        if(debugLevel <=2 ) {
            for(double layerWt[][] : this.crossLayerWts) {
                System.out.println("CrossLayerWt shape : (" + layerWt.length + "," + layerWt[0].length + ")");
            }
        }
    }


    public double[] predictAndTrain(double input[], boolean predictOnly) {
        if(input.length != this.numInputs) {
            throw new IllegalArgumentException("Incorrect number of inputs, expected "+this.numInputs);
        }

        double[] inputVector = null;
        double[] outputNet = null;
        double[] outputVector = input;

        List<double[]> outputs = new LinkedList<>();

        // Predict
        for(int i=1;i<this.numLayers;i++) {

            inputVector = Arrays.copyOf(outputVector, outputVector.length+1);
            inputVector[inputVector.length-1] = 1; // bias

            double[][] weightArray = this.crossLayerWts.get(i-1);
            String activationFunction = this.activationFunctions.get(i-1);

            if(debugLevel <=1 ) {
                System.out.println("\nwt array: "+Arrays.deepToString(weightArray));
                System.out.println("inputVector: "+Arrays.toString(inputVector));
            }

            if(debugLevel <=2 ) {
                System.out.println("\nwt array shape: "+weightArray.length+","+weightArray[0].length);
                System.out.println("inputVector shape: "+inputVector.length);
            }

            RealMatrix weightMatrix = MatrixUtils.createRealMatrix(weightArray);
            outputNet = weightMatrix.operate(inputVector);

            // need to transform outputNet to outputVector
            if(activationFunction.equalsIgnoreCase(ActivationFunctions.SIGMOID)) {
                outputVector = ActivationFunctions.sigmoidOnVector(outputNet);
            }
            else if(activationFunction.equalsIgnoreCase(ActivationFunctions.RELU)) {
                outputVector = ActivationFunctions.reluOnVector(outputNet);
            }

            outputs.add(outputVector);

            if(debugLevel <= 1) {
                System.out.println("outputVector: "+Arrays.toString(outputVector));
            }
            if(debugLevel <= 2) {
                System.out.println("outputVector shape: "+outputVector.length);
            }
        }


        if(predictOnly) {
            return outputVector;
        }

        //Train and backprop
        List<double[]> errors = new LinkedList<>();
        //TODO



    return outputVector;
    }





    /**
     * Initialize the 2d weight array to random values
     * @param double2dArray
     */
    private void initializeToSmalldoubles(double[][] double2dArray) {
        Random randObj = new Random();
        for(double[] row: double2dArray) {
            for(int i=0;i<row.length;i++) {
                row[i] = Math.random();
                // row[i] = 0.1; // Temp for testing
            }
        }
    }


}
