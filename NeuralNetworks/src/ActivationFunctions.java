/**
 * Created by adbhat on 2/12/17.
 */
public class ActivationFunctions {

    public static String SIGMOID = "Sigmoid";
    public static String RELU = "ReLU";

    /**
     * Computes the sigmoid on each member and returns a new vector with those values.
     * @param inputVector vector of doubles
     * @return vector of same dimensions as input, containing the corresponding sigmoid values.
     */
    public static double[] sigmoidOnVector(double[] inputVector) {
        double output[] = new double[inputVector.length];
        for(int i=0;i<inputVector.length;i++) {
            output[i]=sigmoid(inputVector[i]);
        }
        return output;
    }

    public static double sigmoid(double val) {
        return (1.0 / (1.0 + Math.exp(-1.0 * val)));
    }

    public static double[] reluOnVector(double[] inputVector) {
        double output[] = new double[inputVector.length];
        for(int i=0;i<inputVector.length;i++) {
            output[i]=0; // TODO
        }
        return output;
    }
}
