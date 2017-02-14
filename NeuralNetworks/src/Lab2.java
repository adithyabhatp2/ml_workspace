import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Created by adbhat on 2/12/17.
 */
public class Lab2 {

    public static void main(String args[]) {
        NeuralNetwork myNN = new NeuralNetwork(4, 3, 0.1);
        myNN.addHiddenLayer(2, ActivationFunctions.SIGMOID);
        myNN.intialize();


        double[] input = {1, 2.5, 5, 10};
        double[] outputLabels = {1, 2, 3};
        double[] prediction = myNN.predictAndTrain(input, false, outputLabels);

        System.out.println("Done");

//        double[] u = {1, 2, 3, 4 };
//        double[] v = {2, 4 };
//        double[] w = {2, 3, 4 };
//
//        RealMatrix crossProd = new ArrayRealVector(v).outerProduct(new ArrayRealVector(u));
//        System.out.println("(" + crossProd.getRowDimension()+","+crossProd.getColumnDimension()+")");
//        System.out.println(crossProd.toString());

    }
}
