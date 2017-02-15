import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

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

        learnOr();

        learnAnd();


//        double[] u = {1, 2, 3, 4 };
//        double[] v = {2, 4 };
//        double[] w = {2, 3, 4 };
//
//        RealMatrix crossProd = new ArrayRealVector(v).outerProduct(new ArrayRealVector(u));
//        System.out.println("(" + crossProd.getRowDimension()+","+crossProd.getColumnDimension()+")");
//        System.out.println(crossProd.toString());

    }

    public static void learnOr() {

        NeuralNetwork orNN = new NeuralNetwork(2, 1, 0.1);
        orNN.intialize();

        System.out.println("Before training");
        orNN.printWeightArrays();

        Random random = new Random();

        for(int i=0;i<1000;i++) {
            boolean a = random.nextBoolean();
            boolean b = random.nextBoolean();

            boolean op = a || b;

            double input[] = new double[2];
            input[0] = a==true?1.0:0.0;
            input[1] = b==true?1.0:0.0;

            double output[] = new double[1];
            output[0] = op==true?1.0:0.0;

            orNN.predictAndTrain(input, false, output);

            if(i%100 == 0) {
                System.out.println("Epoch: "+i);
                orNN.printWeightArrays();
            }
        }
    }

    public static void learnAnd() {

        NeuralNetwork orNN = new NeuralNetwork(2, 1, 0.1);
        orNN.intialize();

        System.out.println("Before training");
        orNN.printWeightArrays();

        Random random = new Random();

        for(int i=0;i<100000;i++) {
            boolean a = random.nextBoolean();
            boolean b = random.nextBoolean();

            boolean op = a && b;

            double input[] = new double[2];
            input[0] = a==true?1.0:0.0;
            input[1] = b==true?1.0:0.0;

            double output[] = new double[1];
            output[0] = op==true?1.0:0.0;

            orNN.predictAndTrain(input, false, output);

            if(i%100 == 0) {
                System.out.println("Epoch: "+i);
                orNN.printWeightArrays();
            }
        }
    }


}
