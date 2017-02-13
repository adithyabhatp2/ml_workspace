/**
 * Created by adbhat on 2/12/17.
 */
public class Lab2 {

    public static void main(String args[]) {
        NeuralNetwork myNN = new NeuralNetwork(4, 3);
        myNN.addHiddenLayer(2);
        myNN.intialize();


        double[] input = {1, 2.5, 5, 10};
        myNN.predictAndTrain(input, true);
    }
}
