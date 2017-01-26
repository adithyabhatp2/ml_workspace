import java.util.Arrays;
import weka.core.Instance;

public class ANNode
	{
	double learningRate = 0.1;
	double[] weights;

	private double[] inputs; // [0] - weight for factor 1
	int debugLevel = 3;

	public ANNode(int numInputs, double learningRate)
		{
		inputs = new double[numInputs + 1]; // [0] - bias constant 1
		weights = new double[numInputs + 1];
		Arrays.fill(inputs, 0.1);
		Arrays.fill(weights, 0.1);
		inputs[0] = 1;
		this.learningRate = learningRate;
		}

	/**
	 * @param inst
	 * @return predicted value of inst.ClassValue() [nominal : index as double]
	 */
	public double predictAndLearn(Instance inst)
		{
		this.loadInputAttributes(inst);
		double net = this.computeNet();
		double output = computeOutput(net);

		// compute error
		double predictedClass = (output > 0.5)? 1.0 : 0.0;
		double actualClass = inst.classValue();
//		double error = computeError(actualClass, predictedClass, "Squared");
		// TODO : use output instead.. but not used here in paticular

		if(debugLevel <= 2)
			{
			System.out.println("Actual: " + actualClass + "\tOutput: " + output + " \tNet: " + net);
			}

		// stochastic gradient descent to minimize error
		// not optimizing - readability, understanding
		double partialErrors[] = new double[this.inputs.length];
		for (int i = 0;i < this.inputs.length;i++)
			{
			partialErrors[i] = (-1.0) * (actualClass - output); // partial
																// derivative
																// err by o
			partialErrors[i] *= (output) * (1 - output); // partial derivative o
															// by net
			partialErrors[i] *= this.inputs[i]; // partial derivative net by w
			}

		double weightChanges[] = new double[this.inputs.length];
		for (int i = 0;i < this.inputs.length;i++)
			{
			weightChanges[i] = -1.0 * this.learningRate * partialErrors[i];
			}

		if(debugLevel <= 1)
			{
			System.out.println("Partial Errors : \t" + Arrays.toString(partialErrors));
			System.out.println("Original Weights: \t" + Arrays.toString(this.weights));
			System.out.println("Weight Changes : \t" + Arrays.toString(weightChanges));
			}

		// separating just to enable debugging.. TODO: merge loops
		for (int i = 0;i < this.inputs.length;i++)
			{
			this.weights[i] += weightChanges[i];
			}

		return predictedClass;
		}

	private double computeError(double actual, double predicted, String errorType)
		{
		double err = 0.0;
		if(errorType.toLowerCase().trim().equals("squared"))
			{
			err = Math.pow(actual - predicted, 2) / 2.0;
			}
		return err;
		}

	public double predictOnly(Instance inst)
		{
		this.loadInputAttributes(inst);
		double net = this.computeNet();
		double output = computeOutput(net);

		// compute error
		double predictedClass = (output > 0.5)? 1.0 : 0.0;
		double actualClass = inst.classValue();
		double error = computeError(actualClass, predictedClass, "Squared");
		// TODO : check if output instead

		if(debugLevel <= 2)
			{
			System.out.println("Actual: " + actualClass + "\tOutput: " + output + " \tNet: " + net + "\tErr: " + error);
			}
		return predictedClass;
		}

	/** loads instance attributes into this.inputs array */
	private void loadInputAttributes(Instance inst)
		{
		int k = 1;
		for (int attIndex = 0;attIndex < inst.numAttributes();attIndex++)
			{
			if(attIndex != inst.classIndex())
				{
				this.inputs[k] = inst.value(attIndex);
				k++;
				}
			}
		}

	/**
	 * @return sigmoid(net)
	 */
	public double computeOutput(double net)
		{
		return sigmoid(net);
		}

	/**
	 * @return net = sigma(wt*input)
	 */
	public double computeNet()
		{
		double net = 0;
		for (int i = 0;i < this.inputs.length;i++)
			{
			net += (this.weights[i] * this.inputs[i]);
			}
		return net;
		}

	public double sigmoid(double val)
		{
		return(1.0 / (1.0 + Math.exp(-1.0 * val)));
		}

	}
