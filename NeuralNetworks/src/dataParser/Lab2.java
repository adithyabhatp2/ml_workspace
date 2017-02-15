import com.sun.tools.javac.util.ArrayUtils;

import java.util.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.nio.file.Paths;
import java.io.IOException;

public class Lab2 {

	//Array to store test, tune and train sets
	static List<double[]> train_set_output = new LinkedList<>();
	static List<double[]> tune_set_output = new LinkedList<>();
	static List<double[]> test_set_output = new LinkedList<>();
	static List<double[]> train_set_input = new LinkedList<>();
	static List<double[]> tune_set_input = new LinkedList<>();
	static List<double[]> test_set_input = new LinkedList<>();

	static HashMap<String, double[]> features = new HashMap<String, double[]>();
	static HashMap<String, double[]> labels = new HashMap<String, double[]>(); 

	public static void generateUsingSliding(List<List<String>> input, List<List<String>> output, int window_size, int flag) {


		for (int i=0; i< input.size(); i++) {
			String[] protein_input = input.get(i).toArray(new String[input.get(i).size()]);
			String[] protein_output = output.get(i).toArray(new String[output.get(i).size()]);

			//System.out.println(Arrays.toString(protein_input));
			for (int j=0; j< protein_input.length; j++) {
				String[] currentWindow = new String[window_size];
				String currentWindowOutput = protein_output[j];
				Arrays.fill(currentWindow, "Z");
				for(int m=j-1, n = ((window_size-1)/2) - 1; n >=0 && m >=0; m--, n--) {
					currentWindow[n] = protein_input[m];
				}
				currentWindow[(window_size -1)/2] = protein_input[j];
				int upper_limit = Integer.min(j + (window_size -1)/2, protein_input.length - 1);
				for(int k=j+1, l= ((window_size-1)/2) + 1; k <= upper_limit; k++, l++) {
					currentWindow[l] = protein_input[k];
				}
				//System.out.println(Arrays.toString(currentWindow));

				double[] encoded_input = new double[21 * window_size]; //Hardcoded 21
				int index = 0;
				for(int z=0; z<currentWindow.length; z++) {
					double[] temp_array = features.get(currentWindow[z]);
					for (int y=0; y < 21; y++) {
						encoded_input[index] = temp_array[y];
						index++;
					}
				}
				double[] encoded_output = labels.get(currentWindowOutput);

				switch (flag){
					case 0:
						train_set_input.add(encoded_input);
						train_set_output.add(encoded_output);
						break;
					case 1:
						tune_set_input.add(encoded_input);
						tune_set_output.add(encoded_output);
						break;
					case 2:
						test_set_input.add(encoded_input);
						test_set_output.add(encoded_output);
						break;
				}
				//System.out.println(Arrays.toString(encoded_input));
				//System.out.println(index);
			}
		}
	}

	public static void generateDataset(String filePath){

	/* Pseudo code for dataset generation
		1. Open the file
		2. Skip lines starting with # or space
		3. <> indicates the start of a new protein sequence
		4. Extract the entire protein input and output in 2 1D arrays
		5. Append to global protein I/P and O/P arrays - This should be 128
		6. Put proteins 5, 10 and so on to tune set
		7. Put proteins 6, 11 and so on to test set
		8. Put remaining proteins to train set
		9. For each set above, form the sliding window inputs
		10. Apply one-hot encoding and return the final output
	*/

		System.out.println("Parsing data ....");
		List<List<String>> proteins_input = new ArrayList<>();
		List<List<String>> proteins_output = new ArrayList<>();
		int newProtein = 0;

		List<String> protein_input = new ArrayList<String>();
		List<String> protein_output = new ArrayList<String>();

		Set<String> input_features = new HashSet<String>();
		Set<String> output_labels = new HashSet<String>();

		Scanner fileScanner = null;
		try
		{
			fileScanner = new Scanner(Paths.get(filePath));
		} catch(IOException e) {
			System.err.println("Could not find file");
			System.exit(1);
		}	

		while (fileScanner.hasNextLine()) {
			String line = fileScanner.nextLine().trim();

			// Skip blank lines
			if (line.length() == 0) {
				continue;
			}

			// Skip comments
			if (line.startsWith("#")) {
				continue;
			}

			if (line.startsWith("end")) {
				continue;
			}

			if (line.startsWith("<end>")) {
				continue;
			}

			if (line.startsWith("<>")) {

				if (newProtein == 1) {
					//System.out.println(protein_input);
					proteins_input.add(protein_input);
					proteins_output.add(protein_output);
					//System.out.println(proteins_input);
					//System.out.println("------------------");
					protein_input = new LinkedList<String>();
					protein_output = new LinkedList<String>();
				}
				else {
					newProtein = 1;
				}
				continue;
			}
			String input = line.split(" ")[0].trim();
			String output = line.split(" ")[1].trim();

			protein_input.add(input);
			protein_output.add(output);

			input_features.add(input);
			output_labels.add(output);

		}
		proteins_input.add(protein_input);
		proteins_output.add(protein_output);
		//System.out.println("Total Protein Input Count " + proteins_input.size());
		//System.out.println("Total Protein Output Count " + proteins_output.size());

		List<List<String>> train_input = new ArrayList<>();
		List<List<String>> train_output = new ArrayList<>();
		List<List<String>> tune_input = new ArrayList<>();
		List<List<String>> tune_output = new ArrayList<>();
		List<List<String>> test_input = new ArrayList<>();
		List<List<String>> test_output = new ArrayList<>();

		for (int i = 0 ; i < proteins_input.size(); i++)
		{
			//System.out.println(proteins_input.get(i));
			if ((i+1) % 5 == 0) {
				tune_input.add(proteins_input.get(i));
				tune_output.add(proteins_output.get(i));
			}
			else if (i > 0 && i % 5 == 0) {
				test_input.add(proteins_input.get(i));
				test_output.add(proteins_output.get(i));				
			}
			else {
				train_input.add(proteins_input.get(i));
				train_output.add(proteins_output.get(i));		
			}
		}

		//System.out.println(proteins_input.get(proteins_input.size() - 1));
		//System.out.println("Train Protein Input Count " + train_input.size());
		//System.out.println("Train Protein Output Count " + train_output.size());
		//System.out.println("Tune Protein Input Count " + tune_input.size());
		//System.out.println("Tune Protein Output Count " + tune_output.size());
		//System.out.println("Test Protein Input Count " + test_input.size());
		//System.out.println("Test Protein Output Count " + test_output.size());


		input_features.add("Z");

		//System.out.println(input_features);
		//System.out.println(output_labels);
		String[] features_array = null;
		String[] labels_array = null;

		features_array = input_features.toArray(new String[input_features.size()]);
		labels_array = output_labels.toArray(new String[output_labels.size()]);

		for (int i=0; i < features_array.length; i++) {
			double[] encoding = new double[21];
			for (int j = 0; j <=20; j++) {
				if (j == i) {
					encoding[i] = 1.0;
				}
				else {
					encoding[j] = 0.0;
				}
			}
			//System.out.println(Arrays.toString(encoding));
			features.put(features_array[i], encoding);
		}

		/*
        for(Map.Entry entry:features.entrySet()) {

            System.out.println("Key="+entry.getKey()+", Value="+Arrays.toString((double[])entry.getValue()));
        } 
		*/
        for (int i=0; i < labels_array.length; i++) {
			double[] encoding = new double[3];
			for (int j = 0; j <=2; j++) {
				if (j == i) {
					encoding[i] = 1.0;
				}
				else {
					encoding[j] = 0.0;
				}
        	}
        	labels.put(labels_array[i], encoding);
		}

		/*
        for(Map.Entry entry:labels.entrySet()) {

            System.out.println("Key="+entry.getKey()+", Value="+Arrays.toString((double[])entry.getValue()));
        }
        */

        generateUsingSliding(train_input, train_output, 17, 0);
		generateUsingSliding(tune_input, tune_output, 17, 1);
		generateUsingSliding(test_input, test_output, 17, 2);
		System.out.println("Training Set Size: " + train_set_input.size());
		System.out.println("Tune Set Size: " + tune_set_input.size());
		System.out.println("Test Set Size: " + test_set_input.size());
	}

	public static void main(String args[]) throws IOException {

		if (args.length != 1) {
			System.out.println("Usage: java Lab1 <file-name>");
			System.exit(0);
		}

		String originalFile = args[0];
		generateDataset(originalFile);
	}
}