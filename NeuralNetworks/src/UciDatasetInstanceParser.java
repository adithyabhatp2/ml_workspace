import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Scanner;

/**
 * Created by adbhat on 1/25/17.
 * https://weka.wikispaces.com/Programmatic+Use
 * Aimed at UCI data sets such wine, thoracic surgery, etc. (.data files)
 */
public class UciDatasetInstanceParser {

    public static Instances parseFromUciDataFile(String filePath) throws IOException {
        Scanner sc = new Scanner(Paths.get(filePath));

        boolean isNumFeaturesExtracted = false;
        int numFeatures = -1;

        boolean isFeatureProcessingDone = false;
        int numFeaturesProcessed = 0;

        boolean areClassLabelsExtracted = false;

        boolean isNumInstancesExtracted = false;
        int numInstances = -1;

        boolean isInstanceProcessingDone = false;
        int numInstancesProcessed = 0;

        FastVector fvClassVal = new FastVector(2);
        FastVector fvWekaAttributes = new FastVector();

        Instances instances = null;

        while (sc.hasNextLine()) {
            String line = sc.nextLine();

            if (line.trim().isEmpty()) {
                continue;
            }

            if (line.startsWith("//")) {
                continue;
            }

            // Number of Features
            if (!isNumFeaturesExtracted) {
                numFeatures = Integer.parseInt(line);
                isNumFeaturesExtracted = true;
                // System.out.println("Num Features: " + numFeatures);
                continue;
            }

            // Extract Features
            // TODO: All nominal features
            if (isNumFeaturesExtracted && numFeaturesProcessed < numFeatures) {
                String fName = line.split("-")[0].trim();
                String[] fVals = line.split("-")[1].trim().split(" ");

                FastVector fvNominalVal = new FastVector();
                for (String fVal : fVals) {
                    fvNominalVal.addElement(fVal);
                }

                Attribute attribute = new Attribute(fName, fvNominalVal);
                fvWekaAttributes.addElement(attribute);
                numFeaturesProcessed++;
                continue;
            }

            // Extract Class Labels
            //TODO: assumes exactly two classes.. fix later if generalizing.
            if (!areClassLabelsExtracted) {
                fvClassVal.addElement(line.trim());
                fvClassVal.addElement(sc.nextLine().trim());
                Attribute classAttribute = new Attribute("theClass", fvClassVal);
                fvWekaAttributes.insertElementAt(classAttribute, 0);
                areClassLabelsExtracted = true;
                continue;
            }

            // Num Examples, initialize Instances object
            if (!isNumInstancesExtracted) {
                numInstances = Integer.parseInt(line);
                isNumInstancesExtracted = true;
                // System.out.println("Num Examples: " + numInstances);

                instances = new Instances(Paths.get(filePath).getFileName().toString(), fvWekaAttributes, numInstances);
                instances.setClassIndex(0);

                continue;
            }

            // Extract Examples / Instances
            if (numInstancesProcessed < numInstances) {
                Instance example = new Instance(numFeatures + 1);
                Scanner egAttrSc = new Scanner(line);
                String egId = egAttrSc.next();
                // TODO: handling missing values.
                for (int i = 0; i < numFeatures + 1; i++) {
                    example.setValue((Attribute) fvWekaAttributes.elementAt(i), egAttrSc.next());
                }
                instances.add(example);
            }

        }
        return instances;
    }


    public static void main(String args[]) throws Exception {

        String filePath = "/u/a/d/adbhat/private/gitRepository/ml_workspace/NeuralNetworks/inputs/red-wine-quality-train.data";
        String thoracicTrainPath = "/u/a/d/adbhat/private/gitRepository/ml_workspace/NeuralNetworks/inputs/Thoracic_Surgery_Data_train.data";

        Instances instances = parseFromUciDataFile(thoracicTrainPath);

        System.out.println("Num Classes " + instances.numClasses());
        System.out.println("Num Attributes " + instances.numAttributes());
        System.out.println("Num Instances " + instances.numInstances());

        Instance first = instances.firstInstance();

        for(int i=1;i<21;i++) {
            System.out.println(first.value(i) + "\t" + first.stringValue(i));
        }

        // System.out.println(instances.toSummaryString());
        // System.out.println(instances.toString());

    }

}
