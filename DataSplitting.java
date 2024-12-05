import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.util.Random;

public class DataSplitting {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\employees_data.arff");
        Instances data = source.getDataSet();

        // Set class index to the last attribute if not already set
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Shuffle the data to ensure randomness
        data.randomize(new Random(1)); // Set seed for reproducibility

        // Calculate split sizes
        int totalInstances = data.numInstances();
        int trainSize = (int) Math.round(totalInstances * 0.7); // 70% training
        int testSize = totalInstances - trainSize;              // 30% testing

        // Create subsets
        Instances trainData = new Instances(data, 0, trainSize);
        Instances testData = new Instances(data, trainSize, testSize);

        // Save training set
        ArffSaver trainSaver = new ArffSaver();
        trainSaver.setInstances(trainData);
        trainSaver.setFile(new File("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\training_data.arff"));
        trainSaver.writeBatch();

        // Save test set
        ArffSaver testSaver = new ArffSaver();
        testSaver.setInstances(testData);
        testSaver.setFile(new File("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\testing_data.arff"));
        testSaver.writeBatch();

        System.out.println("Data successfully split into training and testing sets.");
    }
}
