import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveDuplicates;
import weka.filters.unsupervised.attribute.Standardize;
import weka.core.converters.ArffSaver;
import java.io.File;


public class DataPreprocess {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\employees_data.arff");
        Instances data = source.getDataSet();

        // Set class index to the last attribute (target variable)
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Remove duplicates
        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        Instances dataNoDuplicates = Filter.useFilter(data, removeDuplicates);

        // Standardize features
        Standardize standardize = new Standardize();
        standardize.setInputFormat(dataNoDuplicates);
        Instances standardizedData = Filter.useFilter(dataNoDuplicates, standardize);

        // Save or use the preprocessed data
        ArffSaver saver = new ArffSaver();
        saver.setInstances(standardizedData);
        saver.setFile(new File("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\employees_data.arff"));
        saver.writeBatch();

        // Print some information about the preprocessed data
        System.out.println("Number of instances after removing duplicates: " + dataNoDuplicates.numInstances());
        System.out.println("Number of instances after balancing classes: " + standardizedData.numInstances());
    }
}
