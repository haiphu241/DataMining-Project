package Preprocessing;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;

public class CSV2Arff {
    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\customers_data.csv"));
        Instances data = loader.getDataSet();

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\customers_data.arff"));
        saver.writeBatch();
    }
}