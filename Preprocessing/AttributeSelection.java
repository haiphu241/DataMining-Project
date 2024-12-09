package Preprocessing;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;

public class AttributeSelection {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\customers_data.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // Evaluate feature importance using information gain
        weka.attributeSelection.AttributeSelection attrSelection = new weka.attributeSelection.AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(10);
        attrSelection.setEvaluator(eval);
        attrSelection.setSearch(ranker);
        attrSelection.SelectAttributes(data);

        // Get selected attributes
        int[] selectedAttributes = attrSelection.selectedAttributes();

        // Create a Remove filter to keep only the selected attributes
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(selectedAttributes);
        remove.setInvertSelection(true);
        remove.setInputFormat(data);

        // Apply the filter
        Instances selectedData = Filter.useFilter(data, remove);

        // Save the selected attributes to a new ARFF file
        ArffSaver saver = new ArffSaver();
        saver.setInstances(selectedData);
        saver.setFile(new File("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\customers_data.arff"));
        saver.writeBatch();

        // Print results
        System.out.println("Selected Attributes:");
        for (int attr : selectedAttributes) {
            System.out.println(data.attribute(attr).name());
        }
    }
}