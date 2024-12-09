package BaseModel;

import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import java.util.Random;

public class NaiveBayesClassifier {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\customers_data.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Create and build the classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(dataset);

        System.out.println("NB params" + String.join(" ", nb.getOptions()));

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(nb, dataset, 10, new Random(42));

        // Print the confusion matrix
        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

        // Print additional evaluation metrics
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toClassDetailsString());

        SerializationHelper.write("C:\\Users\\tonga\\IdeaProjects\\DataMining - Copy\\src\\BayesClassifier\\Naive_Bayes.model", nb);
        SerializationHelper.read("C:\\Users\\tonga\\IdeaProjects\\DataMining - Copy\\src\\BayesClassifier\\Naive_Bayes.model");
    }
}
