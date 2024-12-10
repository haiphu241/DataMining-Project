package BaseModel;

import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.Logistic;
import java.util.Random;

public class LogisticClassifier {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("Data\\customers_data.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Create and build the classifier
        Logistic log = new Logistic();
        log.buildClassifier(dataset);

        System.out.println("Logistic params: " + String.join(" ", log.getOptions()));

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(log, dataset, 10, new Random(1));

        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

        // Print additional evaluation metrics
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toClassDetailsString());
    }
}