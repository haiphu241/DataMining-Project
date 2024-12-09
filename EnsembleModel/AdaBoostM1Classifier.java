package EnsembleModel;

import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class AdaBoostM1Classifier {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\customers_data.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        AdaBoostM1 adaBoostM1 = new AdaBoostM1();
        adaBoostM1.buildClassifier(dataset);

        System.out.println("AdaBoostM1 params: " + String.join(" ", adaBoostM1.getOptions()));

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(adaBoostM1, dataset, 10, new Random(1));

        // Print the confusion matrix
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
