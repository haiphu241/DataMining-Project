package BaseModel;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.SMO;
import java.util.Random;

public class SVMTuning {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("Data\\customers_data.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        String[] options = new String[6];
        options[0] = "-C"; options[1] = "0.5";
        options[2] = "-batch-size"; options[3] = "50";
        options[4] = "-L"; options[5] = "0.1";

        SMO svm = new SMO();
        svm.setOptions(options);
        svm.buildClassifier(dataset);
        System.out.println("SVM Hyperparameters: " + String.join(" ", svm.getOptions()));


        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(svm, dataset, 10, new Random(42));

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
