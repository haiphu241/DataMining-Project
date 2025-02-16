package BaseModel;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.Logistic;
import java.util.Random;
import weka.core.SerializationHelper;

public class LogisticTuning {
    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        DataSource source = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\InfoGain_data.arff");
        Instances dataset = source.getDataSet();

        // Set class index to the last attribute (target variable)
        dataset.setClassIndex(dataset.numAttributes() - 1);

        String[] options = new String[6];
        options[0] = "-R"; options[1] = "1.0E-16";
        options[2] = "-M"; options[3] = "4";
        options[4] = "-num-decimal-places"; options[5] = "4";

        Logistic logistic = new Logistic();
        logistic.setOptions(options);
        logistic.buildClassifier(dataset);
        System.out.println("Logistic Selected Parameters: " + String.join(" ", logistic.getOptions()));

        // Evaluate the tuned classifier using the test dataset
        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(logistic, dataset, 10, new Random(42));

        // Print the confusion matrix
        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

        // Print additional evaluation metrics
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toClassDetailsString());

        SerializationHelper.write("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Model\\LogisticTuningBinaryModel.model", logistic);

        long endTime = System.nanoTime();

        long duration = endTime - startTime;
        System.out.println("Runtime: " + duration + " nanoseconds");
    }
}
