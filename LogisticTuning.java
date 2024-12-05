import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.Logistic;
import java.util.Random;

public class LogisticTuning {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource trainSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\training_data.arff");
        Instances trainDataset = trainSource.getDataSet();

        // Set class index to the last attribute (target variable)
        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

        String[] options = new String[4];
        options[0] = "-R"; options[1] = "1.0E-2";
        options[2] = "-M"; options[3] = "5";

        Logistic logistic = new Logistic();
        logistic.setOptions(options);
        logistic.buildClassifier(trainDataset);

        // Evaluate the tuned classifier using the test dataset
        Evaluation eval = new Evaluation(trainDataset);
        eval.crossValidateModel(logistic, trainDataset, 10, new Random(42));

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
