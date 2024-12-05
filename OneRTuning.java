import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.rules.OneR;
import java.util.Random;

public class OneRTuning {
    public static void main(String[] args) throws Exception {
        DataSource trainSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\training_data.arff");
        Instances trainDataset = trainSource.getDataSet();

        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

        String[] options = new String[2];
        options[0] = "-B"; options[1] = "2";

        // Create and train the OneR classifier
        OneR oner = new OneR();
        oner.setOptions(options);
        oner.buildClassifier(trainDataset);

        Evaluation eval = new Evaluation(trainDataset);
        eval.crossValidateModel(oner, trainDataset, 10, new Random(42));

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
