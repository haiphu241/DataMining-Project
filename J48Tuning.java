import weka.classifiers.trees.J48;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class J48Tuning {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource trainSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\training_data.arff");
        Instances trainDataset = trainSource.getDataSet();
        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

        String[] options = new String[4];
        options[0] = "-C"; options[1] = "0.5";
        options[2] = "-M"; options[3] = "2";

        J48 tree = new J48();
        tree.setOptions(options);

        Evaluation eval = new Evaluation(trainDataset);
        eval.crossValidateModel(tree, trainDataset, 10, new Random(42)); // 42 for reproducibility

        // Print evaluation results
        System.out.println("J48 Hyperparameters: " + String.join(" ", tree.getOptions()));
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
    }
}
