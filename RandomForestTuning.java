import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import java.util.Random;


public class RandomForestTuning {
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource trainSource = new ConverterUtils.DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\training_data.arff");
        Instances trainDataset = trainSource.getDataSet();

        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

        String[] options = new String[4];
        options[0] = "-P"; options[1] = "50";
        options[2] = "-I"; options[3] = "50";

        // Create and train the OneR classifier
        RandomForest randomForest = new RandomForest();
        randomForest.setOptions(options);
        randomForest.buildClassifier(trainDataset);

        Evaluation eval = new Evaluation(trainDataset);
        eval.crossValidateModel(randomForest, trainDataset, 10, new Random(42));

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
