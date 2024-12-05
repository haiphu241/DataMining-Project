import weka.classifiers.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class J48Tuning {

    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource trainSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\training_data.arff");
        Instances trainDataset = trainSource.getDataSet();
        DataSource testSource = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\testing_data.arff");
        Instances testDataset = testSource.getDataSet();

        // Set class index to the last attribute (target variable)
        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
        testDataset.setClassIndex(testDataset.numAttributes() - 1);

        // Set up CVParameterSelection to tune hyperparameters
        CVParameterSelection paramSelection = new CVParameterSelection();
        paramSelection.setClassifier(new J48());
        paramSelection.setNumFolds(10);
        // Define the parameter ranges for tuning
        paramSelection.addCVParameter("C 0.1 0.5 5");
        paramSelection.addCVParameter("M 1 20 1");
        paramSelection.buildClassifier(trainDataset);

        // Print the best parameters
        System.out.println("Best Parameters: " + String.join(" ", paramSelection.getBestClassifierOptions()));

        // Train the final model with the best parameters on the entire dataset
        J48 j48tuning = new J48();
        j48tuning.setOptions(paramSelection.getBestClassifierOptions());
        j48tuning.buildClassifier(trainDataset);

        // Evaluate the tuned classifier using the test dataset
        Evaluation eval = new Evaluation(trainDataset);
        eval.evaluateModel(j48tuning, testDataset);

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