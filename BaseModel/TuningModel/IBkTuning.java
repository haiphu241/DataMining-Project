package BaseModel;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class IBkTuning {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("C:\\Users\\tonga\\IdeaProjects\\DataMining-Project\\Data\\customers_data.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        String[] options = new String[6];
        options[0] = "-K"; options[1] = "3";
        options[2] = "-W"; options[3] = "0";
        options[4] = "-A"; options[5] = "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.ManhattanDistance -R first-last\"";

        IBk ibk = new IBk();
        ibk.setOptions(options);
        ibk.buildClassifier(dataset);

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(ibk, dataset, 10, new Random(42));

        System.out.println("IBk Hyperparameters: " + String.join(" ", ibk.getOptions()));
        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toClassDetailsString());
    }
}
