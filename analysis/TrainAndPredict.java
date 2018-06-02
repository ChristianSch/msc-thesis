
import meka.classifiers.multilabel.PCC;
import meka.core.MLUtils;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.Logistic;
import java.io.FileWriter;


public class TrainAndPredict {

  public static void main(String[] args) throws Exception {
    if (args.length != 2)
      throw new IllegalArgumentException("Required arguments: <train> <predict>");

    System.out.println("Loading train: " + args[0]);
    Instances trainData = DataSource.read(args[0]);
    MLUtils.prepareData(trainData);

    System.out.println("Loading predict: " + args[1]);
    Instances predictData = DataSource.read(args[1]);
    MLUtils.prepareData(predictData);

    // compatible?
    String msg = trainData.equalHeadersMsg(predictData);
    if (msg != null)
      throw new IllegalStateException(msg);

    System.out.println("Build BR classifier on " + args[0]);
    PCC classifier = new PCC();
    classifier.setClassifier(new Logistic());
    // further configuration of classifier
    classifier.buildClassifier(trainData);

    System.out.println("Use BR classifier on " + args[1]);

    FileWriter fw_p = new FileWriter("../data/predictions_meka_pcc_p.csv");
    FileWriter fw_pp = new FileWriter("../data/predictions_meka_pcc_pp.csv");

    for (int i = 0; i < predictData.numInstances(); i++) {
        double[] dist = classifier.distributionForInstance(
                predictData.instance(i));

        fw_p.write(Utils.arrayToString(dist));
        fw_p.write("\n");

        fw_pp.write(Utils.arrayToString(classifier.probabilityForInstance(
                        predictData.instance(i), dist)));
        fw_pp.write("\n");
    }

    fw_p.close();
    fw_pp.close();
  }

}
