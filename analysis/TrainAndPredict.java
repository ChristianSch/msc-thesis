
import meka.classifiers.multilabel.PCC;
import meka.core.MLUtils;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.Logistic;


public class TrainAndPredict {

  public static void main(String[] args) throws Exception {
    if (args.length != 2)
      throw new IllegalArgumentException("Required arguments: <train> <predict>");

    System.out.println("Loading train: " + args[0]);
    Instances train = DataSource.read(args[0]);
    MLUtils.prepareData(train);

    System.out.println("Loading predict: " + args[1]);
    Instances predict = DataSource.read(args[1]);
    MLUtils.prepareData(predict);

    // compatible?
    String msg = train.equalHeadersMsg(predict);
    if (msg != null)
      throw new IllegalStateException(msg);

    System.out.println("Build BR classifier on " + args[0]);
    PCC classifier = new PCC();
    classifier.setClassifier(new Logistic());
    // further configuration of classifier
    classifier.buildClassifier(train);

    System.out.println("Use BR classifier on " + args[1]);
    for (int i = 0; i < predict.numInstances(); i++) {
      double[] dist = classifier.distributionForInstance(predict.instance(i));
      System.out.println((i+1) + ": " + Utils.arrayToString(dist));
    }
  }

}
