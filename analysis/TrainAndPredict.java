import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Comparator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import static java.util.stream.Collectors.toList;

import java.util.Arrays;
import meka.classifiers.multilabel.PCC;
import meka.core.MLUtils;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.Logistic;


public class TrainAndPredict {

	public static void main(String[] args) throws Exception {
		if (args.length != 2)
			throw new IllegalArgumentException("Required arguments: <dataset> <percentage>");
        // take only first ten labels
        int k = 10;

		Instances data = DataSource.read(args[0]);
		MLUtils.prepareData(data);

		// reduce label space to 10 most frequent
        // fetch label frequencies
		double[] cards = MLUtils.labelCardinalities(data);

        // get new list of array of index, frequency per label
		List<double[]> res = IntStream.range(0, cards.length)
			.mapToObj(i -> new double[]{ i, cards[i] })
			.collect(Collectors.toList());

        // DEBUG:
		for (int i = 0; i < res.size(); i++) {
			System.out.println(Arrays.toString(res.get(i)));
		}
        System.out.println("----");

        // sort by frequency
        res.sort(Comparator.comparingDouble((double[] a) -> a[1]).reversed());
        res = res.subList(0, k);

        // retain only the top k labels per instance
        // TODO:

		double percentage = Double.parseDouble(args[1]);
        int trainSize = (int) (data.numInstances() * percentage / 100.0);
		Instances train = new Instances(data, 0, trainSize);
		Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);

		PCC classifier = new PCC();
		classifier.setClassifier(new Logistic());
		// further configuration of classifier
		classifier.buildClassifier(train);

		for (int i = 0; i < test.numInstances(); i++) {
			double[] dist = classifier.distributionForInstance(test.instance(i));
			System.out.println((i+1) + ": " + Utils.arrayToString(dist));
		}
	}

}
