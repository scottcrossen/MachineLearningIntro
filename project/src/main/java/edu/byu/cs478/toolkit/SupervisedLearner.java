package edu.byu.cs478.toolkit;
// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;
import java.util.List;

public abstract class SupervisedLearner {

	// Before you call this method, you need to divide your data
	// into a feature matrix and a label matrix.
	public abstract void train(Matrix features, Matrix labels) throws Exception;

	// A feature vector goes in. A label vector comes out. (Some supervised
	// learning algorithms only support one-dimensional label vectors. Some
	// support multi-dimensional label vectors.)
	public abstract double[] predict(double[] features) throws Exception;

	// The model must be trained before you call this method. If the label is nominal,
	// it returns the predictive accuracy. If the label is continuous, it returns
	// the root mean squared error (RMSE). If confusion is non-NULL, and the
	// output label is nominal, then confusion will hold stats for a confusion matrix.
	public double measureAccuracy(Matrix features, Matrix labels, Matrix confusion) throws Exception
	{
		if(features.rows() != labels.rows())
			throw(new Exception("Expected the features and labels to have the same number of rows"));
		if(labels.cols() != 1)
			throw(new Exception("Sorry, this method currently only supports one-dimensional labels"));
		if(features.rows() == 0)
			throw(new Exception("Expected at least one row"));

		int labelValues = labels.valueCount(0);
		if(labelValues == 0) // If the label is continuous...
		{
			// The label is continuous, so measure root mean squared error
			double[] pred = new double[1];
			double sse = 0.0;

      double[] results = this.predictOnAllValues(features);
      for(int i = 0; i < results.length; i++) {
        double[] targ = labels.row(i);
        double delta = targ[0] - results[i];
        sse += (delta * delta);
      }
			return Math.sqrt(sse / features.rows());
		}
		else
		{
			// The label is nominal, so measure predictive accuracy
			if(confusion != null)
			{
				confusion.setSize(labelValues, labelValues);
				for(int i = 0; i < labelValues; i++)
					confusion.setAttrName(i, labels.attrValue(0, i));
			}
			int correctCount = 0;
      double[] results = this.predictOnAllValues(features);
			for(int i = 0; i < results.length; i++)
			{
				int targ = (int)labels.get(i, 0);
				if(targ >= labelValues)
					throw new Exception("The label is out of range");
				int pred = (int)results[i];
				if(confusion != null)
					confusion.set(targ, pred, confusion.get(targ, pred) + 1);
				if(pred == targ)
					correctCount++;
			}
			return (double)correctCount / features.rows();
		}
	}

  protected double[] predictOnAllValues(Matrix features) throws Exception {
    double[] output = new double[features.rows()];
    for(int i = 0; i < features.rows(); i++)
    {
      double[] feat = features.row(i);
      output[i] = predict(feat)[0];
    }
    return output;
  }

}
