package edu.byu.cs478.toolkit;

import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;
import java.util.List;

public abstract class UnsupervisedLearner implements Learner {

  public abstract void apply(Matrix data);

}
