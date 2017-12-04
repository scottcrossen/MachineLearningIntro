package com.scottcrossen42.machinelearning.intro.main;

import java.util.concurrent.TimeUnit;
import edu.byu.cs478.toolkit.MLSystemManager;

public class Main {

  public static final String dataDir = "/datasets/";
  private static MLSystemManager manager = new MLSystemManager();

  public static void main(String[] args) {
    boolean fail = false;
    while (!fail) {
      System.out.println("===================================");
      System.out.println("Monitor calling test methods");
      try {
        String[] managerArgs1 = {"-L", "baseline", "-A", dataDir + "iris.arff", "-E", "training"};
        //manager.main(managerArgs1);
        String[] managerArgs2 = {"-L", "perceptron", "-A", dataDir + "custom0.arff", "-E", "training"};
        //manager.main(managerArgs2);
        String[] managerArgs3 = {"-L", "perceptron", "-A", dataDir + "custom1.arff", "-E", "training"};
        //manager.main(managerArgs3);
        String[] managerArgs4 = {"-L", "perceptron", "-A", dataDir + "custom2.arff", "-E", "training"};
        //manager.main(managerArgs4);
        String[] managerArgs5 = {"-L", "perceptron", "-A", dataDir + "voting.arff", "-E", "random", ".7"};
        //manager.main(managerArgs5);
        String[] managerArgs6 = {"-L", "perceptron", "-A", dataDir + "iris.arff", "-E", "random", ".7"};
        //manager.main(managerArgs6);S
        String[] managerArgs7 = {"-L", "Backpropagation", "-A", dataDir + "iris.arff", "-E", "random", ".75"};
        //manager.main(managerArgs7);
        String[] managerArgs8 = {"-L", "Backpropagation", "-A", dataDir + "vowel.arff", "-E", "random", ".75"};
        //manager.main(managerArgs8);
        String[] managerArgs9 = {"-L", "decisiontree", "-A", dataDir + "lenses.arff", "-E", "random", ".75"};
        //manager.main(managerArgs9);
        String[] managerArgs10 = {"-L", "decisiontree", "-A", dataDir + "cars.arff", "-E", "cross", "10"};
        //manager.main(managerArgs10);
        String[] managerArgs11 = {"-L", "decisiontree", "-A", dataDir + "voting_full.arff", "-E", "cross", "10"};
        //manager.main(managerArgs11);
        String[] managerArgs12 = {"-L", "knn", "-A", dataDir + "mt_train.arff", "-E", "static", dataDir + "mt_test.arff",};
        //manager.main(managerArgs12);
        String[] managerArgs13 = {"-L", "knn", "-A", dataDir + "mt_train_small.arff", "-E", "static", dataDir + "mt_test_small.arff",};
        //manager.main(managerArgs13);
        String[] managerArgs14 = {"-L", "knn", "-N", "-A", dataDir + "mt_train_small.arff", "-E", "static", dataDir + "mt_test_small.arff",};
        //manager.main(managerArgs14);
        String[] managerArgs15 = {"-L", "knn", "-N", "-A", dataDir + "housing_train.arff", "-E", "static", dataDir + "housing_test.arff",};
        //manager.main(managerArgs15);
        String[] managerArgs16 = {"-L", "knn", "-N", "-A", dataDir + "credit_a.arff", "-E", "random", ".90"};
        //manager.main(managerArgs16);
        String[] managerArgs17 = {"-L", "kmeans", "-A", dataDir + "laborWithID.arff", "-E", "unsupervised"};
        //manager.main(managerArgs17);
        String[] managerArgs18 = {"-L", "kmeans", "-N", "-A", dataDir + "laborWithID.arff", "-E", "unsupervised"};
        //manager.main(managerArgs18);
        String[] managerArgs19 = {"-L", "kmeans", "-A", dataDir + "sponge.arff", "-E", "unsupervised"};
        //manager.main(managerArgs19);
        String[] managerArgs20 = {"-L", "kmeans", "-A", dataDir + "iris_full.arff", "-E", "unsupervised"};
        //manager.main(managerArgs20);
        String[] managerArgs21 = {"-L", "kmeans", "-N", "-A", dataDir + "abalone500.arff", "-E", "unsupervised"};
        //manager.main(managerArgs21);
        TimeUnit.SECONDS.sleep(60);
      } catch (Exception b) {
        fail = true;
      }
    }
  }
}
