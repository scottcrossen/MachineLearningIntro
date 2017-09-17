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
        manager.main(managerArgs1);
        String[] managerArgs2 = {"-L", "perceptron", "-A", dataDir + "custom1.arff", "-E", "training"};
        manager.main(managerArgs2);
        String[] managerArgs3 = {"-L", "perceptron", "-A", dataDir + "iris.arff", "-E", "training"};
        manager.main(managerArgs3);
        TimeUnit.SECONDS.sleep(60);
      } catch (Exception b) {
        fail = true;
      }
    }
  }
}
