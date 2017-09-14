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
        String[] managerArgs = {"-L", "baseline", "-A", dataDir + "iris.arff", "-E", "training"};
        manager.main(managerArgs);
        TimeUnit.SECONDS.sleep(5);
      } catch (Exception b) {
        fail = true;
      }
    }
  }
}
