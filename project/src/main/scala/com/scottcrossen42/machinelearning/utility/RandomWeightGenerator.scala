package com.scottcrossen42.machinelearning.utility

import java.util.Random

object RandomWeightGenerator {
  private[this] var random: Random = new Random()

  def set(rand: Random) = {
    random = rand
  }

  def getRandomWeight: Double = random.nextDouble() - 0.5

  def getRandomInt(bound: Int): Int = random.nextInt(bound)

  def getRand: Random = random
}
