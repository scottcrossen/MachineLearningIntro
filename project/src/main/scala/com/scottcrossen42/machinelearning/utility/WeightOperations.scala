package com.scottcrossen42.machinelearning.utility

object WeightOperations {

  def calculateSingleWeightDelta(
    learningRate: Double,
    outputError: Double,
    input: Double,
    previousWeightDelta: Double,
    momentum: Double
  ): Double = (previousWeightDelta * momentum) + (learningRate * outputError * input)

  def calculateWeightDelta(
    learningRate: Double,
    outputError: Double,
    inputVector: List[Double],
    previousWeightDeltaVector: List[Double],
    momentum: Double
  ): List[Double] = inputVector.zip(previousWeightDeltaVector).map{ iter: Tuple2[Double, Double] =>
    calculateSingleWeightDelta(
      learningRate,
      outputError,
      iter._1,
      iter._2,
      momentum)
  }

  def randomWeightVector(size: Int): List[Double] =
    (0 to size - 1).map(_ => RandomWeightGenerator.getRandomWeight).toList

  def elementMultiply(firstList: List[Double], secondList: List[Double]): List[Double] = {
    require(firstList.size == secondList.size)
    firstList.zip(secondList).map {case (val1: Double, val2: Double) => val1 * val2}
  }

  def elementAdd(firstList: List[Double], secondList: List[Double]): List[Double] = {
    require(firstList.size == secondList.size)
    firstList.zip(secondList).map {case (val1: Double, val2: Double) => val1 + val2}
  }

  def elementSubtract(firstList: List[Double], secondList: List[Double]): List[Double] = {
    require(firstList.size == secondList.size)
    firstList.zip(secondList).map {case (val1: Double, val2: Double) => val1 - val2}
  }

  def dotProduct(firstList: List[Double], secondList: List[Double]): Double = {
    elementMultiply(firstList, secondList).sum
  }
}
