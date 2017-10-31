package com.scottcrossen42.machinelearning.backpropagation

import com.scottcrossen42.machinelearning.utility.RandomWeightGenerator
import com.scottcrossen42.machinelearning.utility.WeightOperations

case class Neuron(
  private val weightVectorSize: Int,
  private val newCurrentWeightVector: Option[List[Double]] = None,
  private val newCurrentWeightBias: Option[Double] = None,
  private val newPreviousWeightVectorDelta: Option[List[Double]] = None,
  private val newPreviousWeightBiasDelta: Option[Double] = None
) {
  private val currentWeightVector: List[Double] = newCurrentWeightVector.getOrElse(WeightOperations.randomWeightVector(weightVectorSize))
  private val currentWeightBias: Double = newCurrentWeightBias.getOrElse(RandomWeightGenerator.getRandomWeight)
  private val previousWeightVectorDelta: List[Double] = newPreviousWeightVectorDelta.getOrElse(List.fill(weightVectorSize)(0.0))
  private val previousWeightBiasDelta: Double = newPreviousWeightBiasDelta.getOrElse(0.0)

  def calcNet(input: List[Double]): Double = WeightOperations.dotProduct(input, currentWeightVector) + currentWeightBias
  def calcOutput(input: List[Double]): Double = (1.0 / (1 + Math.exp(-calcNet(input))))
  def calcGradient(output: Double): Double = output * (1 - output)
  def calcOutputNeuronError(expectedOutput: Double, actualOutput: Double): Double = calcGradient(actualOutput) * (expectedOutput - actualOutput)
  def calcHiddenNeuronError(actualOutput: Double, outputErrors: List[Double], weightsToOutputs: List[Double]): Double =
    calcGradient(actualOutput) * WeightOperations.dotProduct(outputErrors, weightsToOutputs)

  def getWeightVector: List[Double] = currentWeightVector

  def applyWeightChanges(inputVector: List[Double], learningRate: Double, outputError: Double, momentum: Double): Neuron = {
    val newPreviousWeightVectorDelta: List[Double] = WeightOperations.calculateWeightDelta(
      learningRate, outputError, inputVector, previousWeightVectorDelta, momentum)
    val newPreviousWeightBiasDelta: Double = WeightOperations.calculateSingleWeightDelta(
      learningRate, outputError, 1, previousWeightBiasDelta, momentum)
    val newCurrentWeightVector: List[Double] = WeightOperations.elementAdd(currentWeightVector, newPreviousWeightVectorDelta)
    val newCurrentWeightBias: Double = currentWeightBias + newPreviousWeightBiasDelta
    Neuron(weightVectorSize, Some(newCurrentWeightVector), Some(newCurrentWeightBias), Some(newPreviousWeightVectorDelta), Some(newPreviousWeightBiasDelta))
  }

  def compareWith(other: Neuron): Double = {
    if (weightVectorSize != other.weightVectorSize) 0
    else {
      WeightOperations.elementSubtract(currentWeightVector, other.currentWeightVector).foldLeft(0.0) { (soFar: Double, current: Double) =>
        soFar + Math.abs(current)
      } + Math.abs(currentWeightBias - other.currentWeightBias) / (weightVectorSize + 1)
    }
  }

  def debug = println(s"Weights: $currentWeightVector Bias: $currentWeightBias")
}
