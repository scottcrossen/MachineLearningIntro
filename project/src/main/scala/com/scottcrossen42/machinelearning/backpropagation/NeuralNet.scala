package com.scottcrossen42.machinelearning.backpropagation

import edu.byu.cs478.toolkit.Matrix
import com.scottcrossen42.machinelearning.utility.RandomWeightGenerator
import com.scottcrossen42.machinelearning.utility.WeightOperations
import scala.collection.JavaConverters._

case class NeuralNet(
  private val features: Int,
  private val outputs: Int,
  private val newHiddenLayer: Option[List[Neuron]] = None,
  private val newOutputLayer: Option[List[Neuron]] = None
) {
  val hiddenLayer: List[Neuron] = newHiddenLayer.getOrElse(List.fill(features * 2)(Neuron(features)))
  val outputLayer: List[Neuron] = newOutputLayer.getOrElse(List.fill(outputs)(Neuron(features * 2)))

  def getHiddenLayerOutputs(inputs: List[Double]): List[Double] = hiddenLayer.map( _.calcOutput(inputs) )
  def getOutputLayerOutputs(inputs: List[Double]): List[Double] = outputLayer.map( _.calcOutput(inputs) )

  def getOutputLayerErrors(expectedOutput: Double, actualOutput: List[Double]): List[Double] = (0 to outputLayer.size - 1).map { iter: Int =>
    val currentNeuron: Neuron = outputLayer(iter)
    currentNeuron.calcOutputNeuronError( if (expectedOutput == iter) 1 else 0, actualOutput(iter))
  }.toList

  def updateAllLayerWeights(featureOutputs: List[Double], hiddenLayerOutputs: List[Double], outputLayerErrors: List[Double], learningRate: Double, momentum: Double) = {
    val newHiddenLayer: List[Neuron] = updateHiddenLayerWeights(featureOutputs, hiddenLayerOutputs, outputLayerErrors, learningRate, momentum)
    val newOutputLayer: List[Neuron] = updateOutputLayerWeights(hiddenLayerOutputs, outputLayerErrors, learningRate, momentum)
    NeuralNet(features, outputs, Some(newHiddenLayer), Some(newOutputLayer))
  }

  private[this] def updateHiddenLayerWeights(originalInput: List[Double], hiddenLayerOutputs: List[Double], outputLayerErrors: List[Double], learningRate: Double, momentum: Double): List[Neuron] = (0 to hiddenLayer.size - 1).map { iter: Int =>
    val outputLayerWeightVector: List[Double] = outputLayer.map(_.getWeightVector(iter))
    val currentNeuron: Neuron = hiddenLayer(iter)
    val error: Double = currentNeuron.calcHiddenNeuronError(hiddenLayerOutputs(iter), outputLayerErrors, outputLayerWeightVector)
    currentNeuron.applyWeightChanges(originalInput, learningRate, error, momentum)
  }.toList

  private[this] def updateOutputLayerWeights(originalInput: List[Double], outputLayerErrors: List[Double], learningRate: Double, momentum: Double): List[Neuron] = (0 to outputLayer.size - 1).map { iter: Int =>
    val currentNeuron: Neuron = outputLayer(iter)
    val error: Double = outputLayerErrors(iter)
    currentNeuron.applyWeightChanges(originalInput, learningRate, error, momentum)
  }.toList

  def compareWith(other: NeuralNet): Double = {
    if (features != other.features || outputs != other.outputs || hiddenLayer.size != other.hiddenLayer.size || outputLayer.size != other.outputLayer.size) {
      0
    } else {
      hiddenLayer.zip(other.hiddenLayer).map{ (neurons: Tuple2[Neuron, Neuron]) =>
        neurons._1.compareWith(neurons._2)
      }.sum / hiddenLayer.size +
      outputLayer.zip(other.outputLayer).map{ (neurons: Tuple2[Neuron, Neuron]) =>
        neurons._1.compareWith(neurons._2)
      }.sum / outputLayer.size
    }
  }

  def debug = {
    println("Hidden layer:")
    hiddenLayer.foreach(_.debug)
    println("Output layer:")
    outputLayer.foreach(_.debug)
  }
}
