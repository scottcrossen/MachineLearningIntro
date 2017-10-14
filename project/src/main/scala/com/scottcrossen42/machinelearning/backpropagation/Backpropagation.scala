package com.scottcrossen42.machinelearning.backpropagation

import edu.byu.cs478.toolkit.SupervisedLearner
import edu.byu.cs478.toolkit.Matrix
import java.util.Random
import com.scottcrossen42.machinelearning.utility.RandomWeightGenerator
import com.scottcrossen42.machinelearning.utility.WeightOperations
import scala.collection.JavaConverters._
import edu.byu.cs478.toolkit.BaselineLearner

class Backpropagation(rand: Random) extends SupervisedLearner {
  import Backpropagation._

  RandomWeightGenerator.set(rand)

  // Use a neural net for each label
  private[this] var currentNeuralNets: List[NeuralNet] = List[NeuralNet]()

  def train(features: Matrix, labels: Matrix) = {
    if (verbose) println("Starting to train on data")
    // Train the model for each label
    val (trainingFeatures: Matrix, validationFeatures: Matrix) = partitionMatrix(features, 1 - validationAmnt)
    val (trainingLabels: Matrix, validationLabels: Matrix) = partitionMatrix(labels, 1 - validationAmnt)
    currentNeuralNets = (0 to labels.cols - 1).map{ (iter: Int) =>
      val neuralNet: NeuralNet = getNeuralNet(features.cols, labels.valueCount(iter), iter)
      if (verbose) println("\nNeural Net initialized")
      val accuracyFunctionTrainingSet: (NeuralNet => Double) = getAccuracyFunction(trainingFeatures, trainingLabels, iter)
      val accuracyFunctionValidationSet: (NeuralNet => Double) = getAccuracyFunction(validationFeatures, validationLabels, iter)
      val meanSquaredErrorFunctionTrainingSet: (NeuralNet => Double) = getMeanSquaredErrorFunction(trainingFeatures, trainingLabels, iter)
      val meanSquaredErrorFunctionValidationSet: (NeuralNet => Double) = getMeanSquaredErrorFunction(validationFeatures, validationLabels, iter)
      val dataCollector = new DataCollector(
        List(accuracyFunctionTrainingSet,
        accuracyFunctionValidationSet,
        meanSquaredErrorFunctionTrainingSet,
        meanSquaredErrorFunctionValidationSet)
      )
      val (newNeuralNet: NeuralNet, finalDataCollector: DataCollector) = trainNeuralNet(trainingFeatures, trainingLabels, neuralNet, iter, dataCollector, meanSquaredErrorFunctionValidationSet)
      if (verbose) finalDataCollector.print
      newNeuralNet
    }.toList
  }

  def predict(features: Array[Double]): Array[Double] = {
    // Predict the same amount of labels as we have models.
    (0 to currentNeuralNets.size - 1).map{ (iter: Int) =>
      val neuralNet: NeuralNet = getNeuralNet(features.length, 1, iter)
      predictWithNeuralNet(features.toList, neuralNet)
    }.toArray
  }

  private[this] def getNeuralNet(features: Int, outputs: Int, labelNum: Int): NeuralNet = {
    if (labelNum >= currentNeuralNets.size) {
      // Net doesn't exist for label
      new NeuralNet(features, outputs)
    } else {
      // Net exists for label
      currentNeuralNets(labelNum)
    }
  }

}

object Backpropagation {

  private val validationAmnt = 0.25

  private val verbose = false

  private val learningRate = 0.1

  private val maxEpochs = 1000

  private val momentum = 0

  private val endEarlyEpochIterations = 10

  private def predictWithNeuralNet(features: List[Double], neuralNet: NeuralNet): Double = {
    val hiddenLayerOutputs: List[Double] = neuralNet.getHiddenLayerOutputs(features.toList)
    val outputLayerOutputs: List[Double] = neuralNet.getOutputLayerOutputs(hiddenLayerOutputs)
    val (_, bestIndex: Int) = (0 to outputLayerOutputs.size - 1).foldLeft((0.0, -1)) { (soFar: Tuple2[Double, Int], iter: Int) =>
      if (outputLayerOutputs(iter) > soFar._1) {
        (outputLayerOutputs(iter), iter)
      } else {
        soFar
      }
    }
    return bestIndex.toDouble
  }

  private def getAccuracyFunction(validationFeatures: Matrix, validationLabels: Matrix, currentLabel: Int): (NeuralNet => Double) = {
    (neuralNet: NeuralNet) => {
      val amntTotal: Int = validationFeatures.rows
      val amntCorrect: Int = (0 to validationFeatures.rows - 1).filter { iter: Int =>
        val predicted: Double = predictWithNeuralNet(validationFeatures.row(iter).toList, neuralNet)
        val expected: Double = validationLabels.row(iter)(currentLabel)
        predicted == expected
      }.size
      (amntCorrect * 1.0) / (amntTotal * 1.0)
    }
  }

  private def getMeanSquaredErrorFunction(validationFeatures: Matrix, validationLabels: Matrix, currentLabel: Int): (NeuralNet => Double) = {
    (neuralNet: NeuralNet) => {
      (0 to validationFeatures.rows - 1).map{ iter: Int =>
        val features: List[Double] = validationFeatures.row(iter).toList
        val label: Double = validationLabels.row(iter)(currentLabel)
        val hiddenLayerOutputs: List[Double] = neuralNet.getHiddenLayerOutputs(features.toList)
        val outputLayerOutputs: List[Double] = neuralNet.getOutputLayerOutputs(hiddenLayerOutputs)
        val squaredError: Double = (0 to outputLayerOutputs.size - 1).map{ iter: Int =>
          Math.pow((if (iter == label) 1 else 0) - outputLayerOutputs(iter), 2)
        }.sum
        squaredError
      }.sum / validationFeatures.rows
    }
  }

  private def trainNeuralNet(features: Matrix, labels: Matrix, neuralNet: NeuralNet, labelNum: Int, dataCollector: DataCollector, meanSquaredErrorFunction: (NeuralNet => Double)): (NeuralNet, DataCollector) = {
    val baseMSE: Double = meanSquaredErrorFunction.apply(neuralNet)
    val (_, newNeuralNet: NeuralNet, mse: Double, _, finalDataCollector: DataCollector) = (0 to maxEpochs - 1).foldLeft((neuralNet, neuralNet, baseMSE, 0, dataCollector)) { (soFar: Tuple5[NeuralNet, NeuralNet, Double, Int, DataCollector], epochNum: Int) =>
      val oldNet: NeuralNet = soFar._1
      val bssfNet: NeuralNet = soFar._2
      val bssfScore: Double = soFar._3
      val iterationsUnchanged: Int = soFar._4
      val dataCollector: DataCollector = soFar._5
      if (epochNum > 0) features.shuffle(RandomWeightGenerator.getRand, labels)
      val newNet: NeuralNet = presentEpoch(features, labels, oldNet, labelNum)
      val newNetScore: Double = meanSquaredErrorFunction.apply(newNet)
      val newDataCollector = dataCollector.addToValues(newNet)
      if (bssfScore <= newNetScore) {
        if (iterationsUnchanged + 1 >= endEarlyEpochIterations) {
          println(s"\nEnding training early. Neural Net did not improve beyond best for $endEarlyEpochIterations successive epochs on epoch #$epochNum")
          println(s"Final mean-squared-error $bssfScore")
          return (bssfNet, newDataCollector)
        } else {
          (newNet, bssfNet, bssfScore, iterationsUnchanged + 1, newDataCollector)
        }
      } else {
        (newNet, newNet, newNetScore, 0, newDataCollector)
      }
    }
    println(s"\nMax Epoch limit reached during training.")
    println(s"Final mean-squared-error $mse")
    return (newNeuralNet, finalDataCollector)
  }

  private[this] def presentEpoch(features: Matrix, labels: Matrix, neuralNet: NeuralNet, labelNum: Int): NeuralNet = {
    (0 to features.rows - 1).foldLeft(neuralNet) { (currentNet: NeuralNet, rowNum: Int) =>
      applyRowToTrainingModel(features.row(rowNum).toList, labels.get(rowNum, labelNum), currentNet)
    }
  }

  private[this] def applyRowToTrainingModel(row: List[Double], expected: Double, neuralNet: NeuralNet): NeuralNet = {
    val hiddenLayerOutputs: List[Double] = neuralNet.getHiddenLayerOutputs(row)
    val outputLayerOutputs: List[Double] = neuralNet.getOutputLayerOutputs(hiddenLayerOutputs)
    val outputLayerErrors: List[Double] = neuralNet.getOutputLayerErrors(expected, outputLayerOutputs)
    val newNeuralNet: NeuralNet = neuralNet.updateAllLayerWeights(row, hiddenLayerOutputs, outputLayerErrors, learningRate, momentum)
    newNeuralNet
  }

  private def partitionMatrix(matrix: Matrix, split: Double): Tuple2[Matrix, Matrix] = {
    val takeAmnt: Int = Math.floor(matrix.rows * split).toInt
    val matrix1: Matrix = new Matrix(matrix, 0, 0, takeAmnt, matrix.cols)
    val matrix2: Matrix = new Matrix(matrix, takeAmnt - 1, 0, matrix.rows - takeAmnt, matrix.cols)
    (matrix1, matrix2)
  }

}
