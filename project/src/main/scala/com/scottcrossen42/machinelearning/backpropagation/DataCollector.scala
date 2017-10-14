package com.scottcrossen42.machinelearning.backpropagation

class DataCollector(
  val accuracyFunctionTrainingSet: (NeuralNet => Double),
  val accuracyFunctionValidationSet: (NeuralNet => Double),
  val meanSquaredErrorFunctionTrainingSet: (NeuralNet => Double),
  val meanSquaredErrorFunctionValidationSet: (NeuralNet => Double),
  val trainingAccuracy: List[Double] = List[Double](),
  val validationAccuracy: List[Double] = List[Double](),
  val trainingMse: List[Double] = List[Double](),
  val validationMse: List[Double] = List[Double]()
) {

  def addToValues(neuralNet: NeuralNet): DataCollector = {
    val newTrainingAccuracy = trainingAccuracy :+ accuracyFunctionTrainingSet.apply(neuralNet)
    val newValidationAccuracy = validationAccuracy :+ accuracyFunctionValidationSet.apply(neuralNet)
    val newTrainingMse = trainingMse :+ meanSquaredErrorFunctionTrainingSet.apply(neuralNet)
    val newValidationMse = validationMse :+ meanSquaredErrorFunctionValidationSet.apply(neuralNet)
    new DataCollector(
      accuracyFunctionTrainingSet,
      accuracyFunctionValidationSet,
      meanSquaredErrorFunctionTrainingSet,
      meanSquaredErrorFunctionValidationSet,
      newTrainingAccuracy,
      newValidationAccuracy,
      newTrainingMse,
      newValidationMse
    )
  }

  def print = {
    println("\nTraining Set Accuracy: ")
    println(trainingAccuracy)
    println("\nValidation Set Accuracy: ")
    println(validationAccuracy)
    println("\nTraining Set Mean Squared Error: ")
    println(trainingMse)
    println("\nValidation Set Mean Squared Error: ")
    println(validationMse)
  }
}
