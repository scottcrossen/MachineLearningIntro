package com.scottcrossen42.machinelearning.perceptron

import edu.byu.cs478.toolkit.SupervisedLearner
import edu.byu.cs478.toolkit.Matrix
import scala.math.{sqrt, floor, abs}
import java.util.Random
import scala.collection.JavaConverters._
import com.scottcrossen42.machinelearning.utility.WeightOperations

object Perceptron {
  val verbose = false

  val initialWeight = 0

  val trainingConstant = 0.1

  val threshold = 1

  val matchedPairsMethod = true

  val maxEpochs = 100

  val stopEpochEarly = false
}

class Perceptron(rand: Random) extends SupervisedLearner {
  import Perceptron._

  // column of label, pairing of label, weight vector
  private[this] var allWeightVector: Array[List[List[Double]]] = Array()

  // column of label, list of available labels
  private[this] var allDistinctVals: Array[List[Int]] = Array()

  def train(features: Matrix, labels: Matrix): Unit = if (matchedPairsMethod) {
    trainMatchedPairs(features, labels)
  } else {
    trainOnIndividual(features, labels)
  }

  private[this] def trainMatchedPairs(features: Matrix, labels: Matrix): Unit = {
    require(features.rows == labels.rows)
    (0 to labels.cols - 1).foreach { colNum: Int =>
      val (newWeightVector, newDistinctVals) = (if (labels.valueCount(colNum) == 0) {
        (List(), List())
      } else {
        val distinctVals = labels.uniqVals(colNum).toList.map(labels.attrValue(colNum, _))
        val amntValues = distinctVals.size
        val amntPairings = (amntValues - 1) * (amntValues) / 2
        val weightVectors = (0 to amntPairings - 1).map { iter: Int =>
          val (pair1: Int, pair2: Int) = unHashMatchedPairs(distinctVals, iter)
          val goodRows = (0 to features.rows - 1).filter((rowNum) => (labels.get(rowNum, colNum) == pair1 || labels.get(rowNum, colNum) == pair2)).toList
          val goodFeatures = new Matrix(features, goodRows.toSet.asJava, (0 to features.cols - 1).toSet.asJava)
          val goodLabels = new Matrix(labels, goodRows.toSet.asJava, (0 to labels.cols - 1).toSet.asJava)
          val newWeightVector = applyManyEpochs(colNum, goodFeatures, goodLabels, pair1)
          if (verbose) println(s"Perceptron between pairing ${labels.attrValue(colNum, pair1)} and ${labels.attrValue(colNum, pair2)} returned final weight vector $newWeightVector")
          newWeightVector
        }.toList
        (weightVectors, distinctVals)
      })
      allDistinctVals = allDistinctVals :+ newDistinctVals
      allWeightVector = allWeightVector :+ newWeightVector
    }
  }

  private[this] def trainOnIndividual(features: Matrix, labels: Matrix): Unit = {
    require(features.rows == labels.rows)
    (0 to labels.cols - 1).foreach { colNum: Int =>
      val (newWeightVector, newDistinctVals) = (if (labels.valueCount(colNum) == 0) {
        (List(), List())
      } else {
        val distinctVals = labels.uniqVals(colNum).toList.map(labels.attrValue(colNum, _))
        val amntValues = distinctVals.size
        val goodRows = (0 to features.rows - 1).toList
        val weightVector = (0 to amntValues - 1).map { iter: Int =>
          val pair1: Int = distinctVals(iter)
          val goodFeatures = new Matrix(features, goodRows.toSet.asJava, (0 to features.cols - 1).toSet.asJava)
          val goodLabels = new Matrix(labels, goodRows.toSet.asJava, (0 to labels.cols - 1).toSet.asJava)
          val newWeightVector = applyManyEpochs(colNum, goodFeatures, goodLabels, pair1)
          if (verbose) println(s"Perceptron for ${labels.attrValue(colNum, pair1)} returned final weight vector $newWeightVector")
          newWeightVector
        }.toList
        (weightVector, distinctVals)
      })
      allDistinctVals = allDistinctVals :+ newDistinctVals
      allWeightVector = allWeightVector :+ newWeightVector
    }
  }

  def predict(features: Array[Double]): Array[Double] = if (matchedPairsMethod) {
    predictMatchedPairs(features)
  } else {
    predictOnIndividual(features)
  }

  private[this] def predictMatchedPairs(features: Array[Double]): Array[Double] = {
    if(allWeightVector.size == 0) return Array[Double]()
    val output = (0 to allDistinctVals.size - 1).map { colNum: Int =>
      val distinctVals = allDistinctVals(colNum)
      val amntValues = distinctVals.size
      val amntPairings = (amntValues - 1) * (amntValues) / 2
      val winners = (0 to amntPairings - 1).map{ iter =>
        val weightVector: List[Double] = allWeightVector(colNum)(iter)
        val (pair1: Int, pair2: Int) = unHashMatchedPairs(distinctVals, iter)
        //println(s"pair1: $pair1, pair2: $pair2 features: ${addTerms(features.toList)} weightVector: $weightVector dotProd: ${WeightOperations.elementMultiply(addTerms(features.toList), weightVector).sum}")
        if (WeightOperations.elementMultiply(addTerms(features.toList), weightVector).sum > 0) pair1 else pair2
      }.toList
      distinctVals.maxBy{ x: Int =>
        winners.count { winner: Int =>
          winner == x
        }
      }.toDouble
    }.toArray
    return output
  }

  private[this] def predictOnIndividual(features: Array[Double]): Array[Double] = {
    if(allWeightVector.size == 0) return Array[Double]()
    val output = (0 to allDistinctVals.size - 1).map { colNum: Int =>
      val distinctVals = allDistinctVals(colNum)
      val amntValues = distinctVals.size
      val (bestVal: Int, bestScore: Double) = (0 to amntValues - 1).foldLeft((0,0.0)){ (soFar: Tuple2[Int,Double], iter: Int) =>
        val weightVector: List[Double] = allWeightVector(colNum)(iter)
        val pair1: Int = distinctVals(iter)
        //println(s"pair1: $pair1, features: ${addTerms(features.toList)} weightVector: $weightVector dotProd: ${WeightOperations.elementMultiply(addTerms(features.toList), weightVector).sum}")
        if (WeightOperations.elementMultiply(addTerms(features.toList), weightVector).sum > soFar._2) {
          (pair1, WeightOperations.elementMultiply(addTerms(features.toList), weightVector).sum)
        } else {
          soFar
        }
      }
      bestVal.toDouble
    }.toArray
    return output
  }

  private[this] def applyManyEpochs(colNum: Int, features: Matrix, labels: Matrix, correctPair: Int): List[Double] = {
    val startingWeight: Option[List[Double]] = None
    val (outputWeightVector: Option[List[Double]], _) = (0 to maxEpochs - 1).foldLeft((startingWeight, 0)) { (soFar: Tuple2[Option[List[Double]], Int], iter: Int) =>
      if (iter > 0) features.shuffle(rand, labels)
      val newWeightVector = presentEpoch(colNum, features, labels, correctPair)
      val noChange: Boolean = soFar._1.map(lastWeight => WeightOperations.elementSubtract(newWeightVector, lastWeight).sum < newWeightVector.size * trainingConstant * 0.2).getOrElse(false)
      if (noChange) {
        if (soFar._2 > 4) {
          if (verbose) println(s"Ending training early. Weight vector changed less than ${newWeightVector.size * trainingConstant * 0.2} for 5 successive epochs on epoch #$iter")
          return newWeightVector
        } else {
          (Some(newWeightVector), soFar._2 + 1)
        }
      } else {
        (Some(newWeightVector), 0)
      }
    }
    if (verbose) println(s"Max Epoch limit reached during training.")
    return outputWeightVector.getOrElse(List[Double]())
  }

  private[this] def presentEpoch(labelColumn: Int, features: Matrix, labels: Matrix, correctPair: Int): List[Double] = {
    val featureCols = features.cols + /*(features.cols - 1) * (features.cols) / 2*/ + 1
    val (weightVector: List[Double], _) = (0 to features.rows - 1).foldLeft((List.fill(featureCols)(initialWeight * 1.0), 0)) { (soFar: Tuple2[List[Double], Int], rowNum: Int) =>
      val expected = if (labels.get(rowNum, labelColumn) == correctPair) 1 else 0
      val output = applyRowToTrainingModel(
        addTerms(features.row(rowNum).toList),
        expected,
        soFar._1,
        trainingConstant
      )
      if (stopEpochEarly && abs(WeightOperations.elementSubtract(output, soFar._1).sum) < output.size * trainingConstant * 0.5) {
        if(soFar._2 + 1 > floor(features.rows * 0.1)) {
          if (verbose) println(s"Ending training early. Weight vector changed less than ${output.size * trainingConstant * 0.5} for ${floor(features.rows * 0.1)} successive tries on iteration $rowNum")
          return output
        } else {
          (output, soFar._2 + 1)
        }
      } else {
        (output, 0)
      }
    }
    return weightVector
  }

  private[this] def applyRowToTrainingModel(pattern: List[Double], expected: Double, weightVector: List[Double], trainingConstant: Double): List[Double] = {
    val z = if (WeightOperations.elementMultiply(pattern, weightVector).sum > 0) 1 else 0
    val weightChange: List[Double] = if (z != expected) {
      pattern.map(_ * (trainingConstant * (expected - z)))
    } else {
      List.fill(pattern.length)(0)
    }
    //println(s"Pattern: $pattern, Target: $expected, Weight Vector: $weightVector, Net: ${WeightOperations.elementMultiply(pattern, weightVector).sum}, Output: $z, dW: $weightChange")
    return WeightOperations.elementAdd(weightVector, weightChange)
  }

  private[this] def unHashMatchedPairs[A](distinctVals: List[A], iter: Integer): (A, A) = {
    // hash computed from: (n*r) + c - ((r*(r + 1))/2) - (r + 1)
    val amntValues: Integer = distinctVals.size
    val pos1: Integer = amntValues - 2 - floor(sqrt(-8*iter + 4*amntValues*(amntValues-1)-7)/2.0 - 0.5).toInt
    val pos2: Integer = iter + pos1 + 1 - amntValues*(amntValues-1)/2 + (amntValues-pos1)*((amntValues-pos1)-1)/2
    return (distinctVals(pos1), distinctVals(pos2))
  }

  private[this] def addTerms(pattern: List[Double]): List[Double] = {
    // Uncomment the following to use higher terms - aka a quartic machine.
    /*val amntPairings = (pattern.size - 1) * (pattern.size) / 2
    val append = (0 to amntPairings - 1).map { iter =>
      val (pair1, pair2) = unHashMatchedPairs(pattern, iter)
      pair1 * pair2
    }.toList*/
    return pattern /*++ append*/ ++ List(threshold * 1.0)
  }

}
