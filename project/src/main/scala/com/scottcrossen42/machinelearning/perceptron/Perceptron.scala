package com.scottcrossen42.machinelearning.perceptron

import edu.byu.cs478.toolkit.SupervisedLearner
import edu.byu.cs478.toolkit.Matrix
import scala.math.{sqrt, floor}

class Perceptron extends SupervisedLearner {

  implicit def arrayToList[A](a: Array[A]) = a.toList

  private[this] val initialWeight = 1

  private[this] val trainingConstant = 1

  // column of label, pairing of label, weight vector
  private[this] var allWeightVector: Array[List[List[Double]]] = Array()

  private[this] var allDistinctVals: Array[List[Int]] = Array()

	def train(features: Matrix, labels: Matrix): Unit = (0 to labels.cols - 1).foreach { colNum: Int =>
    val (newWeightVector, newDistinctVals) = (if (labels.valueCount(colNum) == 0) {
      (List(), List())
    } else {
      val distinctVals = labels.uniqVals(colNum).toList.map(labels.attrValue(colNum, _))
      val amntValues = distinctVals.size
      val amntPairings = (amntValues - 1) * (amntValues) / 2
      val weightVector = (0 to amntPairings - 1).map { iter: Int =>
        val (pair1: Int, pair2: Int) = unHash(distinctVals, iter)
        require(features.rows == labels.rows)
        val goodRows = (0 to features.rows - 1).filter((rowNum) => (labels.get(rowNum, colNum) == pair1 || labels.get(rowNum, colNum) == pair2)).toList
        val featureCols = features.cols + (features.cols - 1) * (features.cols) / 2
        goodRows.foldLeft(List.fill(featureCols)(initialWeight * 1.0)) { (soFar: List[Double], rowNum: Int) =>
          val expected = if (labels.get(rowNum, colNum) == pair1) 1 else 0
          applyEpoch(
            addTerms(features.row(rowNum)),
            expected,
            soFar,
            trainingConstant)
        }
      }.toList
      (weightVector, distinctVals)
    })
    allDistinctVals = allDistinctVals :+ newDistinctVals
    allWeightVector = allWeightVector :+ newWeightVector
  }

  def unHash[A](distinctVals: List[A], iter: Integer): (A, A) = {
    // https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    // (n*r) + c - ((r*(r + 1))/2) - (r + 1)
    val amntValues: Integer = distinctVals.size
    val pos1: Integer = amntValues - 2 - floor(sqrt(-8*iter + 4*amntValues*(amntValues-1)-7)/2.0 - 0.5).toInt
    val pos2: Integer = iter + pos1 + 1 - amntValues*(amntValues-1)/2 + (amntValues-pos1)*((amntValues-pos1)-1)/2
    return (distinctVals(pos1), distinctVals(pos2))
  }

	def predict(features: Array[Double]): Array[Double] = (0 to allDistinctVals.size - 1).map { colNum: Int =>
    val distinctVals = allDistinctVals(colNum)
    val amntValues = distinctVals.size
    val amntPairings = (amntValues - 1) * (amntValues) / 2
    val winners = (0 to amntPairings - 1).map{ iter =>
      val weightVector: List[Double] = allWeightVector(colNum)(iter)
      val (pair1: Int, pair2: Int) = unHash(distinctVals, iter)
      if (elementMultiply(addTerms(features.toList), weightVector).sum > 0) pair1 else pair2
    }.toList
    distinctVals.maxBy{ x: Int =>
      winners.count { winner: Int =>
        winner == x
      }
    }.toDouble
  }.toArray

  private[this] def addTerms(pattern: List[Double]): List[Double] = {
    val amntPairings = (pattern.size - 1) * (pattern.size) / 2
    val append = (0 to amntPairings - 1).map { iter =>
      val (pair1, pair2) = unHash(pattern, iter)
      pair1 * pair2
    }.toList
    return pattern ++ append
  }

  private[this] def applyEpoch(pattern: List[Double], expected: Double, weightVector: List[Double], trainingConstant: Double): List[Double] = {
    val z = if (elementMultiply(pattern, weightVector).sum > 0) 1 else 0
    val weightChange: List[Double] = if (z != expected) {
      pattern.map(_ * (trainingConstant * (expected - z)))
    } else {
      List.fill(pattern.length)(0)
    }
    //println(s"Pattern: $pattern, Target: $expected, Weight Vector: $weightVector, Net: ${elementMultiply(pattern, weightVector).sum}, Output: $z, dW: $weightChange")
    return elementAdd(weightVector, weightChange)
  }

  private[this] def elementMultiply(firstList: List[Double], secondList: List[Double]): List[Double] = {
    require(firstList.size == secondList.size)
    firstList.zip(secondList).map {case (val1: Double, val2: Double) => val1 * val2}
  }

  private[this] def elementAdd(firstList: List[Double], secondList: List[Double]): List[Double] = {
    require(firstList.size == secondList.size)
    firstList.zip(secondList).map {case (val1: Double, val2: Double) => val1 + val2}
  }

}
