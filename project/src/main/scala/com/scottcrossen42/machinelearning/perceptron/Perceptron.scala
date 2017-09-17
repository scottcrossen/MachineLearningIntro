package com.scottcrossen42.machinelearning.perceptron

import edu.byu.cs478.toolkit.SupervisedLearner
import edu.byu.cs478.toolkit.Matrix
import scala.math.{sqrt, floor}

class Perceptron extends SupervisedLearner {

  implicit def arrayToList[A](a: Array[A]) = a.toList

  private[this] val initialWeight = 0

  private[this] val trainingConstant = 1

  // colum of label, pairing of label, weight vector
  private[this] var allWeightVector: Array[List[List[Double]]] = Array(List(List[Double]()))

  private[this] var allDistinctVals: Array[List[String]] = Array(List[String]())

	def train(features: Matrix, labels: Matrix): Unit = {
    (0 to (labels.cols - 1)).foreach { colNum: Int =>
      if (labels.valueCount(colNum) == 0) {
        println("Cannot predict continuous data.")
      } else {
        val distinctVals = labels.uniqVals(colNum).toList
        val amntValues = distinctVals.size
        val amntPairings = (amntValues - 1) * (amntValues) / 2
        allDistinctVals :+ distinctVals
        allWeightVector :+ (0 to amntPairings).map { iter: Int =>
          val (pair1: String, pair2: String) = deHash(distinctVals, iter)
          require(features.rows == labels.rows)
          val goodRows = (0 to features.rows).filter((rowNum) => labels.get(rowNum, colNum) == pair1 || labels.get(rowNum, colNum) == pair2)
          goodRows.foldLeft(List.fill(features.cols)(initialWeight * 1.0)) { (soFar: List[Double], rowNum: Int) =>
            val expected = if (labels.attrValue(colNum, labels.get(rowNum, colNum).toInt) == pair1) 1 else 0
            applyEpoch(
              features.row(rowNum),
              expected,
              soFar,
              trainingConstant)
          }
        }
      }
    }
  }

  def deHash(distinctVals: List[String], iter: Integer): (String, String) = {
    // https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    // (n*r) + c - ((r*(r + 1))/2) - (r + 1)
    val amntValues: Integer = distinctVals.size
    val pos1: Integer = amntValues - 2 - floor(sqrt(-8*iter + 4*amntValues*(amntValues-1)-7)/2 - 1/2).toInt
    val pos2: Integer = iter + pos1 + 1 - amntValues*(amntValues-1)/2 + (amntValues-pos1)*((amntValues-pos1)-1)/2
    return (distinctVals(pos1), distinctVals(pos2))
  }

	def predict(features: Array[Double], labels: Array[Double]): Unit = {
    labels :+ (0 to allDistinctVals.size).map { colNum: Int =>
      val distinctVals = allDistinctVals(colNum)
      val amntValues = distinctVals.size
      val amntPairings = (amntValues - 1) * (amntValues) / 2
      val winners = (0 to amntPairings).map{ iter =>
        val weightVector: List[Double] = allWeightVector(colNum)(iter)
        val (pair1: String, pair2: String) = deHash(distinctVals, iter)
        if (elementMultiply(features.toList, weightVector).sum > 0) pair1 else pair2
      }
      distinctVals.indexOf(distinctVals.maxBy{ x: String =>
        winners.count { winner: String =>
          winner == x
        }
      })
    }.toArray
  }

  private[this] def applyEpoch(pattern: List[Double], expected: Double, weightVector: List[Double], trainingConstant: Double): List[Double] = {
    val z = if (elementMultiply(pattern, weightVector).sum > 0) 1 else 0
    val weightChange: List[Double] = if (z != expected) {
      pattern.map(_ * (trainingConstant * (expected - z)))
    } else {
      List.fill(pattern.length)(0)
    }

    println(s"Pattern: $pattern, Target: $expected, Weight Vector: $weightVector, Net: ${elementMultiply(pattern, weightVector).sum}, Output: $z, [CapitalDelta]W: $weightChange")

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
