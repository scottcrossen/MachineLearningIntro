package com.scottcrossen42.machinelearning.nearestneighbor

import edu.byu.cs478.toolkit.SupervisedLearner
import edu.byu.cs478.toolkit.Matrix

class InstanceBasedLearner extends SupervisedLearner {
  import InstanceBasedLearner._

  private var trainingFeatures: Matrix = new Matrix()
  private var trainingLabels: Matrix = new Matrix()

  def train(features: Matrix, labels: Matrix) = {
    trainingFeatures = features
    trainingLabels = labels
  }

  def predict(newFeature: Array[Double]): Array[Double] = {
    return predictWithMatrices(trainingFeatures, trainingLabels, newFeature.toList).toArray
  }
}

object InstanceBasedLearner {
  val useDistanceWeighting: Boolean = false

  val numberOfNeighbors: Int = 3

  def categoricalDistance(features: List[Double], labels: List[Double], firstValue: Double, secondValue: Double): Double = {
    val firstOccurrances: List[Tuple2[Double, Double]] = labels.zip(features).filter{
      case (_, feature: Double) => feature == firstValue
    }
    val secondOccurrances: List[Tuple2[Double, Double]] = labels.zip(features).filter{
      case (_, feature: Double) => feature == secondValue
    }
    val numberOfFirst: Double = firstOccurrances.size.toDouble
    val numberOfSecond: Double = secondOccurrances.size.toDouble
    labels.distinct.foldLeft(0.0){ case (totalValue: Double, currentLabel: Double) =>
      val numberOfFirstWithLabel: Double = firstOccurrances.count {
        case (label: Double, _) => currentLabel == label
      }.toDouble
      val numberOfSecondWithLabel: Double = secondOccurrances.count {
        case (label: Double, _) => currentLabel == label
      }.toDouble
      totalValue + Math.pow(numberOfFirstWithLabel / numberOfFirst - numberOfSecondWithLabel / numberOfSecond, 2)
    }
  }

  def predictWithMatrices(features: Matrix, labels: Matrix, newFeature: List[Double]): List[Double] = {
    (0 to labels.cols - 1).map { labelColumn: Int =>
      val currentLabelColumn: List[Double] = labels.col(labelColumn).toList
      val closestNeighbors: List[(Double, Double)] = (0 to features.rows - 1).map { currentRow: Int =>
        ((0 to newFeature.size - 1).map { featureColumn: Int =>
          val currentLabelColumn: List[Double] = features.col(featureColumn).toList
          val firstValue: Double = features.get(currentRow, featureColumn)
          val secondValue: Double = newFeature(featureColumn)
          if (firstValue.equals(Matrix.MISSING) || secondValue.equals(Matrix.MISSING)) {
            1.0
          } else if (features.valueCount(featureColumn) == 0 || labels.valueCount(labelColumn) == 0) {
            Math.abs(firstValue - secondValue)
          } else {
            categoricalDistance(currentLabelColumn, currentLabelColumn, firstValue, secondValue);
          }
        }.sum, labels.get(currentRow, labelColumn))
      }.toList.sortBy(_._1).take(numberOfNeighbors)
      if (labels.valueCount(labelColumn) == 0 && !useDistanceWeighting) {
        val votedLabels: List[Double] = closestNeighbors.map(_._1)
        votedLabels.sum / votedLabels.size.toDouble
      } else if (labels.valueCount(labelColumn) == 0) {
        val (numerator: Double, denominator: Double) = closestNeighbors.foldLeft((0.0,0.0)) { case ((numeratorSum: Double, denominatorSum: Double), (label: Double, distance: Double)) =>
          (numeratorSum + label / Math.pow(distance, 2), denominatorSum + 1 / Math.pow(distance, 2))
        }
        numerator / denominator
      } else if (!useDistanceWeighting) {
        closestNeighbors.groupBy(_._1).toList.sortWith(_._2.size > _._2.size).headOption.map(_._1).getOrElse(Matrix.MISSING)
      } else {
        val labelWeights: List[(Double, Double)] = closestNeighbors.groupBy(_._1).toList.map{ case (label: Double, originals: List[(Double, Double)]) =>
          val weight: Double = originals.map { case (_, distance: Double) =>
            1 / Math.pow(distance, 2)
          }.sum
          (label, weight)
        }
        val denominator: Double = labelWeights.map(_._2).sum
        labelWeights.map{ case (label: Double, weight: Double) =>
          (label, weight / denominator)
        }.sortWith(_._2 > _._2).headOption.map(_._1).getOrElse(Matrix.MISSING)
      }
    }.toList
  }
}
