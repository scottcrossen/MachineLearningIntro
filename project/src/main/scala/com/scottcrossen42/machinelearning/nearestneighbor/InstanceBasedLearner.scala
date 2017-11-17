package com.scottcrossen42.machinelearning.nearestneighbor

import edu.byu.cs478.toolkit.SupervisedLearner
import edu.byu.cs478.toolkit.Matrix
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.{Duration, MILLISECONDS}
import java.util.concurrent.Executor

class InstanceBasedLearner extends SupervisedLearner {
  import InstanceBasedLearner._
  import ExecutionContext.Implicits.global

  private var trainingFeatures: Matrix = new Matrix()
  private var trainingLabels: Matrix = new Matrix()

  def train(features: Matrix, labels: Matrix) = {
    trainingFeatures = features
    trainingLabels = labels
  }

  def predict(newFeature: Array[Double]): Array[Double] = {
    val featuresImmutable: List[List[Double]] = (0 to trainingFeatures.rows - 1).map{ currentRow: Int =>
      trainingFeatures.row(currentRow).toList
    }.toList
    val labelsImmutable: List[List[Double]] = (0 to trainingLabels.rows - 1).map{ currentRow: Int =>
      trainingLabels.row(currentRow).toList
    }.toList
    val featureColsImmutable: List[List[Double]] = (0 to trainingFeatures.cols - 1).map{ currentCol: Int =>
      trainingFeatures.col(currentCol).toList
    }.toList
    val labelColsImmutable: List[List[Double]] = (0 to trainingLabels.cols - 1).map{ currentCol: Int =>
      trainingLabels.col(currentCol).toList
    }.toList
    val featureValueCount: List[Int] = (0 to trainingFeatures.cols - 1).map{ currentCol: Int =>
      trainingFeatures.valueCount(currentCol)
    }.toList
    val labelValueCount: List[Int] = (0 to trainingLabels.cols - 1).map{ currentCol: Int =>
      trainingLabels.valueCount(currentCol)
    }.toList
    return Await.result(predictWithMatrices(
      featuresImmutable,
      labelsImmutable,
      featureColsImmutable,
      labelColsImmutable,
      featureValueCount,
      labelValueCount,
      newFeature.toList
    ), Duration.Inf).toArray
  }

  override def predictOnAllValues(newFeatures: Matrix): Array[Double] = {
    val featuresImmutable: List[List[Double]] = (0 to trainingFeatures.rows - 1).map{ currentRow: Int =>
      trainingFeatures.row(currentRow).toList
    }.toList
    val labelsImmutable: List[List[Double]] = (0 to trainingLabels.rows - 1).map{ currentRow: Int =>
      trainingLabels.row(currentRow).toList
    }.toList
    val featureColsImmutable: List[List[Double]] = (0 to trainingFeatures.cols - 1).map{ currentCol: Int =>
      trainingFeatures.col(currentCol).toList
    }.toList
    val labelColsImmutable: List[List[Double]] = (0 to trainingLabels.cols - 1).map{ currentCol: Int =>
      trainingLabels.col(currentCol).toList
    }.toList
    val featureValueCount: List[Int] = (0 to trainingFeatures.cols - 1).map{ currentCol: Int =>
      trainingFeatures.valueCount(currentCol)
    }.toList
    val labelValueCount: List[Int] = (0 to trainingLabels.cols - 1).map{ currentCol: Int =>
      trainingLabels.valueCount(currentCol)
    }.toList
    val newFeaturesImmutable: List[List[Double]] = (0 to newFeatures.rows - 1).map{ currentRow: Int =>
      newFeatures.row(currentRow).toList
    }.toList
    return Await.result(Future.sequence(newFeaturesImmutable.zipWithIndex.map { case (feature: List[Double], index: Int) =>
      //if (index % 100 == 0) println(s"Testing feature #${index}")
      predictWithMatrices(
        featuresImmutable,
        labelsImmutable,
        featureColsImmutable,
        labelColsImmutable,
        featureValueCount,
        labelValueCount,
        feature
      ).map(_.head)
    }), Duration.Inf).toArray
  }

  // For integration with java.
  def setNumberOfNeighbors(k: Int) = {
    numberOfNeighbors = k
  }

  // Note: expansion project was done in a separate script to pre-process the data. I'm working on integratng it into the training method.
}

object InstanceBasedLearner {
  val useDistanceWeighting: Boolean = false

  var numberOfNeighbors: Int = 3

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

  def predictWithMatrices(
    featuresImmutable: List[List[Double]],
    labelsImmutable: List[List[Double]],
    featureColsImmutable: List[List[Double]],
    labelColsImmutable: List[List[Double]],
    featureValueCount: List[Int],
    labelValueCount: List[Int],
    newFeature: List[Double]
  )(implicit context: ExecutionContext
  ): Future[List[Double]] = {
    Future.sequence((0 to labelColsImmutable.size - 1).map { labelColumn: Int =>
      val currentLabelColumn: List[Double] = labelColsImmutable(labelColumn)
      Future.sequence((0 to featuresImmutable.size - 1).map { currentRow: Int =>
        Future {
          val result: Tuple2[Double, Double] = ((0 to newFeature.size - 1).map { featureColumn: Int =>
            val currentLabelColumn: List[Double] = featureColsImmutable(featureColumn)
            val firstValue: Double = featuresImmutable(currentRow)(featureColumn)
            val secondValue: Double = newFeature(featureColumn)
            if (firstValue.equals(Matrix.MISSING) || secondValue.equals(Matrix.MISSING)) {
              1.0
            } else if (featureValueCount(featureColumn) == 0 || labelValueCount(labelColumn) == 0) {
              Math.abs(firstValue - secondValue)
            } else {
              categoricalDistance(currentLabelColumn, currentLabelColumn, firstValue, secondValue);
            }
          }.sum, labelsImmutable(currentRow)(labelColumn))
          result
        }
      }.toList).map { allNeighbors: List[(Double, Double)] =>
        val closestNeighbors: List[(Double, Double)] = allNeighbors.sortBy(_._1).take(numberOfNeighbors)
        if (labelValueCount(labelColumn) == 0 && !useDistanceWeighting) {
          val votedLabels: List[Double] = closestNeighbors.map(_._2)
          votedLabels.sum / votedLabels.size.toDouble
        } else if (labelValueCount(labelColumn) == 0) {
          val (numerator: Double, denominator: Double) = closestNeighbors.foldLeft((0.0,0.0)) { case ((numeratorSum: Double, denominatorSum: Double), (distance: Double, label: Double)) =>
            (numeratorSum + label / Math.pow(distance, 2), denominatorSum + 1 / Math.pow(distance, 2))
          }
          numerator / denominator
        } else if (!useDistanceWeighting) {
          closestNeighbors.groupBy(_._2).toList.sortWith(_._2.size > _._2.size).headOption.map(_._1).getOrElse(Matrix.MISSING)
        } else {
          val labelWeights: List[(Double, Double)] = closestNeighbors.groupBy(_._2).toList.map{ case (label: Double, originals: List[(Double, Double)]) =>
            val weight: Double = originals.map { case (distance: Double, _) =>
              1 / Math.pow(distance, 2)
            }.sum
            (label, weight)
          }
          val denominator: Double = labelWeights.map(_._2).sum
          labelWeights.map{ case (label: Double, weight: Double) =>
            (label, weight / denominator)
          }.sortWith(_._2 > _._2).headOption.map(_._1).getOrElse(Matrix.MISSING)
        }
      }
    }.toList)
  }
}
