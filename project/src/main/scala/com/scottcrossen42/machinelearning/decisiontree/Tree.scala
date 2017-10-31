package com.scottcrossen42.machinelearning.decisiontree

import edu.byu.cs478.toolkit.Matrix
import scala.collection.JavaConverters._
import java.util.Random

object Tree {
  def buildTree(features: Matrix, labels: List[Double], labelAttrs: Map[Int, String], missingStrategy: MissingStrategy.Value): Node = {
    if (labels.size == 0) {
      ErrorNode("Incorrect Parameters: No labels.")
    } else if (features.rows != labels.size) {
      ErrorNode("Incorrect Parameters: Mismatched feature/label sizes.")
    } else if (labels.groupBy(item => item).size == 1) {
      LabelNode(labelAttrs(labels(0).toInt),labels(0))
    } else if (features.cols == 0) {
      val bestLabel: Double = labels.groupBy(item => item).maxBy(_._2.size)._1
      LabelNode(labelAttrs(bestLabel.toInt), bestLabel)
    } else {
      val bestSplit: Int = (0 to features.cols - 1).map{ col: Int =>
        val featureColumn: List[Double] = features.col(col).toList
        val tempFeatureVals: Map[Double, Int] = featureColumn.groupBy(item => item).mapValues(_.size)
        val featureVals: Map[Double, Int] = if (missingStrategy == MissingStrategy.seperateMissing) {
          tempFeatureVals
        } else {
          if (tempFeatureVals.filter(_._1 != Matrix.MISSING).size == 0) {
            tempFeatureVals
          } else {
            tempFeatureVals.filter(_._1 != Matrix.MISSING)
          }
        }
        featureVals.map{ case (feature: Double, value: Int) =>
          val labelsPerFeature: List[Double] = labels.zipWithIndex.filter{ case (label: Double, index: Int) =>
            featureColumn(index) == feature || ((missingStrategy == MissingStrategy.combineMissing || tempFeatureVals.filter(_._1 != Matrix.MISSING).size == 0) && featureColumn(index) == Matrix.MISSING)
          }.map(_._1)
          (value * 1.0) / (featureColumn.size * 1.0) * labelsPerFeature.groupBy(item => item).mapValues(_.size).map{ case (key: Double, value: Int) =>
            calculateEntropy((value * 1.0) / (labelsPerFeature.size * 1.0))
          }.sum
        }.sum
      }.zipWithIndex.minBy(_._1)._2
      val featureColumn: List[Double] = features.col(bestSplit).toList
      val tempFeatureVals: List[Double] = featureColumn.groupBy(item => item).map(_._1).toList
      val featureVals: List[Double] = if (missingStrategy == MissingStrategy.seperateMissing) {
        tempFeatureVals
      } else {
        if (tempFeatureVals.filter(_ != Matrix.MISSING).size == 0) {
          tempFeatureVals
        } else {
          tempFeatureVals.filter(_ != Matrix.MISSING)
        }
      }
      val keepCols: Set[Int] = (0 to features.cols - 1).filter(_ != bestSplit).toSet
      val children: Map[Double, (Double, String, Node)] = featureVals.map{ feature: Double =>
        val keepRows: Set[Int] = (0 to features.rows - 1).filter{ index: Int =>
          if (missingStrategy == MissingStrategy.seperateMissing) {
            features.row(index)(bestSplit) == feature
          } else {
            features.row(index)(bestSplit) == feature || features.row(index)(bestSplit) == Matrix.MISSING
          }
        }.toSet
        val newFeatures: Matrix = new Matrix(features, keepRows.asJava, keepCols.asJava)
        val newLabels: List[Double] = labels.zipWithIndex.filter{ case (label: Double, index: Int) =>
          keepRows.contains(index)
        }.map(_._1)
        val childName: String = if (features.attrValue(bestSplit, feature.toInt) != null) {
          features.attrValue(bestSplit, feature.toInt)
        } else {
          feature.toString
        }
        (feature, (keepRows.size * 1.0 / (features.rows * 1.0), childName, buildTree(newFeatures, newLabels, labelAttrs, missingStrategy)))
      }.toMap
      FeatureNode(features.attrName(bestSplit), bestSplit, children, missingStrategy)
    }
  }

  private val maxPruningAttempts = 1000

  def pruneTree(root: Node, validationFeatures: Matrix, validationLabels: List[Double]): Node = {
    (0 to maxPruningAttempts - 1).foldLeft((0.0, 0, root)) { case ((bestScore: Double, noChangeIters: Int, bestNode: Node), _) =>
      val (bestCurrentNode: Node, bestCurrentValue: Double) = (getAllPossiblePrunes(root) :+ root).map{ node: Node =>
        (node, (0 to validationFeatures.rows - 1).filter{ index: Int =>
          node.predict(validationFeatures.row(index).toList) == validationLabels(index)
        }.size * 1.0 / (validationFeatures.rows * 1.0))
      }.maxBy(_._2)
      if (bestCurrentValue > bestScore) {
        (bestCurrentValue, 0, bestCurrentNode)
      } else if (noChangeIters > 5) {
        return bestNode
      } else {
        (bestScore, noChangeIters + 1, bestNode)
      }
    }._3
  }

  private[this] def getAllPossiblePrunes(root: Node): List[Node] = {
    root match {
      case current: FeatureNode => {
        val (mostProbableLabel: Double, mostProbableName: String) = current.getMostProbableLabel
        current.children.map{ case (currentValue: Double, (childProbability: Double, childName: String, currentChild: Node)) =>
          val partialChildren: Map[Double, (Double, String, Node)] = current.children.filterKeys(_ != currentValue)
          getAllPossiblePrunes(currentChild).map{ child: Node =>
            FeatureNode(current.name, current.featureIndex, partialChildren + (currentValue -> (childProbability, childName, child)), current.missingStrategy)
          }
        }.toList.flatten :+ LabelNode(mostProbableName, mostProbableLabel)
      }
      case current: LabelNode => List[Node](current)
      case current: ErrorNode => List[Node](current)
    }
  }

  def calculateEntropy(probability: Double): Double = -1.0 * probability * Math.log(probability) / Math.log(2)
}
