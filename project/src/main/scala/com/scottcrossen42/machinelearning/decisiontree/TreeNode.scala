package com.scottcrossen42.machinelearning.decisiontree

import edu.byu.cs478.toolkit.Matrix
import scala.collection.JavaConverters._
import java.util.Random

trait Node {
  def height: Int
  def predict(selection: List[Double]): Double
  def toString(depth: Int): String
  def print: Unit = println(toString(0))
}

class LabelNode(
  private val name: String,
  private val label: Double
) extends Node {
  def height: Int = 1
  def predict(selection: List[Double]): Double = label
  def toString(depth: Int): String = s"Label: $name\n"
}

class FeatureNode(
  private val name: String,
  private val featureIndex: Int,
  // Feature value to tuple of probability and Node
  private val children: Map[Double, (Double, String, Node)]
) extends Node {
  def height: Int = children.foldLeft(1) { case (max: Int, (_, (_, _, node: Node))) =>
    if (max >= node.height) {
      max
    } else {
      node.height
    }
  }
  def predict(selection: List[Double]): Double = {
    val newSelection: List[Double] = selection.zipWithIndex.filter{ case (_, index: Int) =>
      index != featureIndex
    }.map(_._1)
    if (selection(featureIndex) != null && children.contains(selection(featureIndex))) {
      children(selection(featureIndex))._3.predict(newSelection)
    } else {
      children.values.map{ case (probability: Double, _, child: Node) =>
        (child.predict(newSelection), probability)
      }.groupBy(_._1).maxBy(_._2.map(_._2).sum)._1
    }
  }
  def toString(depth: Int): String = s"Feature: $name\n" + children.map{ case (_, (_, name: String, child: Node)) =>
    "\t" * (depth + 1) + s"$name - " + child.toString(depth + 1)
  }.mkString("")
}

class ErrorNode(
  val message: String
) extends Node {
  def height: Int = 0
  def predict(selection: List[Double]): Double = -1.0
  def toString(depth: Int): String = s"${"\t" * depth}Error: $message\n"
}

object Tree {
  def buildTree(features: Matrix, labels: List[Double], labelAttrs: Map[Int, String]): Node = {
    if (labels.size == 0) {
      new ErrorNode("Incorrect Parameters: No labels.")
    } else if (features.rows != labels.size) {
      new ErrorNode("Incorrect Parameters: Mismatched feature/label sizes.")
    } else if (labels.groupBy(item => item).size == 1) {
      new LabelNode(labelAttrs(labels(0).toInt),labels(0))
    } else if (features.cols == 0) {
      val bestLabel: Double = labels.groupBy(item => item).maxBy(_._2.size)._1
      new LabelNode(labelAttrs(bestLabel.toInt), bestLabel)
    } else {
      val bestSplit: Int = (0 to features.cols - 1).map{ col: Int =>
        val featureColumn: List[Double] = features.col(col).toList
        featureColumn.groupBy(item => item).mapValues(_.size).map{ case (feature: Double, value: Int) =>
          val labelsPerFeature: List[Double] = labels.zipWithIndex.filter{ case (label: Double, index: Int) =>
            featureColumn(index) == feature
          }.map(_._1)
          (value * 1.0) / (featureColumn.size * 1.0) * labelsPerFeature.groupBy(item => item).mapValues(_.size).map{ case (key: Double, value: Int) =>
            calculateEntropy((value * 1.0) / (labelsPerFeature.size * 1.0))
          }.sum
        }.sum
      }.zipWithIndex.minBy(_._1)._2
      val featureColumn: List[Double] = features.col(bestSplit).toList
      val featureVals: List[Double] = featureColumn.groupBy(item => item).map(_._1).toList
      val keepCols: Set[Int] = (0 to features.cols - 1).filter(_ != bestSplit).toSet
      val children: Map[Double, (Double, String, Node)] = featureVals.map{ feature: Double =>
        val keepRows: Set[Int] = (0 to features.rows - 1).filter(features.row(_)(bestSplit) == feature).toSet
        val newFeatures: Matrix = new Matrix(features, keepRows.asJava, keepCols.asJava)
        val newLabels: List[Double] = labels.zipWithIndex.filter{ case (label: Double, index: Int) =>
          keepRows.contains(index)
        }.map(_._1)
        val childName: String = if (features.attrValue(bestSplit, feature.toInt) != null) {
          features.attrValue(bestSplit, feature.toInt)
        } else {
          feature.toString
        }
        (feature, (keepRows.size * 1.0 / (features.rows * 1.0), childName, buildTree(newFeatures, newLabels, labelAttrs)))
      }.toMap
      new FeatureNode(features.attrName(bestSplit), bestSplit, children)
    }
  }

  def calculateEntropy(probability: Double): Double = -1.0 * probability * Math.log(probability) / Math.log(2)
}
