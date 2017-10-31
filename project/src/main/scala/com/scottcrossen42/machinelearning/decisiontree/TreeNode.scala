package com.scottcrossen42.machinelearning.decisiontree

import edu.byu.cs478.toolkit.Matrix
import scala.collection.JavaConverters._
import java.util.Random


object MissingStrategy extends Enumeration {
  val seperateMissing = Value(0, "seperate")
  val combineMissing = Value(1, "combine")
  val ignoreMissing = Value(2, "ignore")
}

trait Node {
  def height: Int
  def predict(selection: List[Double]): Double
  def toString(depth: Int): String
  def print: Unit = println(toString(0))
  def getLabelProbabilities: List[(Double, Double, String)]
  def getMostProbableLabel: (Double, String)
  def getTotalChildren: Int
}

case class ErrorNode(
  val message: String
) extends Node {
  def height: Int = 0
  def predict(selection: List[Double]): Double = -1.0
  def toString(depth: Int): String = s"${"\t" * depth}Error: $message\n"
  def getLabelProbabilities: List[(Double, Double, String)] = List((0.0, 0.0, message))
  def getMostProbableLabel: (Double, String) = (0.0, message)
  def getTotalChildren: Int = 0
}

case class LabelNode(
  val name: String,
  val label: Double
) extends Node {
  def height: Int = 1
  def predict(selection: List[Double]): Double = label
  def toString(depth: Int): String = s"Label: $name\n"
  def getLabelProbabilities: List[(Double, Double, String)] = {
    List((label, 1.0, name))
  }
  def getMostProbableLabel: (Double, String) = (label, name)
  def getTotalChildren: Int = 1
}

case class FeatureNode(
  val name: String,
  val featureIndex: Int,
  // Feature value to tuple of probability and Node
  val children: Map[Double, (Double, String, Node)],
  val missingStrategy: MissingStrategy.Value
) extends Node {

  def getTotalChildren: Int = children.values.map(_._3.getTotalChildren).sum + 1

  def height: Int = ((children.values.map { case (_, _, node: Node) =>
    node.height + 1
  }).toList :+ 1).max

  def predict(selection: List[Double]): Double = {
    val newSelection: List[Double] = selection.zipWithIndex.filter{ case (_, index: Int) =>
      index != featureIndex
    }.map(_._1)
    if (children.contains(selection(featureIndex))) {
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

  def getLabelProbabilities: List[(Double, Double, String)] = {
    children.values.map{ case (parentProbability: Double, _, child: Node) =>
      child.getLabelProbabilities.map{ case (label: Double, childProbability: Double, name: String) =>
        (label, parentProbability * childProbability, name)
      }
    }.toList.flatten.groupBy(_._1).toList.map{ case (label: Double, tuples: List[(Double, Double, String)]) =>
      tuples.headOption.map{ case (firstLabel: Double, _, name: String) =>
        (label, tuples.map{ case (_, probability: Double, _) =>
          probability
        }.sum, name)
      }.getOrElse((0.0, 0.0, "Error: Head doesn't exist."))
    }
  }

  def getMostProbableLabel: (Double, String) = {
    val (label: Double, probability: Double, name: String) = getLabelProbabilities.maxBy(_._2)
    (label, name)
  }
}
