package com.scottcrossen42.machinelearning.decisiontree

import edu.byu.cs478.toolkit.SupervisedLearner
import edu.byu.cs478.toolkit.Matrix
import com.scottcrossen42.machinelearning.decisiontree.Tree
import scala.collection.JavaConverters._

class DecisionTree extends SupervisedLearner {
  private var labelRoots: List[Node] = List[Node]()

  def train(features: Matrix, labels: Matrix) = {
    labelRoots = (0 to labels.cols - 1).map{ index: Int =>
      val attrValues: Map[Int, String] = mapAsScalaMapConverter(labels.getAttrValues(index)).asScala.toMap.map { case (key: Integer, value: String) =>
        (key.intValue, value)
      }
      Tree.buildTree(features, labels.col(index).toList, attrValues)
    }.toList
    labelRoots.foreach(_.print)
  }

  def predict(features: Array[Double]): Array[Double] = {
    return labelRoots.map(_.predict(features.toList)).toArray
  }
}
