package com.scottcrossen42.machinelearning.decisiontree

import edu.byu.cs478.toolkit.SupervisedLearner
import edu.byu.cs478.toolkit.Matrix
import com.scottcrossen42.machinelearning.utility.MatrixOperations
import scala.collection.JavaConverters._

class DecisionTree extends SupervisedLearner {
  import DecisionTree._

  private var labelRoots: List[Node] = List[Node]()

  def train(features: Matrix, labels: Matrix) = {
    labelRoots = if (prune) {
      trainWithPrune(features, labels)
    } else {
      trainWithoutPrune(features, labels)
    }
    if (verbose) {
      println("Tree:")
      labelRoots.foreach(_.print)
    }
    if (verbose) labelRoots.foreach{ root: Node =>
      println(s"Children: ${root.getTotalChildren.toString}")
    }
    if (verbose) labelRoots.foreach{ root: Node =>
      println(s"Height: ${root.height}")
    }
    if (verbose) {
      (0 to labelRoots.size - 1).foreach{ labelColumn: Int =>
        // Accuracy is # correct / total
        val accuracy: Double = (0 to features.rows - 1).filter{ row: Int =>
          labelRoots(labelColumn).predict(features.row(row).toList) == labels.get(row, labelColumn)
        }.size * 1.0 / (features.rows * 1.0)
        println(s"Training Accuracy: $accuracy")
      }
    }
  }

  def trainWithPrune(features: Matrix, labels: Matrix): List[Node] = {
    // Partition features and labels into validation sets
    val (trainingFeatures: Matrix, validationFeatures: Matrix) = MatrixOperations.partitionMatrix(features, 1 - validationAmnt)
    val (trainingLabels: Matrix, validationLabels: Matrix) = MatrixOperations.partitionMatrix(labels, 1 - validationAmnt)
    // First train without prune
    trainWithoutPrune(trainingFeatures, trainingLabels).zipWithIndex.map{ case (node: Node, index: Int) =>
      // Prune Tree and return best
      Tree.pruneTree(node, validationFeatures, validationLabels.col(index).toList)
    }
  }

  def trainWithoutPrune(features: Matrix, labels: Matrix): List[Node] = {
    // One tree per label
    val output: List[Node] = (0 to labels.cols - 1).map{ index: Int =>
      // Convert names to scala-friendly
      val attrValues: Map[Int, String] = mapAsScalaMapConverter(labels.getAttrValues(index)).asScala.toMap.map { case (key: Integer, value: String) =>
        (key.intValue, value)
      }
      Tree.buildTree(features, labels.col(index).toList, attrValues, missingStrategy)
    }.toList
    output
  }

  def predict(features: Array[Double]): Array[Double] = {
    // Defer prediction to root
    return labelRoots.map(_.predict(features.toList)).toArray
  }
}

object DecisionTree {

  private val verbose = false

  private val validationAmnt = 0.25

  private val prune: Boolean = false

  private val missingStrategy: MissingStrategy.Value = MissingStrategy.seperateMissing
}
