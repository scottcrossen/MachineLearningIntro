package com.scottcrossen42.machinelearning.clustering

import edu.byu.cs478.toolkit.Matrix;

case class Cluster (
  private val centroid: List[Double],
  private val columnContinuities: List[Int],
  private val rows: List[List[Double]] = List[List[Double]]()
) {
  import Cluster._

  def calcDistanceFromCentroid(row: List[Double]): Double = calcDistance(centroid, row, columnContinuities)

  def addRow(row: List[Double]): Cluster = Cluster(centroid, columnContinuities, rows :+ row)

  def resetClusterCentroid: Cluster = {
    val newCentroid: List[Double] = calcNewCentroid(rows, columnContinuities)
    return Cluster(newCentroid, columnContinuities, rows)
  }

  def clearRows: Cluster = {
    return Cluster(centroid, columnContinuities)
  }

  def calculateAverageDissimilarity: Double = calculateAverageDissimilarityOfRows(rows, rows, columnContinuities)

  def calculateAverageDissimilarity(other: Cluster): Double = calculateAverageDissimilarityOfRows(rows, other.rows, columnContinuities)

  def getCentroid(valueNames: List[Map[Integer, String]]): String = {
    val names: List[String] = centroid.zipWithIndex.map { case (value: Double, index: Int) =>
      if (value == Matrix.MISSING) {
        "?"
      } else if (columnContinuities(index) != 0) {
        s"${valueNames(index).get(value.toInt).getOrElse("Error")}"
      } else {
        f"${value}%1.3f"
      }
    }
    names.mkString(", ")
  }

  def calcSSE: Double = rows.map(calcDistanceFromCentroid(_)).sum

  def isEmpty: Boolean = rows.size == 0

  def amountAssigned: Int = rows.size
}

object Cluster {
  def calcDistance(row1: List[Double], row2: List[Double], columnContinuities: List[Int], squared: Boolean = true): Double = {
    (row1, row2, columnContinuities).zipped.map{ case (firstValue: Double, secondValue: Double, continuity: Int) =>
      if (firstValue == Matrix.MISSING || secondValue == Matrix.MISSING) {
        1.0
      } else if (continuity == 0) {
        if (squared) {
          Math.pow(firstValue - secondValue, 2)
        } else {
          Math.abs(firstValue - secondValue)
        }
      } else if (firstValue == secondValue){
        0.0
      } else {
        1.0
      }
    }.sum
  }
  def calcNewCentroid(rows: List[List[Double]], columnContinuities: List[Int]): List[Double] = {
    (rows.transpose, columnContinuities, (0 to columnContinuities.size - 1).toList).zipped.map { case (column: List[Double], continuity: Int, colNum: Int) =>
      val filteredColumn: List[Double] = column.filter(_ != Matrix.MISSING)
      if (filteredColumn.size == 0) {
        Matrix.MISSING
      } else if (continuity == 0) {
        filteredColumn.sum / filteredColumn.size * 1.0
      } else {
        /*if (filteredColumn.size % 2 == 0) {
          val sorted: List[Double] = filteredColumn.sortWith(_ < _)
          (sorted(filteredColumn.size/2) + sorted(filteredColumn.size/2-1)) / 2
        } else {
          filteredColumn.sortWith(_ < _)(filteredColumn.size/2)
        }*/
        filteredColumn.groupBy(identity).toList.sortBy(_._1).maxBy(_._2.size)._1
      }
    }
  }
  def calculateAverageDissimilarityOfRows(firstRows: List[List[Double]], secondRows: List[List[Double]], columnContinuities: List[Int]): Double = {
    val distances: List[Double] = firstRows.zipWithIndex.map { case (firstRow: List[Double], index: Int) =>
      secondRows.zipWithIndex.filter(_._2 != index).map { case (secondRow: List[Double], _) =>
        calcDistance(firstRow, secondRow, columnContinuities)
      }
    }.flatten
    distances.sum / (distances.size * 1.0)
  }
}
