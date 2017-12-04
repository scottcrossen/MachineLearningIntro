package com.scottcrossen42.machinelearning.clustering


import edu.byu.cs478.toolkit.UnsupervisedLearner
import edu.byu.cs478.toolkit.Matrix
import java.util.Random
import scala.collection.JavaConversions.mapAsScalaMap
import scala.annotation.tailrec

class KMeans(
  random: Random
) extends UnsupervisedLearner {
  import KMeans._

  def apply(rawData: Matrix): Unit = {
    val validCols: Seq[Int] = (if(excludeFirstColumn) 1 else 0) to rawData.cols - (if (excludeLastColumn) 2 else 1)
    val columnContinuities: List[Int] = validCols.map(rawData.valueCount(_)).toList
    val dataset: List[List[Double]] = (0 to rawData.rows - 1).map{ rowNum: Int =>
      validCols.map(rawData.get(rowNum, _)).toList
    }.toList
    val valueNames: List[Map[Integer, String]] = validCols.map { colNum: Int =>
      mapAsScalaMap(rawData.getAttrValues(colNum)).toMap
    }.toList
    val initialClusters: List[Cluster] = seedClusters(dataset, columnContinuities)
    println()
    (0 to maxEpochs - 1).foldLeft((initialClusters, 0, Matrix.MISSING, 0.0, 0.0, numberOfClusters)) { case ((currentClusters: List[Cluster], stoppingCount: Int, bestSSE: Double, bestSilhouette: Double, totalSilhouette: Double, nextCentroidOnError: Int), epochNum: Int) =>
      println(s"***************\nIteration ${epochNum + 1}\n***************")
      currentClusters.zipWithIndex.foreach{ case (cluster: Cluster, index: Int) =>
        println(s"Centroid ${index} = ${cluster.getCentroid(valueNames)}")
      }
      print("Making Assignments")
      val (newClusters: List[Cluster], newNextCentroidOnError: Int) = makeAssignments(currentClusters, dataset, columnContinuities, nextCentroidOnError)
      val individualInstances: String = newClusters.zipWithIndex.map{ case (cluster: Cluster, index: Int) =>
        s"Cluster ${index}: ${cluster.amountAssigned}"
      }.mkString(", ")
      val individualSSEs: String = newClusters.zipWithIndex.map{ case (cluster: Cluster, index: Int) =>
        f"Cluster ${index}: ${cluster.calcSSE}%1.3f"
      }.mkString(", ")
      println(s"Amount of instances: ${individualInstances}")
      println(s"Individual SSE: ${individualSSEs}")
      println(f"Total SSE: ${newClusters.map(_.calcSSE).sum}%1.3f")
      val newClusterCentroids: List[Cluster] = newClusters.map(_.resetClusterCentroid)
      val newSSE: Double = newClusterCentroids.map(_.calcSSE).sum
      //println(f"Updated SSE: ${newSSE}%1.3f")
      val silhouetteMetric: Double = newClusterCentroids.zipWithIndex.map { case (cluster: Cluster, index: Int) =>
        val interalDissimilarity: Double = cluster.calculateAverageDissimilarity
        val externalDissimilarity: Double = newClusterCentroids.zipWithIndex.map { case (compareCluster: Cluster, compareIndex: Int) =>
          if (index == compareIndex) {
            Matrix.MISSING
          } else {
            cluster.calculateAverageDissimilarity(compareCluster)
          }
        }.min
        if (interalDissimilarity < externalDissimilarity) {
          1 - (interalDissimilarity / externalDissimilarity)
        } else if (interalDissimilarity > externalDissimilarity) {
          (externalDissimilarity / interalDissimilarity) - 1
        } else {
          0
        }
      }.sum / (newClusterCentroids.size * 1.0)
      println(f"Silhouette: ${silhouetteMetric}%1.3f")
      println
      val silhouetteDifference: Double = Math.abs(silhouetteMetric - bestSilhouette)
      val (newStoppingCount: Int, newBestSSE: Double, newBestSilhouette: Double) = if (stopOnSSE) {
        val tempBestSilhouette: Double = if (silhouetteMetric > bestSilhouette) silhouetteMetric else bestSilhouette
        if (newSSE >= bestSSE) {
          (stoppingCount + 1, bestSSE, tempBestSilhouette)
        } else {
          (0, newSSE, tempBestSilhouette)
        }
      } else {
        val tempBestSSE: Double = if (newSSE < bestSSE) newSSE else bestSSE
        if (silhouetteMetric > bestSilhouette) {
          (stoppingCount, tempBestSSE, silhouetteMetric)
        } else {
          (stoppingCount + 1, tempBestSSE, bestSilhouette)
        }
      }
      val keepTraining: Boolean = newStoppingCount <= 1 && (stopOnSSE || (newBestSilhouette == 0 || silhouetteDifference > .00001))
      if (keepTraining) {
        (newClusterCentroids, newStoppingCount, newBestSSE, newBestSilhouette, totalSilhouette + silhouetteMetric, newNextCentroidOnError)
      } else {
        if (stopOnSSE) {
          println("SSE has converged")
        } else {
          println("Silhouette has converged")
          println(s"Average Silhouette is ${((totalSilhouette + silhouetteMetric) / (epochNum + 1)).toString}")
        }
        return
      }
    }
    println("Algorithm did not converge within the maximum amount of epochs")
  }
}

object KMeans {
  val numberOfClusters: Int = 3

  val seedWithFirstRows: Boolean = true

  val excludeFirstColumn: Boolean = false

  val excludeLastColumn: Boolean = false

  val maxEpochs: Int = 100

  val stopOnSSE: Boolean = true

  def seedClusters(dataset: List[List[Double]], columnContinuities: List[Int]): List[Cluster] = {
    (if (seedWithFirstRows) {
      dataset
    } else {
      scala.util.Random.shuffle(dataset)
    }).take(numberOfClusters).map{ centroid: List[Double] =>
      Cluster(centroid, columnContinuities)
    }.toList
  }

  @tailrec def makeAssignments(currentClusters: List[Cluster], dataset: List[List[Double]], columnContinuities: List[Int], nextCentroidOnError: Int): (List[Cluster], Int) = {
    val clustersWithAssignments: List[Cluster] = dataset.zipWithIndex.foldLeft(currentClusters.map(_.clearRows)) { case (clusters: List[Cluster], (row: List[Double], rowNum: Int)) =>
      if (rowNum % 10 == 0) print("\n\t") else print(" ")
      val bestClusterIndex: Int = (0 to clusters.size - 1).minBy(clusters(_).calcDistanceFromCentroid(row))
      print(s"${rowNum.toString}=${bestClusterIndex.toString}")
      clusters.zipWithIndex.map { case (cluster: Cluster, index: Int) =>
        if (index == bestClusterIndex) {
          cluster.addRow(row)
        } else {
          cluster
        }
      }
    }
    println
    val emptyClusters: Boolean = clustersWithAssignments.exists(_.isEmpty)
    if (emptyClusters) {
      print("Making New Assignments")
      val (newClusters: List[Cluster], amountEmpty: Int) = clustersWithAssignments.foldLeft(List[Cluster](), 0) { case ((outputClusters: List[Cluster], amountEmptySoFar: Int), cluster: Cluster) =>
        if (cluster.isEmpty) {
          if (seedWithFirstRows) {
            (outputClusters :+ Cluster(dataset((nextCentroidOnError + amountEmptySoFar) % dataset.size), columnContinuities), amountEmptySoFar + 1)
          } else {
            (outputClusters :+ Cluster(scala.util.Random.shuffle(dataset).head, columnContinuities), amountEmptySoFar)
          }
        } else {
          (outputClusters :+ cluster, amountEmptySoFar)
        }
      }
      makeAssignments(newClusters, dataset, columnContinuities, (nextCentroidOnError + amountEmpty) % dataset.size)
    } else {
      (clustersWithAssignments, nextCentroidOnError)
    }
  }
}
