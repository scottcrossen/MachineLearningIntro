package com.scottcrossen42.machinelearning.utility

import edu.byu.cs478.toolkit.Matrix

object MatrixOperations {

  def partitionMatrix(matrix: Matrix, split: Double): Tuple2[Matrix, Matrix] = {
    val takeAmnt: Int = Math.floor(matrix.rows * split).toInt
    val matrix1: Matrix = new Matrix(matrix, 0, 0, takeAmnt, matrix.cols)
    val matrix2: Matrix = new Matrix(matrix, takeAmnt - 1, 0, matrix.rows - takeAmnt, matrix.cols)
    (matrix1, matrix2)
  }

}
