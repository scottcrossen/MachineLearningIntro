package com.scottcrossen42.machinelearning.backpropagation

class DataCollector(
  val functions: List[(NeuralNet => Double)],
  val dataOption: Option[List[List[Double]]] = None
) {
  val data = dataOption.getOrElse(List.fill(functions.size)(List[Double]()))

  def addToValues(neuralNet: NeuralNet): DataCollector = {
    val newDataList: List[List[Double]] = (0 to functions.size - 1).map { iter: Int =>
      val newData = functions(iter).apply(neuralNet)
      data(iter) :+ newData
    }.toList
    new DataCollector(functions, Some(newDataList))
  }

  def print = (0 to data.size - 1).foreach { iter: Int =>
    println(s"Printing collected data set $iter")
    println(data(iter))
  }
}
