package pl.roqad.flinkknn

import org.apache.flink.api.scala._
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.ml.math.{DenseVector, Matrix, Vector => FlinkVector}
import org.apache.flink.ml.nn.KNN
import org.apache.flink.ml.metrics.distances.SquaredEuclideanDistanceMetric


case class Wektor(val data: DenseVector, val label: String) extends FlinkVector {
  override def size: Int = data.size
  override def update(index: Int, value: Double): Unit = data.update(index, value)
  override def copy: FlinkVector = data.copy
  override def dot(other: FlinkVector): Double = data.dot(other)
  override def outer(other: FlinkVector): Matrix = data.outer(other)
  override def magnitude: Double = data.magnitude
  override def apply(index: Int): Double = data.apply(index)
}
case class Result(val dev: String, val nns: Seq[String])
case class WektorResult(val dev: Wektor, val nns: Array[Wektor])
case class DetDeviceIPPair(val dev: String, val ip: String)
case class DetDevicesPair(val dev1: String, dev2: String)
case class KNNDevicesPair(val dev1: String, dev2: String)

object KNNEvaluator {
  def main(args: Array[String]) {
    import org.apache.flink.api.scala.extensions._

    val params: ParameterTool = ParameterTool.fromArgs(args)
    val env = ExecutionEnvironment.getExecutionEnvironment

    val input: String = params.get("input")
    val det: String = params.get("det")
    val k: Int = params.getInt("k")
    val blocks: Int = params.getInt("blocks")
    val approximate: Boolean = params.getBoolean("approximate")

    val data: DataSet[Wektor] = env
      .readTextFile(input)
      .map { str =>
        val splitted = str.split(",")
        Wektor(DenseVector(splitted(1)
          .split(" ")
          .map(_.toDouble)), splitted(0)
        )
      }

    val detData: DataSet[DetDeviceIPPair] = env
      .readCsvFile(det, fieldDelimiter=",")

    val knn: KNN = KNN()
      .setK(k)
      .setBlocks(blocks)
      .setDistanceMetric(SquaredEuclideanDistanceMetric())
      .setUseQuadTree(approximate) // true - use quad tree, false - brute force

    knn.fit(data)
    val preds: DataSet[WektorResult] = knn
      .predict(data)
      .mapWith { case (dev, nns) =>
        WektorResult(dev.asInstanceOf[Wektor], nns.map(_.asInstanceOf[Wektor]))
      }

    // Why this one requires the WektorResult in a partial function?
    val results: DataSet[Result] = preds.mapWith { case WektorResult(dev, nns) =>
      Result(dev.label, nns.map(nn => nn.label))
    }

    // Why this one doesn't let partial functions at all?
    val knnConns: DataSet[KNNDevicesPair] = results.flatMap { result =>
      result.nns.map(nn => if (nn < result.dev) KNNDevicesPair(nn, result.dev) else KNNDevicesPair(result.dev, nn))
    }.distinct

    val detConns: DataSet[DetDevicesPair] = detData
      .groupBy(_.ip)
      .reduceGroup { pairs =>
        val devsSeq: Seq[String] = pairs.map(_.dev).toSeq
        for {
          did1 <- devsSeq
          did2 <- devsSeq if (did2 > did1)
        } yield DetDevicesPair(did1, did2)
      }.flatMap(x => x).distinct

    val detConnsToJoin: DataSet[(DetDevicesPair, Int)] = detConns.map((_, 1))
    val knnConnsToJoin: DataSet[(KNNDevicesPair, Int)] = knnConns.map((_, 1))
    val detIntersectKNN: Long = detConnsToJoin.join(knnConnsToJoin).where(0).equalTo(0).count

    println(s"det connections: ${detConns.count}")
    println(s"knn connections: ${knnConns.count}")
    println(s"intersection connections: ${detIntersectKNN}")
    println(s"recall: ${detIntersectKNN.toDouble / detConns.count}")
  }
}