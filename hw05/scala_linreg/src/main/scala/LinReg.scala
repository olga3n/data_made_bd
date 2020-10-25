import scala.collection.mutable.ListBuffer
import breeze.linalg._
import breeze.numerics._

class LinReg {

  private var coeffs: DenseVector[Double] = _

  def fit(
      x: DenseMatrix[Double],
      y: DenseVector[Double]): Unit = {

    val ones_vec = DenseVector.ones[Double](x.rows)
    val x_upd = DenseMatrix.horzcat(DenseMatrix(ones_vec).t, x)

    coeffs = pinv(x_upd.t * x_upd) * x_upd.t * y
  }

  def predict(x: DenseMatrix[Double]): DenseVector[Double] = {
    val ones_vec = DenseVector.ones[Double](x.rows)
    val x_upd = DenseMatrix.horzcat(DenseMatrix(ones_vec).t, x)

    x_upd * coeffs
  }

  def rmse(
      v1: DenseVector[Double],
      v2: DenseVector[Double]): Double = {

    val res: Double = v1.valuesIterator.zip(v2.valuesIterator)
      .map(item => item._1 - item._2).map(x => x * x).reduce(_ + _)

    sqrt(res / v1.size)
  }

  def cv_scores(
      x: DenseMatrix[Double],
      y: DenseVector[Double], count: Int): Seq[Double] = {

    val chunk_size = y.size / (count + 1)
    var scores = new ListBuffer[Double]

    for (i <- 0 until count) {
      val curr_x_train = x(0 until (i + 1) * chunk_size, ::)
      val curr_y_train = y(0 until (i + 1) * chunk_size)

      fit(curr_x_train, curr_y_train)

      val curr_x_test = x((i + 1) * chunk_size until (i + 2) * chunk_size, ::)
      val curr_y_test = y((i + 1) * chunk_size until (i + 2) * chunk_size)

      val curr_y_pred = predict(curr_x_test)

      val score = rmse(curr_y_test, curr_y_pred)

      scores += score
    }

    scores.toList
  }

}
