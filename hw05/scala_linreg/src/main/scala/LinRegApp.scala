import java.io.{BufferedWriter, File, FileWriter}
import breeze.linalg._
import breeze.stats._

object LinRegApp extends App {
  
  if (args.length < 3) {
    println("Arguments: train_file test_file result_file")
    sys.exit(1)
  }

  val train_file = args(0);
  val test_file = args(1);

  val result_file = args(2);

  val train_data = scala.io.Source.fromFile(train_file)
    .getLines.map(_.split(',').map(_.toDouble)).toSeq

  val test_data = scala.io.Source.fromFile(test_file)
    .getLines.map(_.split(',').map(_.toDouble)).toSeq

  val train_data_x = train_data.map(x => x.slice(0, x.size - 1))
  val train_data_y = train_data.map(x => x(x.size - 1))

  val x_train = DenseMatrix(train_data_x: _*)
  val x_test = DenseMatrix(test_data: _*)

  val y_train = DenseVector(train_data_y: _*)

  val model = new LinReg();

  val cv_scores = model.cv_scores(x_train, y_train, 4);
  val cv_scores_size = cv_scores.length

  for ((score, i) <- cv_scores.view.zipWithIndex) {
      println(f"[CV] fold: $i/$cv_scores_size, error: $score")
  }

  val cv_score_vec = DenseVector(cv_scores: _*)

  val mean_val = mean(cv_score_vec)
  val min_val = min(cv_score_vec)
  val max_val = max(cv_score_vec)
  val std_val = stddev(cv_score_vec)

  println(
    f"[CV] mean: $mean_val%.5f, std: $std_val%.5f, " +
    f"min: $min_val%.5f, max: $max_val%.5f")

  model.fit(x_train, y_train)

  val y_test = model.predict(x_test)

  val file = new File(result_file)

  val bw = new BufferedWriter(new FileWriter(file))

  for (line <- y_test.toArray.map(_.toString)) {
      bw.write(line)
      bw.newLine()
  }

  bw.close()
}
