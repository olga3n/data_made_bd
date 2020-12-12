import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

object TfIdfApp extends App {
  
  if (args.length < 2) {
    println("Arguments: input_file output_file")
    sys.exit(1)
  }

  val input_path = args(0)
  val output_path = args(1)

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("tfidf-app")
    .getOrCreate()

  import spark.implicits._

  val input_df = spark.read
    .option("header", "true")
    .csv(input_path)

  val prepared_df = input_df
    .select(
      row_number().over(
        Window.orderBy(input_df.columns(0))
      ).as("doc_id"),
      col(input_df.columns(0)).as("text")
    )

  val index = new TfIdf(spark)

  val result = index
    .build_index(prepared_df, 100, "doc_id", "text")

  result.write
    .option("header", "true")
    .csv(output_path)

  spark.stop()
}
