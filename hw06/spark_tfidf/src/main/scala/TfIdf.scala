import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class TfIdf(spark: SparkSession) {

  import spark.implicits._

  def build_index(
      input_df: DataFrame, top_n: Int,
      id_col: String, text_col: String): DataFrame = {

    val cleaned_text_col = regexp_replace(
        regexp_replace(lower(col(text_col)), "[^a-z]", " "),
        "\\s+",
        " "
      )

    val prepared_df = input_df
      .select(
        col(id_col),
        split(trim(cleaned_text_col), " ").as("words")
      )

    val all_docs = input_df.count()

    val term_freq_df = prepared_df
      .select(
        col(id_col),
        explode(col("words")).as("word"),
        size(col("words")).as("all_terms")
      )
      .groupBy(
        col(id_col),
        col("word"),
        col("all_terms")
      )
      .count()
      .filter(length(col("word")) > 2)
      .select(
        col(id_col),
        col("word"),
        (col("count") / col("all_terms")).as("tf")
      )

    term_freq_df.persist()
    term_freq_df.count()

    val doc_freq_df = term_freq_df
      .groupBy("word")
      .count()
      .orderBy(desc("count"))
      .limit(top_n)
      .select(
        col("word"),
        log(lit(all_docs) / col("count")).as("idf")
      )

    term_freq_df
      .join(
        doc_freq_df,
        "word"
      )
      .select(
        col(id_col),
        col("word"),
        (col("tf") * col("idf")).as("tfidf"),
        col(input_df.columns(0))
      )
      .groupBy(col(id_col))
      .pivot("word")
      .agg(max(col("tfidf")))
      .join(
        input_df.select(col(id_col)),
        Seq(id_col),
        "right_outer"
      )
      .na.fill(0.0)
  }
}
