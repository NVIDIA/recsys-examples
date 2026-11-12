package fusiontest;
import org.apache.spark.sql.api.java.UDF2;
public class JavaUdfArith implements UDF2<Integer, Integer, Integer> {
  @Override public Integer call(Integer a, Integer b) { return (a == null || b == null) ? null : a * b + 1; }
}
