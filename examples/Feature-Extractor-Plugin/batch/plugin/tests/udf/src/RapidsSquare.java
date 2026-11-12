package fusiontest;
import ai.rapids.cudf.ColumnVector;
import com.nvidia.spark.RapidsUDF;
import org.apache.spark.sql.api.java.UDF1;
/** hand-written GPU UDF (RapidsUDF): x*x on the GPU through cuDF; black box for any fusion plugin */
public class RapidsSquare implements UDF1<Double, Double>, RapidsUDF {
  @Override public Double call(Double x) { return x == null ? null : x * x; }
  @Override public ColumnVector evaluateColumnar(int numRows, ColumnVector... args) { return args[0].mul(args[0]); }
}
