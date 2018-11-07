import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

/**
This file define the Reducer class.
**/

public class KmeanReducer extends Reducer<Text,RecordWritable,Text, RecordWritable> {
	/**
	The main function for the reducer class.
	The reduce accept the key value pairs from mapper and compute the mean of values pair as 
	the new centers.
	the values type is RecordWritable. the res.add(val) will add the pts location and number.
	the normalize can used to computer the mean of values.
	**/
	public void reduce(Text key, Iterable<RecordWritable> values, Context context) 
		throws IOException, InterruptedException {

			RecordWritable res = new RecordWritable();
			for (RecordWritable val : values) {
				res = res.add(val);
			}
			res = res.normalize();
			context.write(key, res);
    }
}