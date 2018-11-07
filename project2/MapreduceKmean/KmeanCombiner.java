import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

/**
This file define the Reducer class whice can be used as combiner and reducer..
When it is used as combiner. 
**/

public class KmeanCombiner extends Reducer<Text,RecordWritable,Text, RecordWritable> {
	/**
	The main functino for the reducer class.
	The reduce
	**/
	public void reduce(Text key, Iterable<RecordWritable> values, Context context) 
		throws IOException, InterruptedException {

			RecordWritable res = new RecordWritable();
			for (RecordWritable val : values) {
				res = res.add(val);
			}
			//load banlance make the normalize() works fine in combiner.
			//res = res.normalize();
			context.write(key, res);
    }
}