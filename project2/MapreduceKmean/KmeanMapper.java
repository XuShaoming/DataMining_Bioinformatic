import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

/**
This file define the Mapper class.
**/

public class KmeanMapper extends Mapper<Object, Text, Text, RecordWritable>{

	private List<RecordWritable> centers;
	BufferedReader reader;

		/**
		Purpose:
			The setup function is used to read centers information from hdfs. In Kmean.java
			we have save the centers file as InitCenters in hdfs. Here we read and parse it to
			the list of RecordWritable List.
			The RecordWritable is defined in RecordWritable.java.
		**/
		public void setup(Context context) throws IOException, InterruptedException {
			centers = new ArrayList<>();
			
			if (context.getCacheFiles() != null && context.getCacheFiles().length > 0) {
		        reader = new BufferedReader(new FileReader("./InitCenters"));

				String line = reader.readLine();
				while(line != null) {
					StringTokenizer st = new StringTokenizer(line, "\t");
					List<Double> row = new ArrayList<>();
					int i = 0;
					while (st.hasMoreTokens()) {
						/*
						The assumption for this code is that the fisrt column of InitCenters file is 
						the id. So we skip the first value here.
						If you center data file not have Id. You have two choice.
						One is to preprocess the center data file to make it has ID.
						Second is to modify the code here. However, you also need to modify the 
						context.write(key,value)in reducer, to make the key to save empty string.
						*/
						if(i == 0) st.nextToken(); //ignore the first value.
						else row.add(new Double(st.nextToken()));
						i++;
					}
					centers.add(new RecordWritable(row, 1));
					line = reader.readLine();
				}
				reader.close();
			}
		}

	/**
	This is the main funcion to do map step.
	The map reduce feeds it Key, value, and context. 
	In general, map reduce will feed map data line by line. In our case, each line of data save in Text value.
	I think the Object Key in map is for the framework to assign jobs to different mappers.
	The Context context is used to submit the result key, vaule pair of mapper to next step. In our case, it
	will be send to combiner.
	**/
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
		StringTokenizer st = new StringTokenizer(value.toString(), "\t");
		List<Double> pt = new ArrayList<>();
		int j = 0;
		while (st.hasMoreTokens()) {
			/*Here, we skip first two column of input data. This can fit my project file format well.
			  However it can narrow the usage of this function.
			  To make it general. We can do these steps:
			  1. we need to preprocess the input file to seperate the label and data.
			  2. we feed the data to this application.
			  3. here we simiply keep line pt.add(new Double(st.nextToken()));
			 */
			if(j <= 1) st.nextToken();
			else pt.add(new Double(st.nextToken()));
			j++;
		}
		RecordWritable check = new RecordWritable(pt, 1);
		int min_index = 0;
		double min_distance = Double.MAX_VALUE;
		for(int i = 0; i < centers.size(); i++) {
			double distance = centers.get(i).euclideanDistance(check);
			if(distance < min_distance) {
				min_index = i;
				min_distance = distance;
			}
		}
		context.write(new Text(Integer.toString(min_index)), check);
	}

}