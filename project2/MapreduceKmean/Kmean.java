
import java.net.URI;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;

public class Kmean{

	/**
	This funcion is used to define the Job at where we define the driver, mapper, reducer, combiner of 
	our K-mean algorithm.
	it accepts Configuration object conf. In conf we can find the path to files which are needed for 
	our algorithm.
	**/
  	public static boolean run(Configuration conf) throws Exception{

  		Job job = Job.getInstance(conf, "K_mean");
	    job.setJarByClass(Kmean.class);
	    job.setMapperClass(KmeanMapper.class);
	    //job.setCombinerClass(KmeanCombiner.class);
	    job.setReducerClass(KmeanReducer.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(RecordWritable.class);
	    job.addCacheFile(new URI(conf.get("MEANS")+"#InitCenters"));
	    FileInputFormat.addInputPath(job, new Path(conf.get("InputPath")));
	    FileOutputFormat.setOutputPath(job, new Path(conf.get("OutputPath")));
	    return job.waitForCompletion(true);
  	}

  	// learn from https://github.com/andreaiacono/MapReduce/blob/master/src/main/java/samples/kmeans/Utils.java
  	/**
  	This function is used to read the content of output file from hdfs, convert it to string. it helps
  	when we check if two consecutive outputs are the same so as to check if algorithm is converged.
  	**/
  	public static String readReducerOutput(Configuration configuration, String path) throws IOException {
        FileSystem fs = FileSystem.get(configuration);
        FSDataInputStream dataInputStream = new FSDataInputStream(fs.open(new Path(configuration.get(path))));
        BufferedReader reader = new BufferedReader(new InputStreamReader(dataInputStream));
        StringBuilder content = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            content.append(line).append("\n");
        }

        return content.toString();
    }

    /**
	This is main funcion of k-mean.
	The outputPath+itr define the folder in hdfs to save the output of the algorithm.
	The "part-r-00000" is the default output filename of hadoop result.
	To easy my project, i predefine them. To make the program more general, we can pass them
	from args.

	When the consecutive outputs are the same, the k-mean converged. In this condition we stop
	algorithm.

	The Configuration object can be used to pass the path of files to Job. 
    **/
	public static void main(String[] args) throws Exception {

	  	String outputPath = "/user/xu/kmean/output/output";
	  	String outFile = "part-r-00000";
	  	boolean converged = false;
	  	int itr = 0;
	  	while(!converged) {
	  		Configuration conf = new Configuration();
	  		String prevMeanPath = outputPath + itr + "/" + outFile;
	  		String output = outputPath + (itr + 1);
	  		conf.set("InputPath", args[0]);
	  		conf.set("MEANS", prevMeanPath);
	  		conf.set("OutputPath", output);
	  		conf.set("OutputFile", output+ "/" + outFile);
	  		if(!Kmean.run(conf)) {
	  			System.exit(1);
	  		}
	  		String prevMean = Kmean.readReducerOutput(conf, "MEANS");
	  		String curMean = Kmean.readReducerOutput(conf, "OutputFile");
	  		if(prevMean.equals(curMean)){
	  			converged = true;
	  		}
	  		itr += 1;
	  	}
  	}
}