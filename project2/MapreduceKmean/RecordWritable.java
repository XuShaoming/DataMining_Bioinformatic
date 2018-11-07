import java.util.ArrayList;
import java.util.List;
import java.io.DataInput;
import java.io.IOException;
import java.io.DataOutput;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

/**
Purpose:
	The RecordWritable class is used to build the value in (key, value) pair of Map Reduce K-mean.
	The object of RecordWritable has pt and num parameters.
	the pt is List<Double> type which save the sum of records in axis=0
	the num is int type, which determines the number of the records that sum up as pt.
	Typically, when num is 1, the pt represent a single record comes from a line from data.
**/

public class RecordWritable implements Writable{
	private List<Double> pt;
	private int num;

	public RecordWritable() {
		this.pt = new ArrayList<>();
		this.num = 0;
	}

	public RecordWritable(List<Double> pt, int num) {
		this.pt = new ArrayList<>(pt);
		this.num = num;
	}

	public List<Double> getPt(){
		return new ArrayList<Double>(this.pt);
	}

	public int getNum(){
		return this.num;
	}

	public void setPt(List<Double> pt){
		if(this.pt.size() == 0) 
			setNum(1);
		this.pt = new ArrayList<Double>(pt);
	}

	public void setNum(int num) {
		this.num = num;
	}

	public RecordWritable add(RecordWritable other) {
		/**
		This function use to add differnt RecordWritable together.
		**/

		if(this.pt.size() == 0){
			return new RecordWritable(other.getPt(), other.getNum());
		}

		List<Double> newPt = new ArrayList<>();
		for(int i = 0; i < this.pt.size(); i++) {
			newPt.add(this.pt.get(i) + other.pt.get(i));
		}

		return new RecordWritable(newPt, this.num + other.num);
	}

	public RecordWritable normalize() {
		/**
		This function is used to calcualte the mean of the current RecordWritable object.
		It will return a new RecordWritable object with num = 1.
		**/

		List<Double> normPt = new ArrayList<>();
		for(int i = 0; i < this.pt.size(); i++) {
			normPt.add(this.pt.get(i) / this.num);
		}

		return new RecordWritable(normPt, 1);
	}

	public double euclideanDistance(RecordWritable other) {
		/**
		This function calculate the euclideanDistance between two RecordWritable object.
		**/

		List<Double> otherPt = other.getPt();

		if(this.pt.size() != otherPt.size()) {
			throw new IndexOutOfBoundsException("length of two points must be equal!"); 
		}

		double res = 0.0;
		for(int i = 0; i < this.pt.size(); i++){
			res += Math.pow(this.pt.get(i) - otherPt.get(i), 2);
		}
		return Math.sqrt(res);
	}

	public String toString() {

		StringBuilder sb = new StringBuilder();

		for(Double val : this.pt){
			sb.append(val+"\t");
		}

		return sb.toString();
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		Text t = new Text();
		t.readFields(in);
		this.pt = new ArrayList<Double>();
		this.num = 1;
		String[] vals = t.toString().split("\t");
		for(String val : vals){
			pt.add(new Double(val));
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		StringBuilder sb = new StringBuilder();

		for(Double val : this.pt){
			sb.append(val+"\t");
		}
		new Text(sb.toString()).write(out);
	}
}