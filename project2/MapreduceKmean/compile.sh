#!/bin/sh

rm *.class
rm *.jar
../../bin/hadoop com.sun.tools.javac.Main RecordWritable.java
../../bin/hadoop com.sun.tools.javac.Main KmeanMapper.java 
../../bin/hadoop com.sun.tools.javac.Main KmeanCombiner.java 
../../bin/hadoop com.sun.tools.javac.Main KmeanReducer.java
../../bin/hadoop com.sun.tools.javac.Main Kmean.java

jar -cf km.jar *.class

