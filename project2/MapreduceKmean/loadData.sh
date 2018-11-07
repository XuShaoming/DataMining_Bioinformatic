../../bin/hdfs dfs -rm -r /user/xu
../../bin/hdfs dfs -mkdir -p /user/xu/kmean/input
../../bin/hdfs dfs -mkdir -p /user/xu/kmean/output/output0
../../bin/hdfs dfs -put data/KmeanData.txt /user/xu/kmean/input
../../bin/hdfs dfs -put data/KmeanCenter.txt /user/xu/kmean/output/output0
../../bin/hdfs dfs -mv /user/xu/kmean/output/output0/KmeanCenter.txt /user/xu/kmean/output/output0/part-r-00000
