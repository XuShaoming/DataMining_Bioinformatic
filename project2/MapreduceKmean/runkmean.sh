#!/bin/sh
#../../sbin/stop-dfs.sh
#../../sbin/start-dfs.sh
../../bin/hdfs dfs -rm -r /user/xu/kmean/output/output[1-9]*
../../bin/hadoop jar km.jar Kmean /user/xu/kmean/input /user/xu/kmean/output
../../bin/hdfs dfs -cat /user/xu/kmean/output/output[0-9]/*


