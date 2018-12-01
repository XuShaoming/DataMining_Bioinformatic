This project is to do the MapReduce K-mean on hadoop.

Please put the this folder under hadoop/share folder

To set up the environment please follow these steps.
1. install java. https://www.oracle.com/technetwork/java/javase/downloads/index.html
2. install hadoop. https://hadoop.apache.org/releases.html
3. set up hadoop. http://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html
   3.1 set dfs.name.dir value as PATH_TO_YOUR_HADOOP/hadoop-2.8.5/mynode/namenode
   3.2 set dfs.data.dir value as PATH_TO_YOUR_HADOOP/hadoop-2.8.5/mynode/datanode
   3.1 and 3.2 save the namenode and datanode in given place, which solves the namenode or datanode can't launch properly problems.	

To run the code you can use these shell.
1. compile.sh. It will compile the all java files and compact class file in the jar.
2. inithdfs.sh. It will init mynode folders in 3.1 and 3.2. Then build and start the namenode.
3. loadData.sh. It will load data/KmeanData.txt and data/KmeanCenter.txt to hdfs in proper place.
4. runkmean.sh. It will remove the previous output, then start new Kmean job. At last it will show you all result.

In data/data folder, I provide two python script.
1. pca_file.py do PCA on given data
   format:  python3 pca_file.py DATA_FILE.txt
   get: DATA_FILE_PCA.txt
2. get_KmeanCenter_file.py can randomly generate K centers from given data. Default seed is 20.
   format: python3 get_KmeanCenter_file.py K
   get: KmeanCenters.txt
