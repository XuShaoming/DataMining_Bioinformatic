# DataMining_Bioinformatic
### Fall 2018, University at Buffalo.
## Description
   This repository includes three projects from course cs601 DataMining&Bioinformatic.
## Prerequisites
There two versions of source code. To run it, you need:  
#### To run the .py version  
To intall python3 :  
	https://www.python.org/download/releases/3.0/  
To install Numpy and Matplotlib :  
    	https://www.scipy.org/install.html     
#### To run the .ipynb version:  
To intall python3, Numpy and Matplotlib as mentioned above.   
To install jupyter notebook :  
	http://jupyter.org/install  
        
## Project 1:
  #### First part: PCA algorithm  
  I implement the PCA algorithm from scratch, then compare its performance with SVD and T-SNE algorithms by using the Matplotlib scatter graph.  
  Command to run the pca.py:	python3 pca.py
  #### Second part: Appriori and Association rule algorithm
  I implement the appriori and association rule algorithm from scratch, then write three templetes to mining rules.  
  ###### command to run apriori.py  
   This command will save the result of apriori on support as 0.5 in support_50.p pickle file.	
   python3 apriori.py filename support
    example:  python3 apriori.py ../data/associationruletestdata.txt 0.5
  ###### command to run asso_rule.py
   the filename link the pickle file which saved apriori result for given support.  
   ../data/support_50.p means the result got by setting support as 0.5 in apriori. If you can't find this file, you need to run apriori.py to generate this file first, before run asso_rule.py.  
   Command Format  
   python3 asso_rule.py filename confidence  
	Example:  python3 asso_rule.py ../data/support_50.p 0.7  
	
## Project 2:
  Implemented Hierarchical Agglomerative, DBSCAN, Expectation Maximization clustering algorithms, and MapReduce K-mean algorithm on Hadoop.
  
## Project 3:
  Implemented K Nearest Neighbor, Naïve Bayes, Decision Tree, Random Forest, and AdaBoost classification algorithms. And used N-fold Cross Validation and F1 score to measure their performance
  
