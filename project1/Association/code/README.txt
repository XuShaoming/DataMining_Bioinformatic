In this project, I implement the appriori and association rule algorithm from scratch, then write three templetes to mining rules. There two versions of source code. To run it, you need:
    
To run the .py version

    To intall python3 : 
        https://www.python.org/download/releases/3.0/

    To install Numpy and Matplotlib :
    	https://www.scipy.org/install.html


To run the .ipynb version:

	To intall python3, Numpy and Matplotlib as mentioned above. 

	To install jupyter notebook :
        http://jupyter.org/install

The command to run apriori.py
	This command will save the result of apriori on support as 0.5 in support_50.p pickle file.	
	python3 apriori.py filename support
		example:  python3 apriori.py ../data/associationruletestdata.txt 0.5

The command to run asso_rule.py
	the filename link the pickle file which saved apriori result for given support.
	../data/support_50.p means the result got by setting support as 0.5 in apriori. If 
	you can't find this file, you need to run apriori.py to generate this file first, before
	run asso_rule.py.

	Command Format
	python3 asso_rule.py filename confidence
		Example:  python3 asso_rule.py ../data/support_50.p 0.7