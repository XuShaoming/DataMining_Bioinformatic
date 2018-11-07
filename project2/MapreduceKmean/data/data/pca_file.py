import mylibrary as mylib
import sys

if __name__ == "__main__":
	print(sys.argv[1])
	mylib.get_pca_file(sys.argv[1])