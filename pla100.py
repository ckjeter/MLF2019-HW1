import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def sign(num):
    if num <= 0:
        return -1.0
    return 1.0

def errorcount(w, x, y):
	count = 0
	for n in range(y.size):
		if sign(np.dot(w, x[n])) != y[n]:
			count += 1
	return count

def pocket(x,y,x_test,y_test):
	random.seed(time.time())
	w = np.zeros((1, 5), dtype=np.float64)
	visit = list(range(y.size))
	random.shuffle(visit)
	flag = 1
	t = 0
	t_error = 0
	#train
	while t <= 100:
		for n in visit:
			dot = sign(np.dot(w,x[n]))
			if dot != y[n]:
				w = w + x[n]*y[n]
				t += 1

	#test
	error = errorcount(w, x_test, y_test)/y_test.size
	return error


def main():
	raw = pd.read_csv("./data/hw1_7_train.dat", sep='\s+', index_col=False, header=None)
	raw_test = pd.read_csv("./data/hw1_7_test.dat", sep='\s+', index_col=False, header=None)
	x = np.ones((raw[0].size,5))
	x[:,1:] = raw.iloc[:,0:4]
	y = raw.iloc[:,4].astype(np.float64)

	x_test = np.ones((raw_test[0].size,5))
	x_test[:,1:] = raw_test.iloc[:,0:4]
	y_test = raw_test.iloc[:,4].astype(np.float64)

	error = []

	for i in range(1126):
		error.append(pocket(x,y,x_test,y_test))

		hashes = '#' * int(i/1126 * 20)
		spaces = ' ' * (20 - len(hashes))
		sys.stdout.write("\r[%s] %f%%  %f"%(hashes + spaces,i/1126*100, error[i]))
		sys.stdout.flush()

		#print(error[i])

	print()
	print(sum(error)/1126)

	plt.hist(error, bins='auto')
	plt.xlabel('error rate')
	plt.ylabel('frequency')
	plt.title('PLA with 100 updates')
	plt.axvline(sum(error)/1126, color='b', linestyle='dashed', linewidth=2, label='mean='+str(sum(error)/1126))
	plt.legend(loc='upper right')
	plt.savefig('./pla100.png')
    plt.show()

if __name__ == '__main__':
	main()
