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

def pla(x,y):
	random.seed(time.time())
	w = np.zeros((1, 5), dtype=np.float64)
	visit = list(range(x.T[0].size))
	random.shuffle(visit)
	flag = 1
	t = 0
	while flag:
		flag = 0
		for n in visit:
			dot = sign(np.dot(w,x[n]))
			if dot != y[n]:
				t += 1
				w += x[n]*y[n]
				flag = 1
				
	return t


def main():
	raw = pd.read_csv("./data/hw1_6_train.dat", sep='\s+', index_col=False, header=None)
	x = np.ones((raw[0].size,5))
	x[:,1:] = raw.iloc[:,0:4]
	y = raw.iloc[:,4].astype(np.float64)
	t = []

	for i in range(1126):
		t.append(pla(x,y))
		hashes = '#' * int(i/1126 * 20)
		spaces = ' ' * (20 - len(hashes))
		sys.stdout.write("\r[%s] %f%%"%(hashes + spaces,i/1126*100))
		sys.stdout.flush()
	print()
	print(sum(t)/1126)

	plt.hist(t, bins='auto')
	plt.xlabel('number of updates')  
	plt.ylabel('frequency') 
	plt.title('normal PLA')
	plt.axvline(sum(t)/1126, color='b', linestyle='dashed', linewidth=2, label='mean='+str(sum(t)/1126))
	plt.legend(loc='upper right')
	plt.savefig('./pla.png')

if __name__ == '__main__':
	main()