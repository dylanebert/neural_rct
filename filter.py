import sys
import os
import numpy as np

def filter(tracktype, dirname):
	filtered = list()
	for filename in os.listdir(dirname):
		data = np.fromfile('{0}/{1}'.format(dirname, filename), dtype=np.uint8)
		if data[1] != int(tracktype, 16):
			filtered.append(filename)
	for filename in filtered:
		print(filename)
		os.remove('{0}/{1}'.format(dirname, filename))
	

if __name__ == '__main__':
	if len(sys.argv) != 3:
		sys.exit('Invalid syntax. Use filter [tracktype] [dirname]')
	filter(sys.argv[1], sys.argv[2])