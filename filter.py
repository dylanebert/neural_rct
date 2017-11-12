import sys
import os
import numpy as np

def filter(tracktype, dirname):
	filtered = list()
	new_dir = '{0}/filtered'.format(dirname)
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)
	for filename in os.listdir(dirname):
		if os.path.isdir('{0}/{1}'.format(dirname, filename)):
			continue
		data = np.fromfile('{0}/{1}'.format(dirname, filename), dtype=np.uint8)
		if data[1] == int(tracktype, 16):
			filtered.append(filename)
	for filename in filtered:
		print(filename)
		with open('{0}/{1}'.format(new_dir, filename), 'wb') as f:
			f.write(bytearray(np.fromfile('{0}/{1}'.format(dirname, filename), dtype=np.uint8)))

if __name__ == '__main__':
	if len(sys.argv) != 3:
		sys.exit('Invalid syntax. Use filter [dirname] [tracktype]')
	filter(sys.argv[2], sys.argv[1])