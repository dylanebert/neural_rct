import sys
import os
import numpy as np

track_size = 250

class TrackError(Exception):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)

def clean_file(filename):
	data = np.fromfile(filename, dtype=np.uint8)
	i = 0xa3
	while i > 0 and data[i] != 1:
		i -= 1
	while i > 2 and data[i-2] == 3:
		i -= 2
	bytes = np.zeros(track_size, dtype=np.uint8)
	j = 0
	while i < len(data) - 3 and data[i] != 0xff and j < track_size:
		bytes[j] = data[i]
		i += 2
		j += 1
	if data[i-2] != 2 and data[i-2] != 3:
		raise TrackError('Track \'{0}\' does not end with station. Ends with {1:x}'.format(filename, data[i-2]))
	for byte in bytes:
		pass
		#print('{0:x}'.format(byte))

def clean_dir(dirname):
	for filename in os.listdir(dirname):
		try:
			clean_file('{0}/{1}'.format(dirname, filename))
		except TrackError as e:
			print(e)
		
if __name__ == '__main__':
	if len(sys.argv) < 2:
		sys.exit('Invalid syntax. Use \'track_extractor [filename]\' or \'track_extractor -r [dirname]\'')	
	if len(sys.argv) == 3 and sys.argv[1] == '-r':
		clean_dir(sys.argv[2])
	elif len(sys.argv) == 2:
		try:
			clean_file(sys.argv[1])
		except TrackError as e:
			sys.exit(e)
	else:
		sys.exit('err')