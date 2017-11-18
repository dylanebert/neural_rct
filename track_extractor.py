#Attempts to extract only the track segment bytes from TD6 files. Will probably fail on about half of them.

import sys
import os
import numpy as np

class TrackError(Exception):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)

def clean_file(filename, dirname):
	path = '{0}/{1}'.format(dirname, filename)
	if os.path.isdir(path):
		return
	data = np.fromfile(path, dtype=np.uint8)
	i = 0xa3
	try:
		while i > 0 and data[i] != 1:
			i -= 1
		while i > 2 and data[i-2] == 3:
			i -= 2
	except:
		raise TrackError('Start error with {0}'.format(filename))
	start_i = i
	segment_counters = dict()
	qualifiers = list()
	bytes = list()
	odd_start = 0
	consec_odd = 0
	while i < len(data) - 3 and data[i] != 0xff:
		#if data[i] > 0xda:
			#raise TrackError('Invalid track piece {0:x}: {1:x} on {2}'.format(i, data[i], filename))
		if data[i-1] == 0xff:
			raise TrackError('Detected shift at {0:x} on {1}. Started at {2:x}'.format(i, filename, start_i))
		bytes.append(data[i])
		if data[i] not in segment_counters:
			segment_counters[data[i]] = 0
		segment_counters[data[i]] += 1
		if data[i+1] not in qualifiers:
			qualifiers.append(data[i+1])
			if consec_odd == 0:
				odd_start = i
			elif consec_odd > 2:
				while i > start_i:
					bytes.pop()
					try:
						segment_counters[data[i]] -= 1
					except:
						raise TrackError('Unknown error near {0:x} on {1}'.format(i, filename))
					if segment_counters[data[i]] == 0:
						i += 1
						consec_odd = 0
						break
					i -= 2					
				#raise TrackError('Error detected starting near {0:x} on {1}'.format(i, filename))
			consec_odd += 1
		else:
			consec_odd = 0
		i += 2
	if data[i-2] != 2 and data[i-2] != 3:
		raise TrackError('Track \'{0}\' does not end with station. Start {1:x}: {2:x}; End {3:x}: {4:x}'.format(filename, start_i, data[start_i], i-2, data[i-2]))
	if not os.path.exists('extracted_tracks'):
		os.makedirs('extracted_tracks')
	with open('extracted_tracks/{1}'.format(dirname, filename), 'wb') as f:
		f.write(bytearray(bytes))
	os.remove(path)

def clean_dir(dirname):
	for filename in os.listdir(dirname):
		try:
			clean_file(filename, dirname)
		except TrackError as e:
			print(e)
	if len(os.listdir(dirname)) == 0:
		os.rmdir(dirname)
		return
		
if __name__ == '__main__':
	if len(sys.argv) < 2:
		sys.exit('Invalid syntax. Use \'track_extractor [filename]\' or \'track_extractor -r [dirname]\'')	
	if len(sys.argv) == 3 and sys.argv[1] == '-r':
		clean_dir(sys.argv[2])
	elif len(sys.argv) == 3:
		try:
			clean_file(sys.argv[2], sys.argv[1])
		except TrackError as e:
			sys.exit(e)
	else:
		sys.exit('err')