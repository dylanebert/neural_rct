import sys
import os
import numpy as np

segment_names = {
	'ELEM_FLAT'                            			   : 0x0,
	'ELEM_END_STATION'                                 : 0x1,
	'ELEM_BEGIN_STATION'                               : 0x2,
	'ELEM_MIDDLE_STATION'                              : 0x3,
	'ELEM_25_DEG_UP'                                     : 0x4,
	'ELEM_60_DEG_UP'                                     : 0x5,
	'ELEM_FLAT_TO_25_DEG_UP'                             : 0x6,
	'ELEM_25_DEG_UP_TO_60_DEG_UP'                        : 0x7,
	'ELEM_60_DEG_UP_TO_25_DEG_UP'                        : 0x8,
	'ELEM_25_DEG_UP_TO_FLAT'                             : 0x9,
	'ELEM_25_DEG_DOWN'                                  : 0x0a,
	'ELEM_60_DEG_DOWN'                                  : 0x0b,
	'ELEM_FLAT_TO_25_DEG_DOWN'                          : 0x0c,
	'ELEM_25_DEG_DOWN_TO_60_DEG_DOWN'                   : 0x0d,
	'ELEM_60_DEG_DOWN_TO_25_DEG_DOWN'                   : 0x0e,
	'ELEM_25_DEG_DOWN_TO_FLAT'                           : 0x0f,
	'ELEM_LEFT_QUARTER_TURN_5_TILES'                     : 0x10,
	'ELEM_RIGHT_QUARTER_TURN_5_TILES'                    : 0x11,
	'ELEM_FLAT_TO_LEFT_BANK'                             : 0x12,
	'ELEM_FLAT_TO_RIGHT_BANK'                            : 0x13,
	'ELEM_LEFT_BANK_TO_FLAT'                             : 0x14,
	'ELEM_RIGHT_BANK_TO_FLAT'                            : 0x15,
	'ELEM_BANKED_LEFT_QUARTER_TURN_5_TILES'              : 0x16,
	'ELEM_BANKED_RIGHT_QUARTER_TURN_5_TILES'             : 0x17,
	'ELEM_LEFT_BANK_TO_25_DEG_UP'                        : 0x18,
	'ELEM_RIGHT_BANK_TO_25_DEG_UP'                       : 0x19,
	'ELEM_25_DEG_UP_TO_LEFT_BANK'                        : 0x1a,
	'ELEM_25_DEG_UP_TO_RIGHT_BANK'                       : 0x1b,
	'ELEM_LEFT_BANK_TO_25_DEG_DOWN'                     : 0x1c,
	'ELEM_RIGHT_BANK_TO_25_DEG_DOWN'                    : 0x1d,
	'ELEM_25_DEG_DOWN_TO_LEFT_BANK'                      : 0x1e,
	'ELEM_25_DEG_DOWN_TO_RIGHT_BANK'                     : 0x1f,
	'ELEM_LEFT_BANK'                                     : 0x20,
	'ELEM_RIGHT_BANK'                                    : 0x21,
	'ELEM_LEFT_QUARTER_TURN_5_TILES_25_DEG_UP'    : 0x22,
	'ELEM_RIGHT_QUARTER_TURN_5_TILES_25_DEG_UP'   : 0x23,
	'ELEM_LEFT_QUARTER_TURN_5_TILES_25_DEG_DOWN' : 0x24,
	'ELEM_RIGHT_QUARTER_TURN_5_TILES_25_DEG_DOWN' : 0x25,
	'ELEM_S_BEND_LEFT'                            : 0x26,
	'ELEM_S_BEND_RIGHT'                           : 0x27,
	'ELEM_LEFT_VERTICAL_LOOP'                     : 0x28,
	'ELEM_RIGHT_VERTICAL_LOOP'                    : 0x29,
	'ELEM_LEFT_QUARTER_TURN_3_TILES'              : 0x2a,
	'ELEM_RIGHT_QUARTER_TURN_3_TILES'             : 0x2b,
	'ELEM_LEFT_QUARTER_TURN_3_TILES_BANK'         : 0x2c,
	'ELEM_RIGHT_QUARTER_TURN_3_TILES_BANK'        : 0x2d,
	'ELEM_LEFT_QUARTER_TURN_3_TILES_25_DEG_UP'    : 0x2e,
	'ELEM_RIGHT_QUARTER_TURN_3_TILES_25_DEG_UP'   : 0x2f,
	'ELEM_LEFT_QUARTER_TURN_3_TILES_25_DEG_DOWN' : 0x30,
	'ELEM_RIGHT_QUARTER_TURN_3_TILES_25_DEG_DOWN' : 0x31,
	'ELEM_LEFT_QUARTER_TURN_1_TILE'               : 0x32,
	'ELEM_RIGHT_QUARTER_TURN_1_TILE'              : 0x33,
	'ELEM_LEFT_TWIST_DOWN_TO_UP'  : 0x34,
	'ELEM_RIGHT_TWIST_DOWN_TO_UP' : 0x35,
	'ELEM_LEFT_TWIST_UP_TO_DOWN' : 0x36,
	'ELEM_RIGHT_TWIST_UP_TO_DOWN' : 0x37,
	'ELEM_HALF_LOOP_UP'           : 0x38,
	'ELEM_HALF_LOOP_DOWN'        : 0x39,
	'ELEM_LEFT_CORKSCREW_UP'      : 0x3a,
	'ELEM_RIGHT_CORKSCREW_UP'     : 0x3b,
	'ELEM_LEFT_CORKSCREW_DOWN'   : 0x3c,
	'ELEM_RIGHT_CORKSCREW_DOWN'  : 0x3d,
	'ELEM_FLAT_TO_60_DEG_UP'   : 0x3e,
	'ELEM_60_DEG_UP_TO_FLAT'   : 0x3f,
	'ELEM_FLAT_TO_60_DEG_DOWN' : 0x40,
	'ELEM_60_DEG_DOWN_TO_FLAT' : 0x41,
	'ELEM_FLAT_COVERED'                       : 0x44,
	'ELEM_25_DEG_UP_COVERED'                  : 0x45,
	'ELEM_60_DEG_UP_COVERED'                  : 0x46,
	'ELEM_FLAT_TO_25_DEG_UP_COVERED'          : 0x47,
	'ELEM_25_DEG_UP_TO_60_DEG_UP_COVERED'     : 0x48,
	'ELEM_60_DEG_UP_TO_25_DEG_UP_COVERED'     : 0x49,
	'ELEM_25_DEG_UP_TO_FLAT_COVERED'          : 0x4a,
	'ELEM_25_DEG_DOWN_COVERED'                : 0x4b,
	'ELEM_60_DEG_DOWN_COVERED'                : 0x4c,
	'ELEM_FLAT_TO_25_DEG_DOWN_COVERED'        : 0x4d,
	'ELEM_25_DEG_DOWN_TO_60_DEG_DOWN_COVERED' : 0x4e,
	'ELEM_60_DEG_DOWN_TO_25_DEG_DOWN_COVERED' : 0x4f,
	'ELEM_25_DEG_DOWN_TO_FLAT_COVERED'        : 0x50,
	'ELEM_LEFT_QUARTER_TURN_5_TILES_COVERED'  : 0x51,
	'ELEM_RIGHT_QUARTER_TURN_5_TILES_COVERED' : 0x52,
	'ELEM_S_BEND_LEFT_COVERED'                : 0x53,
	'ELEM_S_BEND_RIGHT_COVERED'               : 0x54,
	'ELEM_LEFT_QUARTER_TURN_3_TILES_COVERED'  : 0x55,
	'ELEM_RIGHT_QUARTER_TURN_3_TILES_COVERED' : 0x56,
	'ELEM_LEFT_HALF_BANKED_HELIX_UP_SMALL'    : 0x57,
	'ELEM_RIGHT_HALF_BANKED_HELIX_UP_SMALL'   : 0x58,
	'ELEM_LEFT_HALF_BANKED_HELIX_DOWN_SMALL'  : 0x59,
	'ELEM_RIGHT_HALF_BANKED_HELIX_DOWN_SMALL' : 0x5a,
	'ELEM_LEFT_HALF_BANKED_HELIX_UP_LARGE'    : 0x5b,
	'ELEM_RIGHT_HALF_BANKED_HELIX_UP_LARGE'   : 0x5c,
	'ELEM_LEFT_HALF_BANKED_HELIX_DOWN_LARGE'  : 0x5d,
	'ELEM_RIGHT_HALF_BANKED_HELIX_DOWN_LARGE' : 0x5e,
	'ELEM_LEFT_QUARTER_TURN_1_TILE_60_DEG_UP'    : 0x5f,
	'ELEM_RIGHT_QUARTER_TURN_1_TILE_60_DEG_UP'   : 0x60,
	'ELEM_LEFT_QUARTER_TURN_1_TILE_60_DEG_DOWN' : 0x61,
	'ELEM_RIGHT_QUARTER_TURN_1_TILE_60_DEG_DOWN' : 0x62,
	'ELEM_BRAKES' : 0x63,
	'ELEM_ROTATION_CONTROL_TOGGLE'                 : 0x64,
	'ELEM_INVERTED_90_DEG_UP_TO_FLAT_QUARTER_LOOP' : 0x65,
	'ELEM_LEFT_QUARTER_BANKED_HELIX_LARGE_UP'      : 0x66,
	'ELEM_RIGHT_QUARTER_BANKED_HELIX_LARGE_UP'     : 0x67,
	'ELEM_LEFT_QUARTER_BANKED_HELIX_LARGE_DOWN'   : 0x68,
	'ELEM_RIGHT_QUARTER_BANKED_HELIX_LARGE_DOWN'  : 0x69,
	'ELEM_LEFT_QUARTER_HELIX_LARGE_UP'             : 0x6a,
	'ELEM_RIGHT_QUARTER_HELIX_LARGE_UP'            : 0x6b,
	'ELEM_LEFT_QUARTER_HELIX_LARGE_DOWN'          : 0x6c,
	'ELEM_RIGHT_QUARTER_HELIX_LARGE_DOWN'         : 0x6d,
	'ELEM_25_DEG_UP_LEFT_BANKED'  : 0x6e,
	'ELEM_25_DEG_UP_RIGHT_BANKED' : 0x6f,
	'ELEM_ON_RIDE_PHOTO' : 0x72,
	'ELEM_25_DEG_DOWN_LEFT_BANKED'  : 0x73,
	'ELEM_25_DEG_DOWN_RIGHT_BANKED' : 0x74,
	'ELEM_WATER_SPLASH' : 0x75,
	'ELEM_FLAT_TO_60_DEG_UP_LONG_BASE' : 0x76,
	'ELEM_60_DEG_UP_TO_FLAT_LONG_BASE' : 0x77,
	'ELEM_60_DEG_DOWN_TO_FLAT_LONG_BASE' : 0x79,
	'ELEM_FLAT_TO_60_DEG_DOWN_LONG_BASE' : 0x7a,
	'ELEM_CABLE_LIFT_HILL'             : 0x7b,
	'ELEM_90_DEG_UP'                  : 0x7e,
	'ELEM_90_DEG_DOWN'               : 0x7f,
	'ELEM_60_DEG_UP_TO_90_DEG_UP'     : 0x80,
	'ELEM_90_DEG_DOWN_TO_60_DEG_DOWN' : 0x81,
	'ELEM_90_DEG_UP_TO_60_DEG_UP'     : 0x82,
	'ELEM_60_DEG_DOWN_TO_90_DEG_DOWN' : 0x83,
	'ELEM_BRAKE_FOR_DROP'             : 0x84,
	'ELEM_LEFT_EIGHTH_TO_DIAG'             : 0x85,
	'ELEM_RIGHT_EIGHTH_TO_DIAG'            : 0x86,
	'ELEM_LEFT_EIGHTH_TO_ORTHOGONAL'       : 0x87,
	'ELEM_RIGHT_EIGHTH_TO_ORTHOGONAL'      : 0x88,
	'ELEM_LEFT_EIGHTH_BANK_TO_DIAG'        : 0x89,
	'ELEM_RIGHT_EIGHTH_BANK_TO_DIAG'       : 0x8a,
	'ELEM_LEFT_EIGHTH_BANK_TO_ORTHOGONAL'  : 0x8b,
	'ELEM_RIGHT_EIGHTH_BANK_TO_ORTHOGONAL' : 0x8c,
	'ELEM_DIAG_FLAT'                       : 0x8d,
	'ELEM_DIAG_25_DEG_UP'                  : 0x8e,
	'ELEM_DIAG_60_DEG_UP'                  : 0x8f,
	'ELEM_DIAG_FLAT_TO_25_DEG_UP'          : 0x90,
	'ELEM_DIAG_25_DEG_UP_TO_60_DEG_UP'     : 0x91,
	'ELEM_DIAG_60_DEG_UP_TO_25_DEG_UP'     : 0x92,
	'ELEM_DIAG_25_DEG_UP_TO_FLAT'          : 0x93,
	'ELEM_DIAG_25_DEG_DOWN'               : 0x94,
	'ELEM_DIAG_60_DEG_DOWN'               : 0x95,
	'ELEM_DIAG_FLAT_TO_25_DEG_DOWN'       : 0x96,
	'ELEM_DIAG_25_DEG_DOWN_TO_60_DEG_DOWN' : 0x97,
	'ELEM_DIAG_60_DEG_DOWN_TO_25_DEG_DOWN' : 0x98,
	'ELEM_DIAG_25_DEG_DOWN_TO_FLAT'        : 0x99,
	'ELEM_DIAG_FLAT_TO_60_DEG_UP'          : 0x9a,
	'ELEM_DIAG_60_DEG_UP_TO_FLAT'          : 0x9b,
	'ELEM_DIAG_FLAT_TO_60_DEG_DOWN'       : 0x9c,
	'ELEM_DIAG_60_DEG_DOWN_TO_FLAT'        : 0x9d,
	'ELEM_DIAG_FLAT_TO_LEFT_BANK'          : 0x9e,
	'ELEM_DIAG_FLAT_TO_RIGHT_BANK'         : 0x9f,
	'ELEM_DIAG_LEFT_BANK_TO_FLAT'          : 0xa0,
	'ELEM_DIAG_RIGHT_BANK_TO_FLAT'         : 0xa1,
	'ELEM_DIAG_LEFT_BANK_TO_25_DEG_UP'     : 0xa2,
	'ELEM_DIAG_RIGHT_BANK_TO_25_DEG_UP'    : 0xa3,
	'ELEM_DIAG_25_DEG_UP_TO_LEFT_BANK'     : 0xa4,
	'ELEM_DIAG_25_DEG_UP_TO_RIGHT_BANK'    : 0xa5,
	'ELEM_DIAG_LEFT_BANK_TO_25_DEG_DOWN'  : 0xa6,
	'ELEM_DIAG_RIGHT_BANK_TO_25_DEG_DOWN' : 0xa7,
	'ELEM_DIAG_25_DEG_DOWN_TO_LEFT_BANK'   : 0xa8,
	'ELEM_DIAG_25_DEG_DOWN_TO_RIGHT_BANK'  : 0xa9,
	'ELEM_DIAG_LEFT_BANK'                  : 0xaa,
	'ELEM_DIAG_RIGHT_BANK'                 : 0xab,
	'ELEM_LEFT_BARREL_ROLL_UP_TO_DOWN' : 0xae,
	'ELEM_RIGHT_BARREL_ROLL_UP_TO_DOWN' : 0xaf,
	'ELEM_LEFT_BARREL_ROLL_DOWN_TO_UP'  : 0xb0,
	'ELEM_RIGHT_BARREL_ROLL_DOWN_TO_UP' : 0xb1,
	'ELEM_LEFT_BANK_TO_LEFT_QUARTER_TURN_3_TILES_25_DEG_UP'     : 0xb2,
	'ELEM_RIGHT_BANK_TO_RIGHT_QUARTER_TURN_3_TILES_25_DEG_UP'   : 0xb3,
	'ELEM_LEFT_QUARTER_TURN_3_TILES_25_DEG_DOWN_TO_LEFT_BANK'   : 0xb4,
	'ELEM_RIGHT_QUARTER_TURN_3_TILES_25_DEG_DOWN_TO_RIGHT_BANK' : 0xb5,
	'ELEM_POWERED_LIFT' : 0xb6,
	'ELEM_LEFT_LARGE_HALF_LOOP_UP'    : 0xb7,
	'ELEM_RIGHT_LARGE_HALF_LOOP_UP'   : 0xb8,
	'ELEM_RIGHT_LARGE_HALF_LOOP_DOWN' : 0xb9,
	'ELEM_LEFT_LARGE_HALF_LOOP_DOWN' : 0xba,
	'ELEM_LEFT_FLYER_TWIST_UP_TO_DOWN'                : 0xBB,
	'ELEM_RIGHT_FLYER_TWIST_UP_TO_DOWN'               : 0xBC,
	'ELEM_LEFT_FLYER_TWIST_DOWN_TO_UP'                 : 0xBD,
	'ELEM_RIGHT_FLYER_TWIST_DOWN_TO_UP'                : 0xBE,
	'ELEM_FLYER_HALF_LOOP_UP'                          : 0xBF,
	'ELEM_FLYER_HALF_LOOP_DOWN'                       : 0xC0,
	'ELEM_LEFT_FLY_CORKSCREW_UP_TO_DOWN'              : 0xC1,
	'ELEM_RIGHT_FLY_CORKSCREW_UP_TO_DOWN'             : 0xC2,
	'ELEM_LEFT_FLY_CORKSCREW_DOWN_TO_UP'               : 0xC3,
	'ELEM_RIGHT_FLY_CORKSCREW_DOWN_TO_UP'              : 0xC4,
	'ELEM_HEARTLINE_TRANSFER_UP'                       : 0xC5,
	'ELEM_HEARTLINE_TRANSFER_DOWN'                    : 0xC6,
	'ELEM_LEFT_HEARTLINE_ROLL'                        : 0xC7,
	'ELEM_RIGHT_HEARTLINE_ROLL'                        : 0xC8,
	'ELEM_INVERTED_FLAT_TO_90_DEG_DOWN_QUARTER_LOOP'   : 0xCE,
	'ELEM_90_DEG_UP_QUARTER_LOOP_TO_INVERTED'          : 0xCF,
	'ELEM_QUARTER_LOOP_INVERT_TO_90_DEG_DOWN'         : 0xD0,
	'ELEM_LEFT_CURVED_LIFT_HILL'                       : 0xD1,
	'ELEM_RIGHT_CURVED_LIFT_HILL'                      : 0xD2,
	'ELEM_BLOCK_BRAKES'                                : 0xD8,
	'ELEM_BANKED_LEFT_QUARTER_TURN_3_TILES_25_DEG_UP'  : 0xD9,
	'ELEM_BANKED_RIGHT_QUARTER_TURN_3_TILES_25_DEG_UP' : 0xDA,
	'TRACK_END' : 0xFF
}
segment_dict = dict(zip(segment_names.values(), segment_names.keys()))

def clean_data(dir):
	bad_data = list()
	for filename in os.listdir(dir):
		data = np.fromfile('{0}/{1}'.format(dir, filename), dtype = np.uint8)
		segments = list()
		i = 0xa3
		while data[i] != 1:
			i -= 1
		while data[i-2] == 3:
			i -= 2
		if data[i-2] == 2:
			i -= 2
		while i < len(data) - 1:
			if data[i] == 0xff:
				if data[i-2] != 0x3:
					bad_data.append(filename)
				break	
			if data[i] not in segment_dict:
				bad_data.append(filename)
				break
			i += 2
	for file in bad_data:
		os.remove('{0}/{1}'.format(dir, file))
		print('Removed {0}'.format(file))
			
def get_raw_data(dir):
	segments = list()
	for filename in os.listdir(dir):
		data = np.fromfile('{0}/{1}'.format(dir, filename), dtype = np.uint8)
		i = 0xa3
		while data[i] != 1:
			i -= 1
		while data[i-2] == 3:
			i -= 2
		if data[i-2] == 2:
			i -= 2
		while i < len(data) - 1:
			if data[i] not in segment_dict:
				sys.exit('error, clean data')
			segments.append(data[i])
			if data[i] == 0xff:
				break
			i += 2
	segments = np.array(segments, dtype = np.int32)
	return segments
			
def print_data(dir):
	for filename in os.listdir(dir):
		print(filename)
		data = np.fromfile('{0}/{1}'.format(dir, filename), dtype = np.uint8)
		i = 0xa3
		while data[i] != 1:
			i -= 1
		while data[i-2] == 3:
			i -= 2
		if data[i-2] == 2:
			i -= 2
		while i < len(data) - 1:
			if data[i] == 0xff:
				break				
			try:
				print('{0:x}: {1}'.format(i, segment_dict[data[i]]))
			except:
				sys.exit('error, clean data')
			i += 2
		
if __name__ == '__main__':
	if sys.argv[1] == 'clean':
		dir = sys.argv[2]
		clean_data(dir)
	else:
		dir = sys.argv[1]
		print_data(dir)