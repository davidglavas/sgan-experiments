import argparse
import torch

"""
Tiny utility to print the command-line args used for a checkpoint
"""

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint')


def main(args):
	args = '../models/sgan-p-models/zara2_8_model.pt'
	#checkpoint = torch.load(args.checkpoint, map_location='cpu')
	checkpoint = torch.load(args, map_location='cpu')#

	command = ''
	for k, v in checkpoint['args'].items():
		if (v == False and isinstance(v, bool)):
			command += '--' + k + '=' + '0' + ' '
		elif (v == True and isinstance(v, bool)):
			command += '--' + k + '=' + '1' + ' '
		elif type(v) is tuple:
			assert len(v) == 1
			command += '--' + k + '=' + str(v[0]) + ' '
		else:
			command += '--' + k + '=' + str(v) + ' '
		print(k, v)

	print('\n\n Command with these arguments:\n')
	print(command)


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
