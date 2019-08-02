import argparse, root_numpy
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--filename', help='Provide a filename to read')
parser.add_argument('--branches', help='If you want, provide some branches to filter.', default='')

args = parser.parse_args()

def load_root(filename="Out.root_skimmedRH.root", branches=None):
    return root_numpy.root2array(filename, branches=branches)

if __name__ == '__main__':
    if args.branches == '':
        print(load_root(args.filename))
    else:
        print(load_root(args.filename, args.branches))
