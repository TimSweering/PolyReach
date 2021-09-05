#!/usr/bin/env python
from scripts.ReachabilityAlgorithm import read_parameter_file
from matplotlib import pyplot as plt
import sys
sys.path.append("..")
# print('---------\n%s\n---------' % sys.argv[1])
# file_in = 'parameter_files/param.json'
# read_parameter_file(sys.argv[1])
read_parameter_file(sys.argv[1])
plt.show()