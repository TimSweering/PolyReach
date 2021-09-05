#!/usr/bin/env python
from scripts.ReachabilityAlgorithm import PolyReach
from matplotlib import pyplot as plt
import sys
PolyReach.plot_trajectory_from_file(sys.argv[1])
plt.show()