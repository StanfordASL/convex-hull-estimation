# Estimating the convex hull of the image of a set with smooth boundary: error bounds and applications

## About

Code for our work on geometric reconstruction using convex hull approximations (T. Lew, R. Bonalli, L. Janson, M. Pavone, "Estimating the convex hull of the image of a set with smooth boundary: error bounds and applications").
* The geometric inference experiment (Section 5.1) can be reproduced by running
``python sensitivity_analysis.py``
* The robust planning experiment (Section 5.4) can be reproduced by running
``python planning_compute_robust_trajectory.py``
* The error bound for the robust planning experiment (Section 5.4 / Appendix) can be obtained by running
``python planning_bound.py``
* For further experiments, please refer to the following repositories:
	* For neural network verification: https://github.com/StanfordASL/nn_robustness_analysis. 
	* Code for L4DC 2022 publication: https://github.com/StanfordASL/RandUP.
	* Code for CoRL 2020 publication: https://github.com/StanfordASL/UP.
	* Some hardware results: https://youtu.be/sDkblTwPuEg.

Sample python code of the sampling + convex hull algorithm:
```bash
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
# define the problem
X_min = -np.array([1., 1.])
X_max =  np.array([1., 1.])
def f(x):
	return x
# sample + take the convex hull
M    = 100
xs   = np.random.uniform(low=X_max, high=X_min, size=(M, 2))
ys   = f(xs)
hull = scipy.spatial.ConvexHull(ys)
# plot
plt.scatter(ys[:, 0], ys[:, 1], color='b')
for s in hull.simplices:
	plt.plot(ys[s, 0], ys[s, 1], 'g')
plt.show()
```
## Setup
This code was tested with Python 3.6.3 and Python 3.7.13.

All dependencies (i.e., numpy, scipy, jax, and matplotlib) can be installed by running 
``
  pip install -r requirements.txt
``