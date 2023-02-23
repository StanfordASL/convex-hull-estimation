# Analysis of the number of samples required
# to achieve epsilon-accuracy 
# for epsilon = 0.04.

import numpy as np
import matplotlib.pyplot as plt
from src.fibonacci import fibonacci_lattice

M = 100 # number of points
# M = 1300 # (Lipschitz case)
F_max = 0.005 # maximum magnitude of external force
r = np.sqrt(2) * F_max # radius of input ball
pts = fibonacci_lattice(r, M)

# plot
plt.figure().add_subplot(111, projection='3d').scatter(
	pts[:, 0], pts[:, 1], pts[:, 2])
plt.show()

# Compute maximal minimal distance between points
# as an conservative approximation of delta such 
# that the M points of the lattice form a 
# delta-covering.
# Indeed, points are approximately evenly spread
# on the surface of the sphere. Thus, the distance
# between neighbors (the min distance for each
# point) is approximately always the same. By 
# taking the maximum such distance for each point,
# we get an over-approximation of the value of
# delta (we approximately get delta/2 for large 
# numbers of samples M, so returning this maximum
# distance for delta gives a conservative 
# approximation). 
dists = np.ones((M, M))
for i in range(M):
	for j in range(M):
		if i == j:
			continue
		pi, pj = pts[i, :], pts[j, :]
		dists[i, j] = np.linalg.norm(pi - pj)
dists_min = np.min(dists, 1)
dist_max = np.max(dists_min)
delta = dist_max
print("delta =", delta)

# Hausdorff distance error bounds
T = 30 # planning horizon sec / number of controls
gamma = 9.0 / 4.0 # rescaling constant
u_max = 0.2 # maximum control magnitude
mass_min = 30 # minimum system mass
M_max = 1.0 / mass_min # maximum mass
# Smoothness constants
L_bar = (T**2 / (np.sqrt(2.0) * gamma)) * (
	max(gamma * M_max + F_max, u_max + F_max))
H_bar = T**2 / (2.0 * gamma)
alpha_delta = 1 + (1 - np.sqrt(1 - 2*delta / r)) / (
	1 + np.sqrt(1 - 2 * delta / r))
R_submersion = 1.0 / (alpha_delta**2 * (
	L_bar / r + H_bar))

epsilon_lipschitz = L_bar * delta
epsilon_submersion = delta**2 / (
	2 * R_submersion)

print("Hausdorff distance error bound (Lipschitz) =",
	epsilon_lipschitz)
print("Hausdorff distance error bound (submersion) =",
	epsilon_submersion)