import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from src.stats import sample_pts_ellipsoid_surface
from src.utils import Hausdorff_dist_two_convex_hulls
from src.utils import length_spherical_cap_two_circles

"""
Creates a plot of number of samples
vs theoretical accuracy probability.
"""

np.random.seed(0)
B_generate_results = True

# ----------------------
epsilon = 1e-2 # desired accuracy
L_vec = np.array([1, 2]) # Lipschitz constant
# ----------------------

# ----------------------
# Ball input set, with ellipsoidal parameterization
# (xi-X_mu)^T X_Q^{-1} (xi-X_mu) <= 1.
# Code / bounds not tested for different 
# values of (x_dim, r, X_Q)
x_dim = 2
r = 1.
X_mu = np.zeros(x_dim)
X_Q = r**2 * np.eye(x_dim)
def f(x, L):
	# x - (x_dim, M) with M number of samples
	# L - scalar
	A = np.eye(x.shape[0])
	A[0, 0] = L
	return A @ x

def Y_Q_matrix(X_Q, L):
	# X_Q - (x_dim, x_dim) - Q-shape matrix parameterizing the
	# ellpisoidal set Y such that 
	# (yi-Y_mu)^T Y_Q^{-1} (yi-Y_mu) <= 1.
	# L - scalar
	A = np.eye(X_Q.shape[0])
	A[0, 0] = L
	Y_Q = A @ X_Q @ A.T
	return Y_Q
# ----------------------

# ----------------------
def covering_number_ball(r, eps):
	# Only works in 2d
	D = (2 * np.pi * r) / (2 * eps) + 1 # place balls on circle
	return D

def bound_conservatism_prob_naive(r, L, epsilon, M):
	# constants
	Lip_f = max(L, 1)

	# delta value required to achieve epsilon-accuracy
	delta = epsilon / Lip_f

	covering_number = covering_number_ball(r, delta / 2)

	# uniform distribution
	length_sampling = length_spherical_cap_two_circles(
		r, r, delta / 2)
	Lambda_delta = length_sampling / (2 * np.pi * r)

	# probability bound
	beta_M = covering_number * ((1 - Lambda_delta)**M)
	return np.maximum(1 - beta_M, 0)

def bound_conservatism_prob_diffeo(r, L, epsilon, M):
	# constants
	Lip_f = max(L, 1)
	Lip_f_inv = max(1 / L, 1)
	Rdiffeo = 1 / ((Lip_f / r) * (Lip_f * Lip_f_inv)**2)

	# delta value required to achieve epsilon-accuracy
	delta = np.sqrt(epsilon * 2 * Rdiffeo)

	covering_number = covering_number_ball(r, delta / 2)

	# uniform distribution
	length_sampling = length_spherical_cap_two_circles(
		r, r, delta / 2)
	Lambda_delta = length_sampling / (2 * np.pi * r)

	# probability bound
	beta_M = covering_number * ((1 - Lambda_delta)**M)
	return np.minimum(np.maximum(1 - beta_M, 0), 1)
# ----------------------

# ----------------------
if B_generate_results:
	N_runs = 100

	for L in L_vec:
		print("L =", L)
		M_vec_exp = np.geomspace(50, 400, num=20).astype(int)
		M_vec_theo = np.geomspace(30, 1e4, num=int(1e4)).astype(int)
		if L==2:
			M_vec_exp = np.geomspace(50, 300, num=20).astype(int)
			M_vec_theo = np.geomspace(30, 3e4, num=int(3e4)).astype(int)

		# ----------------------
		# Experimental
		haus_dists_exp = np.zeros((len(M_vec_exp), N_runs))
		# L4DC bound
		prob_bound_naive = np.zeros((len(M_vec_theo)))
		# New smooth bound
		prob_bound_diffeo = np.zeros((len(M_vec_theo)))

		# Ground truth
		Y_mu        = f(X_mu, L)
		Y_Q         = Y_Q_matrix(X_Q, L)
		ys_true     = sample_pts_ellipsoid_surface(
			Y_mu, Y_Q, NB_pts=2000)
		Y_true_hull = scipy.spatial.ConvexHull(ys_true.T)

		print("- Computing theoretical bounds")
		for j, M in enumerate(M_vec_theo):
			# L4DC theoretical bound (Naive Lipschitz)
			prob_bound_naive[j] = bound_conservatism_prob_naive(
				r, L, epsilon, M)
			# New smooth bound
			prob_bound_diffeo[j] = bound_conservatism_prob_diffeo(
				r, L, epsilon, M)

		# Empirical estimate
		print("- Evaluating empirical bounds")
		for j, M in enumerate(M_vec_exp):
			print("--- M =", M)
			for k in range(N_runs):
				xs = sample_pts_ellipsoid_surface(
					X_mu, X_Q, NB_pts=M)
				ys = f(xs, L)
				Y_est_hull = scipy.spatial.ConvexHull(ys.T)
				dist = Hausdorff_dist_two_convex_hulls(
					Y_true_hull, Y_est_hull)
				haus_dists_exp[j, k] = dist
	with open("sensitivity_L="+str(L)+".npy", 'wb') as f:
		np.savez(f,
			M_vec_exp=M_vec_exp,
			M_vec_theo=M_vec_theo,
			haus_dists_exp=haus_dists_exp,
			prob_bound_naive=prob_bound_naive,
			prob_bound_diffeo=prob_bound_diffeo)
# ----------------------

# ----------------------
# Data analysis and plotting

# https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
def conf_bounds_of_percentage(percentage, n, z=1.96):
    return (z*np.sqrt(percentage*(1-percentage)/n))

for L in L_vec:
	with open("sensitivity_L="+str(L)+".npy", 'rb') as f:
		data = np.load(f)
		L_vec = data['L_vec']
		M_vec_exp = data['M_vec_exp']
		M_vec_theo = data['M_vec_theo']
		haus_dists_exp = data['haus_dists_exp']
		prob_bound_naive = data['prob_bound_naive']
		prob_bound_diffeo = data['prob_bound_diffeo']
	N_runs = M_vec_exp.shape[-1]

	# How many violate the epsilon bound?
	dists_exp_violations = np.zeros_like(haus_dists_exp)
	for j, M in enumerate(M_vec_exp):
		for k in range(len(haus_dists_exp[0, :])):
			if haus_dists_exp[j, k] < epsilon:
				dists_exp_violations[j, k] = 1
	dists_exp_prob_violations = np.mean(dists_exp_violations, axis=-1)

	# Plot
	plt.figure(figsize=(7, 3.5))
	plt.subplot()
	M_max = 30000
	plt.plot(np.append(M_vec_exp, M_max),
		np.append(dists_exp_prob_violations, 1), 
		color='b', linewidth=3, label="Empirical")
	ci = conf_bounds_of_percentage(dists_exp_prob_violations, N_runs)
	plt.fill_between(M_vec_exp,
		np.maximum(dists_exp_prob_violations-ci, 0),
		np.minimum(dists_exp_prob_violations+ci, 1),
		color='b', alpha=.2)
	plt.plot(M_vec_theo, prob_bound_naive,
		color='r', linestyle="--", linewidth=3, label="Lipschitz")
	plt.plot(M_vec_theo, prob_bound_diffeo,
		color='g', linestyle="-.", linewidth=3, label="(New)")
	plt.plot(np.append(M_vec_exp, M_max),
		np.append(dists_exp_prob_violations, 1), 
		color='b', linewidth=3) # Replot
	plt.xlim([35, 3e4+5])
	plt.xlabel(r'$M$', fontsize=20)
	plt.ylabel(r'$\mathbb{P}(d_H(H(Y),\hat{Y}^M)\leq\epsilon)$',
		fontsize=20)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.gca().set_xscale('log')
	plt.grid(which='minor', alpha=0.5, linestyle='--')
	plt.grid(which='major', alpha=0.75, linestyle=':')
	plt.subplots_adjust(
		top=0.953,
		bottom=0.26,
		left=0.13,
		right=0.982,
		hspace=0.2,
		wspace=0.2)
	plt.tight_layout()
	if L==1:
		plt.legend(fontsize=17, loc='upper right', 
			bbox_to_anchor=(0.68, 0.75))
	if L==2:
		plt.legend(fontsize=17, loc='upper right',
			bbox_to_anchor=(0.83, 0.75))
	plt.show()
	# ----------------------