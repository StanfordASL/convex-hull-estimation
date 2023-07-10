import numpy as np
import osqp

from scipy import sparse
from scipy.sparse import csr_matrix, vstack, hstack, eye
from scipy.spatial import ConvexHull
from time import time

import jax.numpy as jnp
from jax import jacfwd, jit, vmap
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import rc, rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)

from src.stats import sample_pts_ellipsoid_surface
from src.fibonacci import fibonacci_lattice

np.random.seed(0)
B_generate_results = True

# -----------------------------------------
T = 30 # planning horizon sec / number of controls
T_mid = int(T / 2) + 5 # intermediate time
M = 100 # number of samples
n_x = 4 # (px, py, vx, vy) - number of state variables
n_u = 2 # (ux, uy) - number of control variables
n_pos = 2 # number of position variables
n_obs = 2 # number of obstacles (two halfplanes)

epsilon_padding = 0.025 # constraints padding

# constants for uncertain parameters
gamma = 9.0 / 4.0 # rescaling coefficient
u_max = 0.2 # max control force
F_max = 0.005 # max external force
mass_nom = 32.0 # nominal mass
# -----------------------------------------

# -----------------------------------------
class Model:
    def __init__(self, M, epsilon_padding=epsilon_padding):
        # initial / final states
        self.x0 = jnp.zeros(n_x)

        # X_goal = {x: xg-xg_delta <= x <= xg+xg_delta}
        self.pg = jnp.array([2.0, 0.05])
        self.pg_deltas = jnp.array([0.3, 0.14])
        self.u_max = u_max
        self.u_min = -self.u_max

        # linear obstacle avoidance constraints written as
        # H[i, :] @ p <= h[i] for obstacle i
        self.obs_H = jnp.zeros((n_obs, 2))
        self.obs_h = jnp.zeros((n_obs))
        c1 = jnp.array([0., -0.1])
        n1 = jnp.array([1., -5.])
        n1 = n1 / jnp.linalg.norm(n1)
        c2 = jnp.array([2., -0.1])
        n2 = jnp.array([-1., -5.])
        n2 = n2 / jnp.linalg.norm(n2)
        h1 = n1 @ c1
        h2 = n2 @ c2
        self.obs_H = self.obs_H.at[0, :].set(n1)
        self.obs_h = self.obs_h.at[0].set(h1)
        self.obs_H = self.obs_H.at[1, :].set(n2)
        self.obs_h = self.obs_h.at[1].set(h2)

        # uncertain parameters (force and mass)
        parameters = fibonacci_lattice(np.sqrt(2) * F_max, M)
        self.forces = parameters[:, :2]
        masses_inv = (1.0 / mass_nom) + (parameters[:, 2] / gamma)
        self.masses = 1.0 / masses_inv
        print("masses =", min(self.masses), ',', max(self.masses))

        # Constraints relaxation constant
        self.epsilon_padding = epsilon_padding

        self.define_problem()

    def get_nb_vars(self):
        return n_u*T

    def convert_us_vec_to_us_mat(self, us_vec):
        us_mat = jnp.reshape(us_vec, (n_u, T), 'F')
        us_mat = us_mat.T # (T, n_u)
        return us_mat

    def convert_us_mat_to_us_vec(self, us_mat):
        us_vec = jnp.reshape(us_mat, (T*n_u), 'F')
        return us_vec

    def next_state(self, x, u, mass, force):
        dt = 1.0
        p, v = x[:n_pos], x[n_pos:]
        pn = p + dt * v + (dt**2 / (2.0 * mass)) * (u + force)
        vn = v + dt * (1.0 / mass) * (u + force)
        xn = jnp.concatenate((pn, vn))
        return xn

    @partial(jit, static_argnums=(0,))
    def us_to_state_trajectory(self, us_mat, mass, force):
        xs = jnp.zeros((T+1, n_x))
        xs = xs.at[0, :].set(self.x0)
        for t in range(T):
            xt, ut = xs[t, :], us_mat[t, :]
            xs = xs.at[t+1, :].set(self.next_state(xt, ut, mass, force))
        return xs
    def us_to_state_trajectories(self, us_mat):
        Us = jnp.repeat(us_mat[jnp.newaxis, :, :], M, axis=0)
        Xs = vmap(self.us_to_state_trajectory)(
            Us, self.masses, self.forces)
        return Xs

    def final_constraints(self, xs):
        pT = xs[-1, :n_pos]
        return (pT - self.pg)

    def obstacle_avoidance_constraints(self, xs):
        gs = jnp.zeros(T)
        for t in range(T+1):
            pt = xs[t, :n_pos]
            if t <= T_mid:
                # first obstacle
                gs = gs.at[t].set(self.obs_H[0, :]@pt-self.obs_h[0])
            else:
                # second obstacle
                gs = gs.at[t].set(self.obs_H[1, :]@pt-self.obs_h[1])
        gs = gs + self.epsilon_padding
        return gs


    ### ----------------------------------------------------
    ### ----------------------------------------------------
    ### ----------------------------------------------------
    ### ----------------------------------------------------
    ### OSQP

    @partial(jit, static_argnums=(0,))
    def get_control_constraints_coeffs_all(self):
        # Returns (A, l, u) corresponding to control constraints
        # such that l <= A uvec <= u.
        A = jnp.eye(n_u*T)
        l = self.u_min * jnp.ones(n_u*T)
        u = self.u_max * jnp.ones(n_u*T)
        return A, l, u

    @partial(jit, static_argnums=(0,))
    def get_final_constraints_coeffs(self, us_mat, mass, force):
        # Returns (A, l, u) corresponding to final constraints
        # such that l <= A uvec <= u.
        def final_constraints_us(us_mat, mass, force):
            xs = self.us_to_state_trajectory(us_mat, mass, force)
            val = self.final_constraints(xs)
            return val
        def final_constraints_dus(us_mat, mass, force):
            return jacfwd(final_constraints_us)(us_mat, mass, force)
        val = final_constraints_us(us_mat, mass, force)
        val_du = final_constraints_dus(us_mat, mass, force)
        val_du = jnp.reshape(val_du, (n_pos, n_u*T), 'C') # reshape gradient
        val = -val + val_du @ self.convert_us_mat_to_us_vec(us_mat)
        val_lower = val - self.pg_deltas + self.epsilon_padding
        val_upper = val + self.pg_deltas - self.epsilon_padding
        return val_du, val_lower, val_upper
    @partial(jit, static_argnums=(0,))
    def get_final_constraints_coeffs_all(self, us_mat):
        Us = jnp.repeat(us_mat[jnp.newaxis, :, :], M, axis=0)
        v_du, v_low, v_up = vmap(self.get_final_constraints_coeffs)(
            Us, self.masses, self.forces)
        v_du  = jnp.reshape(v_du,  (n_pos*M, n_u*T), 'F')
        v_low = jnp.reshape(v_low, (n_pos*M,), 'F')
        v_up  = jnp.reshape(v_up,  (n_pos*M,), 'F')
        return v_du, v_low, v_up

    @partial(jit, static_argnums=(0,))
    def get_obstacle_avoidance_constraints_coeffs(self, us_mat, mass, force):
        # Returns (A, l, u) corresponding to obstacle avoidance constraints
        # such that l <= A uvec <= u.
        def obstacle_avoidance_constraints_us(us_mat, mass, force):
            xs = self.us_to_state_trajectory(us_mat, mass, force)
            val = self.obstacle_avoidance_constraints(xs)
            return val
        def obstacle_avoidance_constraints_dus(us_mat, mass, force):
            return jacfwd(obstacle_avoidance_constraints_us)(us_mat, mass, force)
        val = obstacle_avoidance_constraints_us(us_mat, mass, force)
        val_du = obstacle_avoidance_constraints_dus(us_mat, mass, force)
        val_du = jnp.reshape(val_du, (T, n_u*T), 'C') # reshape gradient
        val = -val + val_du @ self.convert_us_mat_to_us_vec(us_mat)
        val_lower = -jnp.inf*jnp.ones(T)
        val_upper = val
        return val_du, val_lower, val_upper
    @partial(jit, static_argnums=(0,))
    def get_obstacle_avoidance_constraints_coeffs_all(self, us_mat):
        Us = jnp.repeat(us_mat[jnp.newaxis, :, :], M, axis=0)
        v_du, v_low, v_up = vmap(self.get_obstacle_avoidance_constraints_coeffs)(
            Us, self.masses, self.forces)
        v_du  = jnp.reshape(v_du,  (T*M, n_u*T), 'F')
        v_low = jnp.reshape(v_low, (T*M,), 'F')
        v_up  = jnp.reshape(v_up,  (T*M,), 'F')
        return v_du, v_low, v_up

    def get_objective_coeffs(self):
        # Returns (P, q) corresponding to objective
        #        min (1/2 z^T P z + q^T z)
        # where z = umat is the optimization variable.
        R = 0.01 * np.eye(n_u)
        # Quadratic Objective
        P = sparse.block_diag([sparse.kron(eye(T), R)], format='csc')
        # Linear Objective
        q = np.zeros(T*n_u)
        return P, q

    def get_constraints_coeffs(self):
        us = jnp.zeros(self.get_nb_vars())
        us = self.convert_us_vec_to_us_mat(us)
        us = us + 1e-2 # to avoid zero gradients

        # Constraints: l <= A z <= u, with z = umat

        # control constraints
        A_con, l_con, u_con = self.get_control_constraints_coeffs_all()
        # final constraints
        A_xf, l_xf, u_xf = self.get_final_constraints_coeffs_all(us)
        # obstacle avoidance
        A_obs, l_obs, u_obs = self.get_obstacle_avoidance_constraints_coeffs_all(us)

        A_xf, A_obs, A_con = csr_matrix(A_xf), csr_matrix(A_obs), csr_matrix(A_con)
        A = vstack([A_xf, A_obs, A_con], format='csc')
        l = np.hstack([l_xf, l_obs, l_con])
        u = np.hstack([u_xf, u_obs, u_con])
        return A, l, u

    def define_problem(self):
        # objective and constraints
        self.P, self.q         = self.get_objective_coeffs()
        self.A, self.l, self.u = self.get_constraints_coeffs()
        # Setup OSQP problem
        self.osqp_prob = osqp.OSQP()
        print("OSQP Problem size: ",
              "P =",self.P.shape,"q =",self.q.shape,
              "A =",self.A.shape,"l =",self.l.shape,"u =",self.u.shape)
        self.osqp_prob.setup(self.P, self.q, self.A, self.l, self.u, 
                             warm_start=False, verbose=False)
        return True

    def solve(self):
        self.res = self.osqp_prob.solve()
        if self.res.info.status != 'solved':
            print("[solve]: Problem infeasible.")
        us_sol = self.convert_us_vec_to_us_mat(self.res.x)
        return us_sol
# -----------------------------------------

# -----------------------------------------
if B_generate_results:
    model = Model(M)
    start, goal = model.x0[:n_pos], model.pg
    goal_deltas = model.pg_deltas

    start_time = time()
    model.define_problem()
    define_elapsed = time() - start_time
    start_time = time()
    us = model.solve()
    solve_elapsed = time() - start_time

    print("define elapsed = ", define_elapsed)
    print("solve elapsed = ", solve_elapsed)
    print("total elapsed =", define_elapsed + solve_elapsed)

    xs = model.us_to_state_trajectories(us)

    with open('robust_planning_results.npy', 'wb') as f:
        np.save(f, start)
        np.save(f, goal)
        np.save(f, goal_deltas)
        np.save(f, us)
        np.save(f, xs)
# -----------------------------------------

# -----------------------------------------
# plot
with open('robust_planning_results.npy', 'rb') as f:
    start = np.load(f)
    goal = np.load(f)
    goal_deltas = np.load(f)
    us = np.load(f)
    xs = np.load(f)

fig = plt.figure(figsize=[7, 3])
plt.subplot()
plt.scatter(start[0], start[1], color='k')
plt.scatter(goal[0], goal[1], color='k')
obstacle = Polygon([(0, -0.1), (1, 0.1), (2, -0.1),],
    color='r', alpha=0.3)
goal_region = Polygon([
    (goal[0]-goal_deltas[0], goal[1]+goal_deltas[1]),
    (goal[0]+goal_deltas[0], goal[1]+goal_deltas[1]),
    (goal[0]+goal_deltas[0], goal[1]-goal_deltas[1]),
    (goal[0]-goal_deltas[0], goal[1]-goal_deltas[1]),],
    color='k', alpha=0.2)
plt.gca().add_patch(obstacle)
plt.gca().add_patch(goal_region)
offset = 0.15
plt.text(start[0]-offset, start[1]+0.04, 
    r'$x_{0}$', fontsize=24, weight="bold")
plt.text(1.0-offset, -0.05, 
    r'$\mathcal{X}_{obs}$', fontsize=28)
plt.text(goal[0]-offset, goal[1]+0.165, 
    r'$\mathcal{X}_{goal}$', fontsize=24, weight="bold")
plt.text(0.1, 0.175, 
    r'$\mathcal{Y}_{u}(t)$', fontsize=24, weight="bold", color='b')
for t in range(1, T+1):
    hull = ConvexHull(xs[:, t, :n_pos])
    for simplex in hull.simplices:
        plt.plot(xs[simplex, t, 0], xs[simplex, t, 1],
            'b', alpha=0.7)
    plt.fill(xs[hull.vertices, t, 0], xs[hull.vertices, t, 1], 
        color='b', alpha=0.15)
plt.xlabel(r'$p_x$', fontsize=24)
plt.ylabel(r'$p_y$', fontsize=24, rotation=0)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim([-0.2, 2.2])
plt.ylim([-0.1, 0.275])
plt.subplots_adjust(
    top=0.953,
    bottom=0.26,
    left=0.13,
    right=0.982,
    hspace=0.2,
    wspace=0.2)
plt.tight_layout()
plt.show()
plt.close()

fig = plt.figure(figsize=[7, 3])
plt.subplot()
plt.step(np.arange(T+1), np.append(us[:, 0], us[-1, 0]),
    where='post', color='r', label=r'$u_1(t)$')
plt.step(np.arange(T+1), np.append(us[:, 1], us[-1, 1]),
    where='post', color='b', label=r'$u_2(t)$')
plt.xlabel(r'$t$', fontsize=24)
plt.ylabel(r'$u(t)$', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.subplots_adjust(
    top=0.953,
    bottom=0.26,
    left=0.13,
    right=0.982,
    hspace=0.2,
    wspace=0.2)
plt.tight_layout()
plt.grid()
plt.show()
# -----------------------------------------