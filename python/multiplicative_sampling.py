import numpy as np
import scipy as sp
import math
import pandas as pd 
import time
import sys
from numba import njit
import os
import geweke
import gc

import multiplicative_gibbs as m_gibbs

def convergence_geweke_test(trace,top5_beta_trace,start,end):
	max_z = []

	## convergence for the trace values
	n = trace.shape[1]
	for t in range(n):
		trace_convergence = trace[start:end,t]
		trace_t_convergence_zscores = geweke.geweke(trace_convergence)[:,1]
		max_z.append(np.amax(np.absolute(trace_t_convergence_zscores)))

	m = top5_beta_trace.shape[1]
	for b in range(m):
		top_beta_convergence = top5_beta_trace[start:end,b]
		beta_b_convergence_zscores = geweke.geweke(top_beta_convergence)[:,1]
		max_z.append(np.amax(np.absolute(beta_b_convergence_zscores)))

	if np.amax(max_z) < 1.5:
		return(1)

def welford(mean,M2,x,n):
	n = n + 1

	delta = x - mean
	mean += delta / n
	delta2 = x - mean
	M2 += delta * delta2

	return(mean,M2)

def sampling(verbose,y,C,HapDM,prefix,num,trace_container,gamma_container,beta_container,alpha_container,pi_b):

	## set random seed for the process
	np.random.seed(int(time.time()) + os.getpid())

	#initiate beta,gamma and H matrix
	C_r, C_c = C.shape

	H = np.array(HapDM)
	H = np.asfortranarray(H)

	H_r,H_c = H.shape

	##specify hyper parameters
	pie_a = 1
	pie_b = H_c*pi_b
	a_sigma = 1
	b_sigma = 1
	a_e = 1
	b_e = 1

	sigma_1 = math.sqrt(1/np.random.gamma(a_sigma,b_sigma))
	sigma_e = math.sqrt(1/np.random.gamma(a_e,b_e))
	pie = np.random.beta(pie_a,pie_b)
	
	print("parameter initiation:",sigma_1,sigma_e,pie)

	#initiate alpha, alpha_trace, beta_trace and gamma_trace

	it = 0
	burn_in_iter = 2000
	step_size = 2000


	convergence_start_iter = burn_in_iter
	convergence_end_iter = np.array(range(convergence_start_iter*2,convergence_start_iter+step_size*4,step_size))

	convergence_iter = convergence_start_iter+step_size*3


	trace = np.empty((convergence_end_iter[-1]-burn_in_iter,7))
	top5_beta_trace = np.empty((convergence_end_iter[-1]-burn_in_iter,5))

	alpha = np.random.random(size = C_c)
	gamma = np.random.binomial(1,pie,H_c)
	beta = np.array(np.zeros(H_c))
	
	for i in range(H_c):
		if gamma[i] == 0:
			beta[i] = 0
		else:
			beta[i] = np.random.normal(0,sigma_1) 

	## Pre-compute the H_beta and C_alpha in the beginning, and update them later in the MCMC
	H_beta = m_gibbs.circle_product_matrix(H,beta)
	C_alpha = np.matmul(C,alpha)

	C_norm_2 = np.sum(C**2,axis=0)
	H_norm_2 = np.sum(H**2,axis=0)

	#first sampling for convergence

	while it < convergence_iter:
	
		before = time.time()
		sigma_1 = m_gibbs.sample_sigma_1(beta,gamma,a_sigma,b_sigma)
		if sigma_1 < 0.05:
			sigma_1 = 0.05
		pie = m_gibbs.sample_pie(gamma,pie_a,pie_b)
		sigma_e = m_gibbs.sample_sigma_e(y,H_beta,C_alpha,a_e,b_e)
		gamma = m_gibbs.sample_gamma_numba(y,C_alpha,H,beta,pie,sigma_1,sigma_e,gamma,H_beta)
		alpha,C_alpha = m_gibbs.sample_alpha(y,C,alpha,sigma_e,H_beta,C_alpha)
		beta,H_beta = m_gibbs.sample_beta_numba(y,C_alpha,H,beta,gamma,sigma_1,sigma_e,H_beta)
		genetic_var = np.var(H_beta)
		pheno_var = np.var(y - C_alpha)
		large_beta_ratio = np.sum(np.absolute(beta) > 0.3) / len(beta)
		total_heritability = genetic_var / pheno_var
		alpha_norm = np.linalg.norm(alpha, ord=2)
		beta_norm = np.linalg.norm(beta, ord=2)

		after = time.time()
		if it > 2000 and (total_heritability > 1):
			print("unrealistic beta sample",it,genetic_var,pheno_var,total_heritability)
			continue

		else:
			if verbose:
				print(it,str(after - before),pie,sigma_1,sigma_e,sum(gamma),large_beta_ratio,max(abs(beta)),total_heritability)

			if it >= burn_in_iter:
				trace[it-burn_in_iter,:] = [alpha_norm,beta_norm,sigma_1,sigma_e,large_beta_ratio,total_heritability,sum(gamma)]
				top5_beta_trace[it-burn_in_iter,:] = np.sort(np.absolute(beta))[::-1][:5]

			## test for convergence using values in trace from multiple consecutive draws
			
			if it == convergence_end_iter[-1] - 1:
				
				num_convergence_test = len(convergence_end_iter)

				convergence_scores = np.zeros(len(convergence_end_iter))

				for s in range(num_convergence_test):
					convergence_scores[s] = convergence_geweke_test(trace,top5_beta_trace,convergence_start_iter-burn_in_iter,convergence_end_iter[s]-burn_in_iter)

				if np.sum(convergence_scores) == num_convergence_test:
					print("convergence has been reached at %i iterations. The MCMC Chain has enterred a stationary stage" %(it))
					print("trace values:", trace[it-burn_in_iter,:])
					break
				else:
					trace_ = np.empty((1000,7))
					top5_beta_trace_ = np.empty((1000,5))


					trace = np.concatenate((trace[-(convergence_iter - burn_in_iter-1000):,:],trace_),axis=0)
					top5_beta_trace = np.concatenate((top5_beta_trace[-(convergence_iter - burn_in_iter-1000):,:],top5_beta_trace_),axis = 0)

					burn_in_iter += 1000
					convergence_iter += 1000

					convergence_start_iter += 1000
					convergence_end_iter += 1000

					print(it,burn_in_iter,convergence_iter,convergence_start_iter,convergence_end_iter,trace.shape)

			it += 1



	## MCMC draws for posterior mean

	posterior_draws = 10000

	alpha_mean = np.zeros(C_c)
	beta_mean = np.zeros(H_c)
	gamma_sum = np.zeros(H_c)

	alpha_M2 = np.zeros(C_c)
	beta_M2 = np.zeros(H_c)

	posterior_trace = np.empty((posterior_draws,7))

	alpha_trace = np.empty((posterior_draws,C_c))

	it = 0

	while it < posterior_draws:
	
		before = time.time()
		sigma_1 = m_gibbs.sample_sigma_1(beta,gamma,a_sigma,b_sigma)
		if sigma_1 < 0.05:
			sigma_1 = 0.05
		pie = m_gibbs.sample_pie(gamma,pie_a,pie_b)
		sigma_e = m_gibbs.sample_sigma_e(y,H_beta,C_alpha,a_e,b_e)
		gamma = m_gibbs.sample_gamma_numba(y,C_alpha,H,beta,pie,sigma_1,sigma_e,gamma,H_beta)
		alpha,C_alpha = m_gibbs.sample_alpha(y,C,alpha,sigma_e,H_beta,C_alpha)
		beta,H_beta = m_gibbs.sample_beta_numba(y,C_alpha,H,beta,gamma,sigma_1,sigma_e,H_beta)
		genetic_var = np.var(H_beta)
		pheno_var = np.var(y - C_alpha)
		large_beta_ratio = np.sum(np.absolute(beta) > 0.3) / len(beta)
		total_heritability = genetic_var / pheno_var
		alpha_norm = np.linalg.norm(alpha, ord=2)
		beta_norm = np.linalg.norm(beta, ord=2)

		after = time.time()
		if total_heritability > 1:
			print("unrealistic beta sample",it,genetic_var,pheno_var,total_heritability)
			continue

		else:
			if verbose:
				print(it,str(after - before),pie,sigma_1,sigma_e,sum(gamma),large_beta_ratio,max(abs(beta)),total_heritability)

			posterior_trace[it,:] = [alpha_norm,beta_norm,sigma_1,sigma_e,large_beta_ratio,total_heritability,sum(gamma)]
			alpha_trace[it,:] = alpha
			beta_mean,beta_M2 = welford(beta_mean,beta_M2,beta,it)
			alpha_mean,alpha_M2 = welford(alpha_mean,alpha_M2,alpha,it)
			gamma_sum += gamma

			if it % 2000 == 0:
				print("Posterior draws: %i iterations have sampled" %(it), str(after - before),posterior_trace[it,:])

			it += 1

	trace_container[num] = posterior_trace

	#alpha values
	alpha_container[num] = {'avg': alpha_mean,
							'M2': alpha_M2}

	#beta values
	beta_container[num] = {'avg':beta_mean,
							'M2':beta_M2}

	gamma_container[num] = gamma_sum / posterior_draws



