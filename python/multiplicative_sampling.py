import numpy as np
import math
import time
import os

import multiplicative_gibbs as m_gibbs
import utility

def sampling(verbose,y,C,H,prefix,num,trace_container,gamma_container,beta_container,alpha_container,convergence_container,pi_b):

	convergence_container[num] = 0

	## set random seed for the process
	np.random.seed(int(time.time()) + os.getpid())

	#initiate beta,gamma and H matrix
	C_r, C_c = C.shape

	H_r,H_c = H.shape

	##specify hyper parameters
	pie_a = 1
	pie_b = H_c*pi_b
	a_sigma = 1
	b_sigma = 1
	a_e = 1
	b_e = 1

	sigma_1 = math.sqrt(1/np.random.gamma(a_sigma,1/b_sigma))
	sigma_e = math.sqrt(1/np.random.gamma(a_e,1/b_e))
	pie = np.random.beta(pie_a,pie_b)
	
	if verbose > 0:
		print("parameter initiation:",sigma_1,sigma_e,pie)

	it = 0
	burn_in_iter = 2000

	convergence_start_iter = burn_in_iter
	convergence_end_iter = burn_in_iter + 10000

	trace = np.empty((convergence_end_iter-burn_in_iter,7))
	top5_beta_trace = np.empty((convergence_end_iter-burn_in_iter,5))

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

	#C_norm_2 = np.sum(C**2,axis=0)
	#H_norm_2 = utility.col_norm2_chunked(H, chunk_rows=2000)

	#first sampling for convergence

	while it < convergence_end_iter:
	
		before = time.time()
		sigma_1 = m_gibbs.sample_sigma_1_optimized(beta,gamma,a_sigma,b_sigma)
		pie = m_gibbs.sample_pie(gamma,pie_a,pie_b)
		sigma_e = m_gibbs.sample_sigma_e(y,H_beta,C_alpha,a_e,b_e)
		gamma = m_gibbs.sample_gamma_numba_optimized(y,C_alpha,H,beta,pie,sigma_1,sigma_e,gamma,H_beta)
		beta,H_beta = m_gibbs.sample_beta_numba_optimized(y,C_alpha,H,beta,gamma,sigma_1,sigma_e,H_beta)
		alpha,C_alpha = m_gibbs.sample_alpha(y,C,alpha,sigma_e,H_beta,C_alpha)
		genetic_var = np.var(H_beta)
		total_heritability = genetic_var / (genetic_var + sigma_e**2)
		#pheno_var = np.var(y - C_alpha)
		beta_p99 = np.percentile(np.absolute(beta), 99)
		#total_heritability = genetic_var / pheno_var
		alpha_norm = np.linalg.norm(alpha, ord=2)
		beta_norm = np.linalg.norm(beta, ord=2)

		after = time.time()	
		if verbose > 1:
			print(it,str(after - before),pie,sigma_1,sigma_e,np.sum(gamma),beta_p99,np.max(np.abs(beta)),total_heritability)

		if it >= burn_in_iter:
			trace[it-burn_in_iter,:] = [alpha_norm,beta_norm,sigma_1,sigma_e,beta_p99,total_heritability,np.sum(gamma)]
			top5_beta_trace[it-burn_in_iter,:] = np.sort(np.absolute(beta))[::-1][:5]

		## test for convergence using values in trace from multiple consecutive draws
			
		if it == convergence_end_iter - 1:
			convergence_scores = utility.convergence_geweke_test(trace,top5_beta_trace,convergence_start_iter-burn_in_iter,convergence_end_iter-burn_in_iter)

			if convergence_scores == 1:
				convergence_container[num] = 1
				if verbose > 0:
					print("convergence has been reached at %i iterations for chain %i. The MCMC Chain has entered a stationary stage" %(it,num))
					print("trace values:", trace[it-burn_in_iter,:])
				break
			else:
				trace_ = np.empty((1000,7))
				top5_beta_trace_ = np.empty((1000,5))
				trace = np.concatenate((trace[-(convergence_end_iter - burn_in_iter-1000):,:],trace_),axis=0)
				top5_beta_trace = np.concatenate((top5_beta_trace[-(convergence_end_iter - burn_in_iter-1000):,:],top5_beta_trace_),axis = 0)

				burn_in_iter += 1000
				convergence_start_iter += 1000
				convergence_end_iter += 1000

		it += 1
 	
		if it > 100000: 
			convergence_container[num] = 0
			break

	#print("convergence indicator:", convergence_container[num] )

	if convergence_container[num] == 1:

		#print("passed convergence test %i" %(num))
		## MCMC draws for posterior mean

		posterior_draws = 10000

		alpha_mean = np.zeros(C_c)
		beta_mean = np.zeros(H_c)
		gamma_sum = np.zeros(H_c)

		alpha_M2 = np.zeros(C_c)
		beta_M2 = np.zeros(H_c)

		posterior_trace = np.empty((posterior_draws,7))

		it = 0

		while it < posterior_draws:
		
			before = time.time()
			sigma_1 = m_gibbs.sample_sigma_1_optimized(beta,gamma,a_sigma,b_sigma)
			pie = m_gibbs.sample_pie(gamma,pie_a,pie_b)
			sigma_e = m_gibbs.sample_sigma_e(y,H_beta,C_alpha,a_e,b_e)
			gamma = m_gibbs.sample_gamma_numba_optimized(y,C_alpha,H,beta,pie,sigma_1,sigma_e,gamma,H_beta)
			beta,H_beta = m_gibbs.sample_beta_numba_optimized(y,C_alpha,H,beta,gamma,sigma_1,sigma_e,H_beta)
			alpha,C_alpha = m_gibbs.sample_alpha(y,C,alpha,sigma_e,H_beta,C_alpha)
			genetic_var = np.var(H_beta)
			total_heritability = genetic_var / (genetic_var + sigma_e**2)
			#pheno_var = np.var(y - C_alpha)
			#large_beta_ratio = np.sum(np.absolute(beta) > 0.3) / len(beta)
			beta_p99 = np.percentile(np.absolute(beta), 99)
			#total_heritability = genetic_var / pheno_var
			alpha_norm = np.linalg.norm(alpha, ord=2)
			beta_norm = np.linalg.norm(beta, ord=2)

			after = time.time()

			if verbose > 1:
				print(it,str(after - before),pie,sigma_1,sigma_e,np.sum(gamma),beta_p99,np.max(np.abs(beta)),total_heritability)

			posterior_trace[it,:] = [alpha_norm,beta_norm,sigma_1,sigma_e,beta_p99,total_heritability,np.sum(gamma)]
			beta_mean,beta_M2 = utility.welford(beta_mean,beta_M2,beta,it)
			alpha_mean,alpha_M2 = utility.welford(alpha_mean,alpha_M2,alpha,it)
			gamma_sum += gamma

			if verbose > 0:
				if it > 0 and it % 2000 == 0:
					print("Posterior draws: %i iterations have been sampled for chain %i" %(it,num), str(after - before),posterior_trace[it,:])

			it += 1

		trace_container[num] = posterior_trace

		#alpha values
		alpha_container[num] = {'avg': alpha_mean,
								'M2': alpha_M2}

		#beta values
		beta_container[num] = {'avg':beta_mean,
								'M2':beta_M2}

		gamma_container[num] = gamma_sum / posterior_draws

	else:

		#print("failed convergence test %i" %(num))
		trace_container[num] = []

		#alpha values
		alpha_container[num] = {'avg': [],
								'M2': []}

		#beta values
		beta_container[num] = {'avg':[],
								'M2':[]}

		gamma_container[num] = []



