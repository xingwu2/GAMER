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

def circle_product_matrix(X, beta):
	non0_index = np.where(beta != 0)[0]
	# Element-wise multiplication of each row of X with beta.
	x_beta = np.multiply(X[:,non0_index], beta[non0_index])
	# Compute the product along the second axis (columns) to get the product for each row.
	y = np.prod(x_beta + 1, axis=1)
	return(y)

@njit
def rbernoulli(p):
	return 1 if np.random.random() < p else 0

def circle_prodcut_vector(x,beta):
	y = x * beta + 1
	return(y)

def sample_pie(gamma,pie_a,pie_b):
	a_new = np.sum(gamma)+pie_a
	b_new = np.sum(1-gamma)+pie_b
	draw_a = np.random.gamma(a_new,1.0)
	draw_b = np.random.gamma(b_new,1.0)
	pie_new = draw_a / (draw_a + draw_b)

	#pie_new = np.random.beta(a_new,b_new)
	return(pie_new)

def sample_sigma_1(beta,gamma,a_sigma,b_sigma):
	a_new = 0.5*np.sum(gamma)+a_sigma
	b_new = 0.5*np.sum(np.multiply(np.square(beta),gamma))+b_sigma
	sigma_1_neg2 =np.random.gamma(a_new,1.0/b_new)
	sigma_1_new = math.sqrt(1/sigma_1_neg2)
	return(sigma_1_new)

def sample_sigma_e(y,H_beta,C_alpha,a_e,b_e):
	n = len(y)
	a_new = float(n)/2+a_e
	resid = y - H_beta - C_alpha
	b_new = np.sum(np.square(resid))/2+b_e
	sigma_e_neg2 =np.random.gamma(a_new,1.0/b_new)
	sigma_e_new = np.sqrt(1/sigma_e_neg2)
	return(sigma_e_new)


def sample_alpha(y,C,alpha,sigma_e,H_beta,C_alpha):
	r,c = C.shape
	if c == 1:
		new_variance = 1/(np.sum(C[:,0]**2) * sigma_e**-2)
		new_mean = new_variance*np.dot((y-H_beta),C[:,0])*sigma_e**-2
		alpha = np.random.normal(new_mean,math.sqrt(new_variance))
		C_alpha = C[:,0] * alpha
	else:
		for i in range(c):
			new_variance = 1/(np.sum(C[:,i]**2) * sigma_e**-2)
			C_alpha_negi = C_alpha - C[:,i] * alpha[i]
			new_mean = new_variance*np.dot(y-C_alpha_negi-H_beta,C[:,i])*sigma_e**-2
			alpha[i] = np.random.normal(new_mean,math.sqrt(new_variance))
			C_alpha = C_alpha_negi + C[:,i] * alpha[i]
	return(alpha,C_alpha)


@njit
def sample_gamma_numba(y,C_alpha,H,beta,pie,sigma_1,sigma_e,gamma,H_beta):
	sigma_e_neg2 = sigma_e**-2
	sigma_1_neg2 = sigma_1**-2
	sigma_1_sq = sigma_1**2
	ncols = beta.shape[0]
	nrows = y.shape[0]

	for i in range(ncols):
		H_beta_neg_H_norm2 = 0.0
		dot_val = 0.0
		for r in range(nrows):
			H_beta_neg = H_beta[r] / (1+H[r, i] * beta[i])
			H_beta_neg_H_norm2 += (H_beta_neg * H[r,i])**2
			res_val = y[r] - C_alpha[r] - H_beta_neg 
			dot_val += res_val * H_beta_neg * H[r, i]

		f = 1.0 / np.sqrt(H_beta_neg_H_norm2 * sigma_1_sq * sigma_e_neg2 + 1)
		variance = 1.0/ (H_beta_neg_H_norm2 * sigma_e_neg2+sigma_1_neg2)

		mean = variance * dot_val * sigma_e_neg2
		A = f * np.exp(0.5*mean**2/variance)
		gamma_0_pie = (1.0 - pie) / ((1.0-pie)+pie*A)
		gamma[i] = rbernoulli(1.0-gamma_0_pie)
	return(gamma)

def sample_gamma(y,C_alpha,H,beta,pie,sigma_1,sigma_e,gamma,H_beta):
	sigma_e_neg2 = sigma_e**-2
	sigma_1_neg2 = sigma_1**-2
	H_beta_neg = H_beta[:,None] / (H*beta + 1)
	#H_beta_neg = np.transpose(np.ones((len(beta),len(y))) * H_beta) / (H*beta + 1)
	H_beta_neg_H =  np.multiply(H ,H_beta_neg)
	variance = 1/(np.sum(H_beta_neg_H**2,axis=0)*sigma_e_neg2+sigma_1_neg2)
	f = np.sqrt(1/(np.sum(H_beta_neg_H**2,axis=0) * sigma_1**2 * sigma_e_neg2 + 1))
	gamma_0_pie = np.zeros(len(beta))
	for i in range(len(beta)):
		residual = y - C_alpha - H_beta_neg[:,i]
		mean = variance[i]*np.dot(residual,H_beta_neg_H[:,i])*sigma_e_neg2
		A = f[i] * np.exp(0.5*mean**2/variance[i])
		gamma_0_pie[i] = (1-pie) / ((1-pie)+pie*A)
	gamma = np.random.binomial(1,1-gamma_0_pie)
	return(gamma)

@njit
def sample_beta_numba(y,C_alpha,H,beta,gamma,sigma_1,sigma_e,H_beta):

	sigma_e_neg2 = 1 / (sigma_e *sigma_e)
	sigma_1_neg2 = 1 / (sigma_1 * sigma_1)
	sigma_1_sq = sigma_1 * sigma_1
	ncols = beta.shape[0]
	nrows = y.shape[0]

	# precompute y - C_alpha once
	y_minus_C = np.empty(nrows)
	for r in range(nrows):
		y_minus_C[r] = y[r] - C_alpha[r]

	for i in range(ncols):

		beta_i_old = beta[i]

		if beta_i_old != 0.0:
			for r in range(nrows):
				h = H[r,i]
				denom = 1.0 + h*beta_i_old
				H_beta[r] /= denom

		

		if gamma[i] ==0:
			beta[i] = 0.0
			continue
		else:
			dot_val = 0.0
			H_beta_neg_H_norm2 = 0.0

			for r in range(nrows):
				hb = H_beta[r]
				h = H[r,i]
				hb_h = hb * h
				H_beta_neg_H_norm2 += hb_h * hb_h
				res_val = y_minus_C[r] - hb
				dot_val += res_val * hb_h
			
			variance = 1.0/ (H_beta_neg_H_norm2 * sigma_e_neg2+sigma_1_neg2)
			mean = variance * dot_val * sigma_e_neg2
			beta[i] = mean + math.sqrt(variance) * np.random.randn()
			if abs(beta[i]) < 0.05:
				beta[i] = 0.0
			else:
				for r in range(nrows):
					H_beta[r] *= (1+H[r, i] * beta[i])
	return(beta,H_beta)


def sample_beta(y,C_alpha,H,beta,gamma,sigma_1,sigma_e,H_beta):

	sigma_e_neg2 = sigma_e**-2
	sigma_1_neg2 = sigma_1**-2

	for i in range(len(beta)):
		if gamma[i] == 0:
			H_beta = np.divide(H_beta,circle_prodcut_vector(H[:,i], beta[i]))
			beta[i] = 0
		else:
			H_beta_negi = np.divide(H_beta,circle_prodcut_vector(H[:,i], beta[i]))
			new_variance = 1/(np.sum((H_beta_negi*H[:,i])**2) * sigma_e_neg2+sigma_1_neg2)
			residual = y - C_alpha -  H_beta_negi
			new_mean = new_variance*np.dot(H_beta_negi*H[:,i],residual)*sigma_e_neg2
			beta[i] = np.random.normal(new_mean,math.sqrt(new_variance))
			if abs(beta[i]) < 0.05:
				beta[i] = 0
				H_beta = H_beta_negi
			else:
				H_beta = H_beta_negi * circle_prodcut_vector(H[:,i], beta[i])
	return(beta,H_beta)


def sampling(verbose,y,C,HapDM,iters,prefix,num,trace_container,gamma_container,beta_container,alpha_container):

	## set random seed for the process
	np.random.seed(int(time.time()) + os.getpid())

	#initiate beta,gamma and H matrix
	C_r, C_c = C.shape

	H = np.array(HapDM)
	H = np.asfortranarray(H)


	H_r,H_c = H.shape

	##specify hyper parameters
	pie_a = 1
	## assume 10 causal loci 
	pie_b = H_c / 10
	
	if pie_b < 10000:
		pie_b = 10000
	
	a_sigma = 2
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
	trace = np.empty((iters-2000,5))
	alpha_trace = np.empty((iters-2000,C_c))
	gamma_trace = np.empty((iters-2000,H_c))
	beta_trace = np.empty((iters-2000,H_c))
	top5_beta_trace = np.empty((iters-2000,5))

	alpha = np.random.random(size = C_c)
	gamma = np.random.binomial(1,pie,H_c)
	#beta = np.array(np.zeros(H_c))
	beta = np.array(np.zeros(H_c))
	
	for i in range(H_c):
		if gamma[i] == 0:
			beta[i] = 0
		else:
			beta[i] = np.random.normal(0,sigma_1) 

	## Pre-compute the H_beta and C_alpha in the beginning, and update them later in the MCMC
	H_beta = circle_product_matrix(H,beta)
	C_alpha = np.matmul(C,alpha)

	C_norm_2 = np.sum(C**2,axis=0)
	H_norm_2 = np.sum(H**2,axis=0)

	#start sampling

	while it < iters:
	
		before = time.time()
		sigma_1 = sample_sigma_1(beta,gamma,a_sigma,b_sigma)
		pie = sample_pie(gamma,pie_a,pie_b)
		sigma_e = sample_sigma_e(y,H_beta,C_alpha,a_e,b_e)
		gamma = sample_gamma_numba(y,C_alpha,H,beta,pie,sigma_1,sigma_e,gamma,H_beta)
		#gamma = sample_gamma(y,C_alpha,H,beta,pie,sigma_1,sigma_e,gamma,H_beta)
		alpha,C_alpha = sample_alpha(y,C,alpha,sigma_e,H_beta,C_alpha)
		#beta,H_beta = sample_beta(y,C_alpha,H,beta,gamma,sigma_1,sigma_e,H_beta)
		beta,H_beta = sample_beta_numba(y,C_alpha,H,beta,gamma,sigma_1,sigma_e,H_beta)
		genetic_var = np.var(H_beta)
		pheno_var = np.var(y - C_alpha)
		large_beta_ratio = np.sum(np.absolute(beta) > 0.3) / len(beta)
		total_heritability = genetic_var / pheno_var
		after = time.time()
		if it > 5000 and (total_heritability > 1):
			print("unrealistic beta sample",it,genetic_var,pheno_var,total_heritability)
			continue

		else:
			if verbose:
				print(it,str(after - before),pie,sigma_1,sigma_e,sum(gamma),large_beta_ratio,total_heritability)

			if it >= burn_in_iter:
				trace[it-burn_in_iter,:] = [sigma_1,sigma_e,large_beta_ratio,total_heritability,sum(gamma)]
				gamma_trace[it-burn_in_iter,:] = gamma
				beta_trace[it-burn_in_iter,:] = beta
				alpha_trace[it-burn_in_iter,:] = alpha
				top5_beta_trace[it-burn_in_iter,:] = np.sort(np.absolute(beta))[::-1][:5]

			if it >= burn_in_iter + 9999: # after burn-in iterations, test convergence

				max_z = []
				
				for a in range(C_c):
					after_burnin_alpha = alpha_trace[:,a]
					alpha_zscores = geweke.geweke(after_burnin_alpha)[:,1]
					max_z.append(np.amax(np.absolute(alpha_zscores)))

				for b in range(5):
					after_burnin_beta = top5_beta_trace[:,b]
					beta_zscores = geweke.geweke(after_burnin_beta)[:,1]
					max_z.append(np.amax(np.absolute(beta_zscores)))

				#convergence for large beta ratio
				after_burnin_pie = trace[:,2]
				pie_zscores = geweke.geweke(after_burnin_pie)[:,1]
				max_z.append(np.amax(np.absolute(pie_zscores)))

				#convergence for total_heritability
				after_burnin_var = trace[:,3]
				var_zscores = geweke.geweke(after_burnin_var)[:,1]
				max_z.append(np.amax(np.absolute(var_zscores)))

				# #convergence for sigma_1
				# after_burnin_sigma1 = trace[:,0]
				# sigma1_zscores = geweke.geweke(after_burnin_sigma1)[:,1]
				# max_z.append(np.amax(np.absolute(sigma1_zscores)))

				#convergence for sigma_e
				after_burnin_sigmae = trace[:,1]
				sigmae_zscores = geweke.geweke(after_burnin_sigmae)[:,1]
				max_z.append(np.amax(np.absolute(sigmae_zscores)))
				
				if  np.amax(max_z) < 1.5:
					print("convergence has been reached at %i iterations." %(it))
					break

				else:
					trace_ = np.empty((1000,5))
					gamma_trace_ = np.empty((1000,H_c))
					beta_trace_ = np.empty((1000,H_c))
					alpha_trace_ = np.empty((1000,C_c))
					top5_beta_trace_ = np.empty((1000,5))

					trace = np.concatenate((trace[-(iters - burn_in_iter-1000):,:],trace_),axis=0)
					gamma_trace = np.concatenate((gamma_trace[-(iters - burn_in_iter-1000):,:],gamma_trace_),axis=0)
					beta_trace = np.concatenate((beta_trace[-(iters - burn_in_iter-1000):,:],beta_trace_),axis=0)
					alpha_trace = np.concatenate((alpha_trace[-(iters - burn_in_iter-1000):,:],alpha_trace_),axis=0)
					top5_beta_trace = np.concatenate((top5_beta_trace[-(iters - burn_in_iter-1000):,:],top5_beta_trace_),axis = 0)

					burn_in_iter += 1000
					iters += 1000

			if (it - burn_in_iter) >= 0 and (it - burn_in_iter ) % 1000 == 0:
				print("%i iterations have sampled" %(it), str(after - before),trace[it-burn_in_iter,:])

			it += 1
	
	# trace values
	trace_container[num] = {'avg': np.mean(trace,axis=0),
							'sd' : np.std(trace,axis=0)}
	del trace
	gc.collect()

	#print("trace_container[num]",num,trace_container[num]['avg'])
	#alpha values
	alpha_container[num] = {'avg': np.mean(alpha_trace,axis=0),
							'sd': np.std(alpha_trace,axis=0)}
	del alpha_trace
	gc.collect()
	#print("alpha_container[num]",num,alpha_container[num]['avg'])
	#beta values

	beta_container[num] = {'avg':np.mean(beta_trace,axis=0),
							'sd':np.std(beta_trace,axis=0)}
	del beta_trace
	gc.collect()

	gamma_container[num] = {'pip':np.mean(gamma_trace,axis=0)}
	del gamma_trace
	gc.collect()

	# trace_container[num] = trace
	# alpha_container[num] = alpha_trace
	# beta_container[num] = beta_trace
	# gamma_container[num] = gamma_trace


