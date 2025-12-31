import numpy as np
import scipy as sp
import math
import pandas as pd 
import time
from numba import njit
import os
import gc

@njit
def rbernoulli(p):
	return 1 if np.random.random() < p else 0

def sample_pie(gamma,pie_a,pie_b):
	a_new = np.sum(gamma)+pie_a
	b_new = np.sum(1-gamma)+pie_b
	pie_new = np.random.beta(a_new,b_new)
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
	sigma_e_new = math.sqrt(1/sigma_e_neg2)
	return(sigma_e_new)

def sample_alpha(y,H_beta,C,alpha,sigma_e,C_alpha):

	r,c = C.shape
	if c == 1:
		new_variance = 1/(np.linalg.norm(C[:,0])**2*sigma_e**-2)
		new_mean = new_variance*np.dot((y-H_beta),C[:,0])*sigma_e**-2
		alpha = np.random.normal(new_mean,math.sqrt(new_variance))
		C_alpha = C[:,0] * alpha 
	else:
		for i in range(c):
			new_variance = 1/(np.sum(C[:,i]**2)*sigma_e**-2)
			C_alpha_negi = C_alpha - C[:,i] * alpha[i]
			new_mean = new_variance*np.dot(y-C_alpha_negi-H_beta,C[:,i])*sigma_e**-2
			alpha[i] = np.random.normal(new_mean,math.sqrt(new_variance))
			C_alpha = C_alpha_negi + C[:,i] * alpha[i]
	return(alpha,C_alpha)


@njit
def sample_gamma_numba(y,C_alpha,H,beta,pie,sigma_1,sigma_e,gamma,H_beta,H_norm_2):
	sigma_e_neg2 = 1 / (sigma_e * sigma_e)
	sigma_1_neg2 = 1 / (sigma_1 * sigma_1)
	sigma_1_sq = sigma_1 * sigma_1
	ncols = beta.shape[0]
	nrows = y.shape[0]

	residual = np.empty(nrows)
	for r in range(nrows):
		residual[r] = y[r] - C_alpha[r] - H_beta[r]

	for i in range(ncols):
		dot_val = 0
		for r in range(nrows):
			h = H[r,i]
			dot_val += (residual[r]+h*beta[i]) * h
		hnorm2 = H_norm_2[i]
		variance = 1.0 / (hnorm2*sigma_e_neg2 + sigma_1_neg2)
		mean = variance * sigma_e_neg2 * dot_val
		f = 1.0 / np.sqrt(hnorm2 * sigma_1_sq * sigma_e_neg2 + 1)
		A = f * np.exp(0.5*mean*mean/variance)
		gamma_0_pie = (1.0 - pie) / ((1.0-pie)+pie*A)
		gamma[i] = rbernoulli(1.0-gamma_0_pie)
	return(gamma)


def sample_gamma(y,C_alpha,H,beta,pie,sigma_1,sigma_e,gamma,H_beta):
	sigma_e_neg2 = sigma_e**-2
	sigma_1_neg2 = sigma_1**-2
	e = y - C_alpha -  H_beta 
	residual_negi = np.transpose(np.ones((len(beta),len(y))) * e)+ H * beta
	variance = 1/(np.sum(H**2,axis=0) * sigma_e_neg2+sigma_1_neg2)
	mean = variance * np.sum(residual_negi * H,axis=0) * sigma_e_neg2
	f = np.sqrt(1/(np.sum(H**2,axis=0) * sigma_1**2 * sigma_e_neg2 + 1))
	gamma_0_pie = (1-pie) / ((1-pie)+pie*(f*np.exp(0.5*mean**2/variance)))	
	gamma = np.random.binomial(1,1-gamma_0_pie)
	return(gamma)

@njit
def sample_beta_numba(y, C_alpha, H_beta, H, beta, gamma, sigma_1, sigma_e, H_norm_2):
	sigma_e_neg2 = sigma_e ** -2
	sigma_1_neg2 = sigma_1 ** -2
	ncols = beta.shape[0]
	nrows = y.shape[0]
    
	for i in range(ncols):

		beta_i_old = beta[i]

		if beta_i_old != 0.0:
			for r in range(nrows):
				H_beta[r] -= H[r, i] * beta[i]
				# h = H[r,i]
				# minus = h*beta_i_old
				# H_beta[r] -= minus

		# for r in range(nrows):
		# 	H_beta[r] -= H[r, i] * beta[i]

		if gamma[i] == 0:
			beta[i] = 0

		else:
	        # Compute the dot product over the column using the updated H_beta.
			dot_val = 0.0
			for r in range(nrows):
	            # residual = y[r] - C_alpha[r] - H_beta[r]
				res_val = y[r] - C_alpha[r] - H_beta[r] 
				dot_val += res_val * H[r, i]
	        
			new_variance = 1.0 / (H_norm_2[i]*sigma_e_neg2 + sigma_1_neg2)
			new_mean = new_variance * sigma_e_neg2 * dot_val
	        
	        # Sample new beta using standard normal (Numba supports np.random.randn)
			beta[i] = new_mean + math.sqrt(new_variance) * np.random.randn()

			if abs(beta[i]) < 0.05:
				beta[i] = 0.0
			else:
				for r in range(nrows):
					H_beta[r] += H[r, i] * beta[i]
	return (beta, H_beta)

def sample_beta(y,C_alpha,H,beta,gamma,sigma_1,sigma_e,H_beta):

	sigma_e_neg2 = sigma_e**-2
	sigma_1_neg2 = sigma_1**-2	
	for j in range(len(beta)):
		if gamma[j] == 0:
			H_beta = H_beta - H[:,j] * beta[j]
			beta[j] = 0
		else:
			H_beta_negj = H_beta - H[:,j] * beta[j]
			new_variance = 1/(np.sum(H[:,j]**2)*sigma_e_neg2+sigma_1_neg2)
			residual = y - C_alpha -  H_beta + H[:,j] * beta[j]
			new_mean = new_variance*np.dot(residual,H[:,j])*sigma_e_neg2
			beta[j] = np.random.normal(new_mean,math.sqrt(new_variance))
			if abs(beta[j]) < 0.05:
				beta[j] = 0
			H_beta = H_beta_negj + H[:,j] * beta[j]
	return(beta,H_beta)




