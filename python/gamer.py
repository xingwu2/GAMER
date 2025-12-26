import time
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import re
import multiprocessing as mp
import time

import multiplicative_gibbs as multi
import additive_gibbs as add

parser = argparse.ArgumentParser()
parser.add_argument('-g',type = str, action = 'store', dest = 'geno')
parser.add_argument('-c',type = str, action = 'store', dest = 'covar')
parser.add_argument('-y',type = str, action = 'store', dest = 'pheno')
parser.add_argument('-m',type = int, action = 'store', dest = 'mode',default = "1",help="1: multiplicative;	2: additive")
parser.add_argument('-n',type = int, action = 'store', default = 5, dest = "num", help = 'number of MCMC chains run parallelly')
parser.add_argument('-v',action = 'store_true', dest = 'verbose',default = False, help = "print out each MCMC iteration")
parser.add_argument('-b',type = float, action = 'store', default = 10, dest = "pi_b", help = 'tuning parameter for pi_b')
parser.add_argument('-o',type = str, action = 'store', dest = 'output',help = "the prefix of output files")
args = parser.parse_args()


X = np.loadtxt(args.geno,delimiter="\t")
n = X.shape[0]

y = []
with open(args.pheno) as f:
	for line in f:
		line = line.strip("\n")
		y.append(float(line))

y = np.asarray(y)

if args.covar is None:
	C = np.ones(n)
	C = C.reshape(n, 1)
else:
	C =  np.array(pd.read_csv(args.covar,sep="\t",header=None)) 

if args.mode == 1:
	prefix = args.output + "_multiplicative_"
else:
	prefix = args.output + "_additive_"

if __name__ == '__main__':

	trace_container = mp.Manager().dict()
	gamma_container = mp.Manager().dict()
	beta_container = mp.Manager().dict()
	alpha_container = mp.Manager().dict()

	processes = []

	if args.mode == 1:
		for num in range(args.num):
			p = mp.Process(target = multi.sampling, args=(args.verbose,y,C,X,12000,args.output,num,trace_container,gamma_container,beta_container,alpha_container,args.pi_b))
			processes.append(p)
			p.start()
	else:
		for num in range(args.num):
			p = mp.Process(target = add.sampling, args=(args.verbose,y,C,X,12000,args.output,num,trace_container,gamma_container,beta_container,alpha_container,args.pi_b))
			processes.append(p)
			p.start()

	for process in processes:
		process.join()

	alpha_posterior = []
	alpha_posterior_sd = []
	beta_posterior = []
	beta_posterior_sd = []
	pip = []
	trace_posterior = []
	trace_posterior_sd = []

	for num in range(args.num):
		alpha_posterior.append(alpha_container[num]["avg"])
		alpha_posterior_sd.append(alpha_container[num]["sd"])
		beta_posterior.append(beta_container[num]["avg"])
		beta_posterior_sd.append(beta_container[num]["sd"])
		trace_posterior.append(trace_container[num]["avg"])
		trace_posterior_sd.append(trace_container[num]["sd"])
		pip.append(gamma_container[num]["pip"])
		
		# alpha_posterior.append(np.mean(alpha_container[num],axis=0))
		# alpha_posterior_sd.append(np.std(alpha_container[num],axis=0))
		# beta_posterior.append(np.mean(beta_container[num],axis=0))
		# beta_posterior_sd.append(np.std(beta_container[num],axis=0))
		# trace_posterior.append(np.mean(trace_container[num],axis=0))
		# trace_posterior_sd.append(np.std(trace_container[num],axis=0))
		# pip.append(np.mean(gamma_container[num],axis = 0))

	alpha_posterior_median = np.median(alpha_posterior,axis=0)
	alpha_posterior_sd_median = np.median(alpha_posterior_sd,axis=0)
	beta_posterior_median = np.median(beta_posterior,axis=0)
	beta_posterior_sd_median = np.median(beta_posterior_sd,axis=0)
	trace_posterior_median = np.median(trace_posterior,axis=0)
	trace_posterior_sd_median = np.median(trace_posterior_sd,axis=0)
	pip_median = np.median(pip,axis=0)

	OUTPUT_TRACE = open(prefix+"param.txt","w")
	for i in range(len(trace_posterior_median)):
		print("%f\t%f" %(trace_posterior_median[i],trace_posterior_sd_median[i]),file = OUTPUT_TRACE)
		
	OUTPUT_ALPHA = open(prefix+"alpha.txt","w")
	for i in range(len(alpha_posterior_median)):
		print("%f\t%f" %(alpha_posterior_median[i],alpha_posterior_sd_median[i]),file = OUTPUT_ALPHA)

	OUTPUT_BETA = open(prefix+"beta.txt","w")
	for i in range(len(beta_posterior_median)):
		print("%s\t%f\t%f" %(i,beta_posterior_median[i],beta_posterior_sd_median[i]),file = OUTPUT_BETA)

	OUTPUT_PIP = open(prefix+"pip.txt","w")
	for i in range(len(pip_median)):
		print("%s\t%f" %(i,pip_median[i]),file = OUTPUT_PIP)
	
