import time
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import re
import multiprocessing as mp
import time

import multiplicative_sampling as multi
import additive_sampling as add
import utility

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-g',type = str, action = 'store', dest = 'vcf',help = "vcf.gz file (bgzipped)")
	parser.add_argument('-c',type = str, action = 'store', dest = 'covar')
	parser.add_argument('-y',type = str, action = 'store', dest = 'pheno')
	parser.add_argument('-m',type = int, action = 'store', dest = 'mode',default = "1",help="1: multiplicative;	2: additive")
	parser.add_argument('-n',type = int, action = 'store', default = 8, dest = "num", help = 'number of MCMC chains run parallelly')
	parser.add_argument('-v',type = int, action = 'store', dest = 'verbose',default = 0, help = "verbose levels 0: no stdout; 1: convergence and minimal stdout; 2: per MCMC iteration stdout")
	parser.add_argument('-mb',type = float, action = 'store', default = 10, dest = "multi_pi_b", help = 'tuning parameter for pi_b in the multiplicative model')
	parser.add_argument('-ab',type = float, action = 'store', default = 1, dest = "add_pi_b", help = 'tuning parameter for pi_b in the additive model')
	parser.add_argument('-o',type = str, action = 'store', dest = 'output',help = "the prefix of output files")
	args = parser.parse_args()


	X = np.loadtxt(args.vcf,delimiter="\t")


	#chromosome, ID, position, X = utility.vcf_processing(args.vcf)
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

	trace_container = mp.Manager().dict()
	gamma_container = mp.Manager().dict()
	beta_container = mp.Manager().dict()
	alpha_container = mp.Manager().dict()

	processes = []

	if args.mode == 1:
		for num in range(args.num):
			p = mp.Process(target = multi.sampling, args=(args.verbose,y,C,X,args.output,num,trace_container,gamma_container,beta_container,alpha_container,args.multi_pi_b))
			processes.append(p)
			p.start()
	else:
		for num in range(args.num):
			p = mp.Process(target = add.sampling, args=(args.verbose,y,C,X,args.output,num,trace_container,gamma_container,beta_container,alpha_container,args.add_pi_b))
			processes.append(p)
			p.start()

	for process in processes:
			process.join()

	alpha_posterior_all_chains = []
	alpha_posterior_sd_all_chains = []
	beta_posterior_all_chains = []
	beta_posterior_sd_all_chains = []
	gamma_all_chains = []
	trace_posterior_all_chains = []

	for num in range(args.num):
		alpha_posterior_all_chains.append(alpha_container[num]["avg"])
		alpha_posterior_sd_all_chains.append(alpha_container[num]["M2"])
		beta_posterior_all_chains.append(beta_container[num]["avg"])
		beta_posterior_sd_all_chains.append(beta_container[num]["M2"])
		trace_posterior_all_chains.append(trace_container[num])
		gamma_all_chains.append(gamma_container[num])

	trace_posterior_all_chains = np.vstack(trace_posterior_all_chains)
	trace_posterior = np.mean(trace_posterior_all_chains,axis=0)
	trace_posterior_sd = np.std(trace_posterior_all_chains,axis=0)
		

	pip = np.mean(gamma_all_chains,axis=0)

	beta_posterior = []
	beta_posterior_M2 = []
	alpha_posterior = []
	alpha_posterior_M2 = []
		
	N_beta=0
	N_alpha=0


	for num in range(args.num):
		beta_posterior,beta_posterior_M2,N_beta = utility.merge_welford(beta_posterior,beta_posterior_M2,N_beta,beta_container[num]["avg"],beta_container[num]["M2"],10000)
		alpha_posterior,alpha_posterior_M2,N_alpha = utility.merge_welford(alpha_posterior,alpha_posterior_M2,N_alpha,alpha_container[num]["avg"],alpha_container[num]["M2"],10000)


	beta_posterior_sd = np.sqrt(beta_posterior_M2/(N_beta-1))
	alpha_posterior_sd = np.sqrt(alpha_posterior_M2/(N_alpha-1))

	np.savetxt(prefix+"_model_trace.txt",trace_posterior_all_chains,delimiter="\t")


	OUTPUT_TRACE = open(prefix+"param.txt","w")
	for i in range(len(trace_posterior)):
		print("%f\t%f" %(trace_posterior[i],trace_posterior_sd[i]),file = OUTPUT_TRACE)
			
	OUTPUT_ALPHA = open(prefix+"alpha.txt","w")
	for i in range(len(alpha_posterior)):
		print("%f\t%f" %(alpha_posterior[i],alpha_posterior_sd[i]),file = OUTPUT_ALPHA)

	OUTPUT_BETA = open(prefix+"beta.txt","w")
	for i in range(len(beta_posterior)):
		print("%s\t%f\t%f" %(i,beta_posterior[i],beta_posterior_sd[i]),file = OUTPUT_BETA)

	OUTPUT_PIP = open(prefix+"pip.txt","w")
	for i in range(len(pip)):
		print("%s\t%f" %(i,pip[i]),file = OUTPUT_PIP)

if __name__ == "__main__":
	main()
	
