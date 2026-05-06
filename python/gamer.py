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
	parser.add_argument('-m',type = int, action = 'store', dest = 'model',default = 1,help="1: multiplicative;	2: additive")
	parser.add_argument('-r', action='store_true', dest='recode', help = "recode the alternative allele to be phenotype-increasing alleles")
	parser.add_argument('-n',type = int, action = 'store', default = 8, dest = "num", help = 'number of MCMC chains run parallelly')
	parser.add_argument('-v',type = int, action = 'store', dest = 'verbose',default = 0, help = "verbose levels 0: no stdout; 1: convergence and minimal stdout; 2: per MCMC iteration stdout")
	parser.add_argument('-mb',type = float, action = 'store', default = 10, dest = "multi_pi_b", help = 'tuning parameter for pi_b in the multiplicative model')
	parser.add_argument('-ab',type = float, action = 'store', default = 1, dest = "add_pi_b", help = 'tuning parameter for pi_b in the additive model')
	parser.add_argument('-o',type = str, action = 'store', dest = 'output',help = "the prefix of output files")
	args = parser.parse_args()

	chromosome, ID, position, X = utility.vcf_processing(args.vcf)
	#X = np.asfortranarray(X)
	n = X.shape[0]

	y = []
	with open(args.pheno) as f:
		for line in f:
			line = line.strip("\n")
			y.append(float(line))

	y = np.asarray(y)
	y_mu = y.mean()
	y_std = y.std()
	if args.model == 1:
		# Multiplicative: only center. Dividing by std breaks the product structure
		# because prod(1+X*beta)/std is not of the form prod(1+X*beta_new).
		y = y - y_mu
	elif args.model == 2:
		# Additive: full standardization is fine; X*beta/std = X*(beta/std).
		y = (y - y_mu) / y_std

	if args.covar is None:
		C = np.ones(n)
		C = C.reshape(n, 1)
	else:
		C =  np.array(pd.read_csv(args.covar,sep="\t",header=None))
		has_intercept = np.any(np.all(np.isclose(C, 1.0, rtol=0.0, atol=1e-8), axis=0))
		if not has_intercept:
			C = np.column_stack([np.ones(n, dtype=float), C])
	print(C[:5,:])
	
	print(args.recode)
	if args.recode:
		X, flipped_snps = utility.recode_genotype(X,y,C)

	if args.model == 1:
		prefix = args.output + "_multiplicative_"
	elif args.model == 2:
		prefix = args.output + "_additive_"
	else:
		raise SystemError("please specificy the model type")

	# Persist recode decisions for traceability when -r is used.
	if args.recode:
		n_flipped = int(flipped_snps.sum())
		print("Recoded %i/%i SNPs (alt allele was phenotype-decreasing in original coding)"
		      % (n_flipped, len(flipped_snps)))
		with open(prefix + "flipped_snps.txt", "w") as OUTPUT_FLIPPED:
			print("chromosome\tposition\tID\tflipped", file=OUTPUT_FLIPPED)
			for i in range(len(flipped_snps)):
				print("%s\t%i\t%s\t%i" % (chromosome[i], position[i], ID[i], int(flipped_snps[i])),
				      file=OUTPUT_FLIPPED)

	trace_container = mp.Manager().dict()
	gamma_container = mp.Manager().dict()
	beta_container = mp.Manager().dict()
	alpha_container = mp.Manager().dict()
	convergence_container = mp.Manager().dict()

	print("multiprocessing method:",mp.get_start_method())

	processes = []

	if args.model == 1:
		for num in range(args.num):
			p = mp.Process(target = multi.sampling, args=(args.verbose,y,C,X,args.output,num,trace_container,gamma_container,beta_container,alpha_container,convergence_container,args.multi_pi_b))
			processes.append(p)
			p.start()
	elif args.model == 2:
		for num in range(args.num):
			p = mp.Process(target = add.sampling, args=(args.verbose,y,C,X,args.output,num,trace_container,gamma_container,beta_container,alpha_container,convergence_container,args.add_pi_b))
			processes.append(p)
			p.start()
		
	else:
		raise SystemError("please specificy the model type")

	for process in processes:
		process.join()


	convergence_all_chains = []
	alpha_posterior_all_chains = []
	alpha_posterior_sd_all_chains = []
	beta_posterior_all_chains = []
	beta_posterior_sd_all_chains = []
	gamma_all_chains = []
	trace_posterior_all_chains = []

	column_names_list = ["y_info","alpha_norm_2", "beta_norm_2", "sigma_1", "sigma_e", "beta_p99", "total_heritability", "sum_gamma"]


	for num in range(args.num):
		convergence_all_chains.append(convergence_container[num])

	print("%i/%i chains have reached the convergence." %(np.sum(convergence_all_chains),len(convergence_all_chains)))

	if np.sum(convergence_all_chains) > 0:

		for num in range(args.num):
			if convergence_all_chains[num] == 1:
				alpha_posterior_all_chains.append(alpha_container[num]["avg"])
				alpha_posterior_sd_all_chains.append(alpha_container[num]["M2"])
				beta_posterior_all_chains.append(beta_container[num]["avg"])
				beta_posterior_sd_all_chains.append(beta_container[num]["M2"])
				trace_posterior_all_chains.append(trace_container[num])
				gamma_all_chains.append(gamma_container[num])

		trace_posterior_all_chains = np.vstack(trace_posterior_all_chains)
		print(trace_posterior_all_chains.shape)
		trace_posterior = np.mean(trace_posterior_all_chains,axis=0)
		trace_posterior_sd = np.std(trace_posterior_all_chains,axis=0)
		trace_posterior = np.insert(trace_posterior,0,y_mu)
		trace_posterior_sd = np.insert(trace_posterior_sd,0,y_std)

	
		pip = np.mean(gamma_all_chains,axis=0)

		beta_posterior = np.zeros(X.shape[1])
		beta_posterior_M2 = np.zeros(X.shape[1])
		alpha_posterior = np.zeros(C.shape[1])
		alpha_posterior_M2 = np.zeros(C.shape[1])
			
		N_beta=0
		N_alpha=0


		for num in range(args.num):
			if convergence_all_chains[num] == 1:
				beta_posterior,beta_posterior_M2,N_beta = utility.merge_welford(beta_posterior,beta_posterior_M2,N_beta,beta_container[num]["avg"],beta_container[num]["M2"],10000)
				alpha_posterior,alpha_posterior_M2,N_alpha = utility.merge_welford(alpha_posterior,alpha_posterior_M2,N_alpha,alpha_container[num]["avg"],alpha_container[num]["M2"],10000)


		beta_posterior_sd = np.sqrt(beta_posterior_M2/(N_beta-1))
		alpha_posterior_sd = np.sqrt(alpha_posterior_M2/(N_alpha-1))
		np.savetxt(prefix+"model_trace.txt",trace_posterior_all_chains,delimiter="\t",header="\t".join(column_names_list))


		with open(prefix + "param.txt", "w") as OUTPUT_TRACE:
			for i in range(len(trace_posterior)):
				print("%s\t%f\t%f" %(column_names_list[i],trace_posterior[i],trace_posterior_sd[i]),file = OUTPUT_TRACE)
				
		with open(prefix+"alpha.txt","w") as OUTPUT_ALPHA:
			for i in range(len(alpha_posterior)):
				print("%f\t%f" %(alpha_posterior[i],alpha_posterior_sd[i]),file = OUTPUT_ALPHA)

		with open(prefix+"beta.txt","w") as OUTPUT_BETA:
			print("chromosome\tposition\tID\tbeta_mean\tbeta_sd\tpip",file = OUTPUT_BETA)
			for i in range(len(beta_posterior)):
				print("%s\t%i\t%s\t%f\t%f\t%f" %(chromosome[i],position[i],ID[i],beta_posterior[i],beta_posterior_sd[i],pip[i]),file = OUTPUT_BETA)

	else:
		with open(prefix + "param.txt", "w") as OUTPUT_TRACE:
			for i in range(len(column_names_list)):
				print("%s\t%s\t%s" %(column_names_list[i],"NA","NA"),file = OUTPUT_TRACE)
				
		with open(prefix+"alpha.txt","w") as OUTPUT_ALPHA:
			for i in range(C.shape[1]):
				print("%s\t%s" %("NA","NA"),file = OUTPUT_ALPHA)

		with open(prefix+"beta.txt","w") as OUTPUT_BETA:
			print("chromosome\tposition\tID\tbeta_mean\tbeta_sd\tpip",file = OUTPUT_BETA)
			for i in range(X.shape[1]):
				print("%s\t%i\t%s\t%s\t%s\t%s" %(chromosome[i],position[i],ID[i],"NA","NA","NA"),file = OUTPUT_BETA)

if __name__ == "__main__":
	main()
	
