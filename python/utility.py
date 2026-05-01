import numpy as np
from cyvcf2 import VCF
import geweke


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

    if np.amax(max_z) < 2:
        return(1)
    else: 
        return(0)

def welford(mean,M2,x,it):
    n = it + 1

    delta = x - mean
    mean += delta / n
    delta2 = x - mean
    M2 += delta * delta2

    return(mean,M2)

def merge_welford(A_mean, A_M2,A_n,B_mean,B_M2,B_n):

    if A_n == 0:
        return(B_mean,B_M2,B_n)

    if B_n == 0:
        return(A_mean,A_M2,A_n)
    
    n_new = A_n + B_n
    delta = B_mean - A_mean

    mean = A_mean + delta * (B_n / n_new)
    M2 = A_M2 + B_M2 + (delta * delta) * (A_n * B_n / n_new)

    return(mean,M2,n_new)

# def vcf_processing(vcf):
#     ID = []
#     chromosome =[]
#     position = []
#     gt = []

#     for v in VCF(vcf):
#         ID.append(v.ID)
#         chromosome.append(v.CHROM)
#         position.append(v.POS)
#         g = np.array(v.genotypes)[:,:2]
#         dosage = np.sum(g,axis=1)
#         gt.append(dosage)

#     X = np.asfortranarray(np.vstack(gt).T, dtype=np.uint8)

#     print("finished loading %i variants and %i individuals" %(X.shape[1],X.shape[0]))
    
#     return(chromosome, ID, position, X )

def vcf_processing(vcf):
    # First pass: count variants and collect metadata
    vcf_reader = VCF(vcf)
    n_samples = len(vcf_reader.samples)
    
    IDs = []
    chromosomes = []
    positions = []
    for v in vcf_reader:
        IDs.append(v.ID)
        chromosomes.append(v.CHROM)
        positions.append(v.POS)
    
    n_variants = len(IDs)
    
    # Preallocate in Fortran order
    X = np.zeros((n_samples, n_variants), dtype=np.uint8, order='F')
    
    # Second pass: fill genotypes
    vcf_reader = VCF(vcf)
    for i, v in enumerate(vcf_reader):
        g = np.array(v.genotypes, dtype=np.int8)[:, :2]
        
        if np.any(g < 0):
            raise ValueError(f"Missing genotype at variant {IDs[i]} (chr{chromosomes[i]}:{positions[i]}), index {i}. Impute or filter before running.")
        
        dosage = g[:, 0] + g[:, 1]
        X[:, i] = dosage
    
    print("finished loading %i variants and %i individuals" 
          % (X.shape[1], X.shape[0]))
    
    return (chromosomes, IDs, positions, X)

def col_norm2_chunked(H, chunk_rows=2000, out_dtype=np.float64):
    n, p = H.shape
    out = np.zeros(p, dtype=out_dtype)

    for s in range(0, n, chunk_rows):
        block = H[s:s+chunk_rows, :].astype(np.float32, copy=False)
        # sum of squares per column for this block
        out += np.einsum('ij,ij->j', block, block)

    return(out)



