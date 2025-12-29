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
