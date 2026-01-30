from sampling import frange
from scipy.sparse import csr_matrix
import numpy as np

def build_basis_idx(coords, h_max, k_max, nm_min, n_max, m_max, ij_max, total_max, ab_max = 0, nrDeltas=1):

    X = None
    rows = []

    if coords == "rij":
    # (rA1^n*rB1^m*rA2^i*rB2^j*r12^k*rAB^h)/sqrt(rA1^2+rA2^2)^a/sqrt(rB1^2+rB2^2)^b*exp(-alpha*(rA1+rA2+rB1+rB2)-beta*rAB-delta*r12);
    # (1-2 swap: n-i, m-j. A-B swap: n-m, i-j, a-b)
    # (n, m, i, j, a, b)
    # (i, j, n, m, a, b)
    # (m, n, j, i, b, a)
    # (j, i, m, n, b, a)
    # Unique: n>m, n>=i, n>=j
    # or n=m, n>=i, i>j
    # or n=m=i=j, a>=b
        sym_b = {}
        for h in frange(0, h_max, 1):
            for k in frange(0, k_max, 1):
                for n in frange(0, n_max, 1):
                    for m in frange(0, n_max, 1):  # careful: Whenever using negative indices, adjust power_table call accordingly.
                        for i in frange(0, n_max, 1):
                            for j in frange(0, n_max, 1):
                                for ah in frange(0, ab_max, 1):
                                    for bh in frange(0, ab_max, 1):

                                        a = ah
                                        b = bh
                                        # constraints
                                        t = h + k + n + m + i + j
                                        if t > total_max: continue
                                        if a>0 and abs(n-i)>1: continue # prevent repeating basis functions by cancellation of rA, rB terms
                                        if b>0 and abs(m-j)>1: continue

                                        indexTuples = [(h, k, n, m, i, j, a, b), (h, k, i, j, n, m, a, b), (h, k, m, n, j, i, b, a), (h, k, j, i, m, n, b, a)]
                                        indexTuples = tuple(sorted(set(indexTuples)))
                                        rep = indexTuples[0]  # representative for all symmetry-equivalent tuples
                                        if rep in sym_b:
                                            sym_b[rep].append(len(rows))
                                        else:
                                            sym_b[rep] = [len(rows)]
                                        rows.append((h, k, n, m, i, j, a, b))

        row_idx = []
        col_idx = []
        for col, l in enumerate(sym_b.values()):
            for idx in l:
                row_idx.append(idx)
                col_idx.append(col)
        data = np.ones(len(row_idx), dtype=np.int8)
        X = csr_matrix((data, (row_idx, col_idx)), shape=(len(rows), len(sym_b)))

    elif coords == "stmu" or coords == "s12mu" or coords == "s12mu_morse":
        for h in frange(0, h_max, 1):
            for k in frange(0, k_max, 1):
                for n in frange(nm_min, n_max, 1):
                    for m in frange(nm_min, m_max, 1):  # careful: Whenever using negative indices, adjust power_table call accordingly.
                        for i in range(ij_max + 1):
                            for j in range(i + 1):
                                # constraints
                                if (i + j) % 2 != 0: continue  # A<->B symmetry
                                if coords == "stmu":
                                    if m % 2 != 0: continue
                                elif coords == "s12mu" or coords == "s12mu_morse":
                                    if i == j and m > n: continue # avoid duplication of (n, m, i, j=i) and (m, n, j=i, i)
                                t = h + n-nm_min + m + i + j # todo: temp: + k
                                if t > total_max: continue
                                # rows.append((h, k, n-i-k/2., m-j-k/2., i, j))
                                # rows.append((h, k, n-i, m-j, i, j))
                                # if coords == "s12mu_morse":
                                #     d = 0
                                #     while d <= nrDeltas:
                                #         rows.append((h+i+j, k, n, m, i, j, d))
                                #         d += 1
                                # else:
                                rows.append((h+i+j, k, n, m, i, j))


    elif coords == "s12uv_morse":
        for h in frange(0, h_max, 1):
            for k in frange(0, k_max, 1):
                for n in frange(0, n_max, 1):
                    for m in frange(0, n, 1):  # careful: Whenever using negative indices, adjust power_table call accordingly.
                        for i in frange(0, ij_max, 1): # Automatically symmetric in A-B and 1-2 due to cos() def
                            for j in frange(0, ij_max, 1):
                                t = h + n + m + i + j
                                if t > total_max: continue
                                if i + j > ij_max: continue # these functions are all really similar; the whole point of these coords is needing few i,j powers.
                                rows.append((h+i+j, k, n, m, i, j))

    basis_idx = np.array(rows, dtype=np.int16)

    matSize = basis_idx[:, 0].size
    bSize = len(sym_b) if coords == "rij" else matSize

    return basis_idx, bSize, X

def expand_idx(basis_idx, coords):
    h_idx = basis_idx[:, 0]
    k_idx = basis_idx[:, 1]
    n_idx = basis_idx[:, 2]
    m_idx = basis_idx[:, 3]
    i_idx = basis_idx[:, 4]
    j_idx = basis_idx[:, 5]

    if coords == "rij":
        a_idx = basis_idx[:, 6]
        b_idx = basis_idx[:, 7]
    # elif coords in ["s12mu_morse"]:
    #     a_idx = basis_idx[:, 6]  # index of delta grid
    #     b_idx = np.zeros_like(h_idx)
    else:
        a_idx = np.zeros_like(h_idx)
        b_idx = np.zeros_like(h_idx)
    return h_idx, k_idx, n_idx, m_idx, i_idx, j_idx, a_idx, b_idx