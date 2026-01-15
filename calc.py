import numpy as np
from numpy.ma.core import zeros_like


def potential_ri(rAB, rA1, rB1, rA2, rB2, r12):
    return 1/rAB + 1/r12 - 1/rA1 - 1/rB1 - 1/rA2 - 1/rB2

def potential_ri_inv(rAB_inv, rA1_inv, rB1_inv, rA2_inv, rB2_inv, r12_inv):  # For cases when the inv's are pre-computed
    return rAB_inv + r12_inv - rA1_inv - rB1_inv - rA2_inv - rB2_inv


def power_table(x, p_max, p_min=0):
    x = np.asarray(x, dtype=np.float64)

    n_pos = p_max + 1              # includes 0
    n_neg = -p_min if p_min < 0 else 0
    n_cols = n_pos + n_neg

    out = np.empty((x.shape[0], n_cols) if x.ndim else (n_cols,), dtype=np.float64)
    out[..., 0] = 1.0  # s^0

    for e in range(1, p_max + 1):
        out[..., e] = out[..., e - 1] * x  # s^1 ... s^p_max

    if p_min < 0:
        col = p_max + 1
        out[..., col] = x ** p_min
        for e in range(p_min + 1, 0):
            out[..., col + 1] = out[..., col] * x   # s^p_min ... s^-1
            col += 1

    return out

def calc_F_rij(basis_idx):

    h, k, n, m, i, j, a, b = expand_idx(basis_idx.astype(np.float64), "rij")

    F = np.stack([n * n + n, n, k * n, n * m, n * i, m * m + m, m, m * k, m * j,
                     i * i + i, i, i * j, i * k, i * h, j * j + j, j, j * k, j * h,
                     h * h + h, h, k * k + k, k, n*h, m*h, np.ones_like(n),
                     k*a, a*b, k*b, m*b, n*b, i*b, h*b, h*a, m*a, j*a, j*b, n*a, i*a, a*a, a, b*b, b
                    ], axis=0)

    return F


def calc_Fij_s12mu(basis_idx):

    h, k, n, m, i, j, _, _ = expand_idx(basis_idx.astype(np.float64), "s12mu")
    Fij = np.stack([n*n - n, n, k*k + k, k, m*m - m, m, i*i, i, j*j, j, h*h + h, h, n*i, m*j, n*j, n*m, j*k, n*h, i*k, m*h, n*k, m*i, j*h, m*k, i*j, i*h, np.ones_like(n)], axis=0)

    h, k, m, n, j, i, _, _ = expand_idx(basis_idx.astype(np.float64), "s12mu")  # switch 1 <-> 2
    Fji = np.stack([n*n - n, n, k*k + k, k, m*m - m, m, i*i, i, j*j, j, h*h + h, h, n*i, m*j, n*j, n*m, j*k, n*h, i*k, m*h, n*k, m*i, j*h, m*k, i*j, i*h, np.ones_like(n)], axis=0)

    # h, k, n, m, i, j, _, _ = expand_idx(basis_idx.astype(np.float64), "s12mu")
    # Fij = np.stack([n*n, n, k*k, k, m*m, m, i*i, i, j*j, j, h*h, h, n*i, m*j, n*j, n*m, j*k, n*h, i*k, m*h, n*k, m*i, j*h, m*k, i*j, i*h, np.ones_like(n)], axis=0)
    #
    # h, k, m, n, j, i, _, _ = expand_idx(basis_idx.astype(np.float64), "s12mu")  # switch 1 <-> 2
    # Fji = np.stack([n*n, n, k*k, k, m*m, m, i*i, i, j*j, j, h*h, h, n*i, m*j, n*j, n*m, j*k, n*h, i*k, m*h, n*k, m*i, j*h, m*k, i*j, i*h, np.ones_like(n)], axis=0)

    return Fij, Fji


def calc_Fij_stmu(basis_idx):

    h, k, n, m, i, j, _, _ = expand_idx(basis_idx.astype(np.float64), "stmu")

    Fij = np.stack([n * n - n, n, k * n, n * m, n * (m - h), n * i, n * j,
                     m * m, m, m * k, m * i, m * j, m * (i + j - h),
                     i * i - i, i, i * j, i * k, i * h,
                     j * j - j, j, j * k, j * h,
                     h * h + h, h, k * k + k, k,
                     np.ones_like(n)], axis=0)
    Fji = np.stack([n * n - n, n, k * n, n * m, n * (m - h), n * j, n * i,
                    m * m, m, m * k, m * j, m * i, m * (j + i - h),
                    j * j - j, j, j * i, j * k, j * h,
                    i * i - i, i, i * k, i * h,
                    h * h + h, h, k * k + k, k,
                    np.ones_like(n)], axis=0)
    return Fij, Fji

def calc_H_alphabeta_stmu(Fij, Fji, rAB, rA1, rB1, rA2, rB2, r12, M_inv, M1M, s1, s2, s, mu1, mu2, delta):

    use_delta = delta != 0

    # rAB, s - only primitives
    inv_rAB = 1.0 / rAB
    rAB_2 = rAB * rAB
    inv_s = 1.0 / s
    inv_s_2 = inv_s * inv_s
    inv_rAB_2 = inv_rAB * inv_rAB

    # e1-only primitives
    inv_rA1 = 1.0 / rA1
    inv_rB1 = 1.0 / rB1
    rA1_2 = rA1 * rA1
    rB1_2 = rB1 * rB1
    # mu1 = (rA1 - rB1) * inv_rAB
    inv_mu1 = 1 / mu1
    inv_mu1_2 = inv_mu1 * inv_mu1
    v1 = inv_rA1 + inv_rB1
    cos1AB = (rA1_2 + rAB_2 - rB1_2) * inv_rA1 * inv_rAB  * 0.5
    cos1BA = (rAB_2 + rB1_2 - rA1_2) * inv_rAB * inv_rB1  * 0.5
    cosA1B = (rA1_2 - rAB_2 + rB1_2) * inv_rA1 * inv_rB1  * 0.5
    c11 = M_inv * (cos1AB - cos1BA)

    # e2-only primitives
    inv_rA2 = 1.0 / rA2
    inv_rB2 = 1.0 / rB2
    rA2_2 = rA2 * rA2
    rB2_2 = rB2 * rB2
    # mu2 = (rA2 - rB2) * inv_rAB
    inv_mu2 = 1.0 / mu2
    inv_mu2_2 = inv_mu2 * inv_mu2
    v2 = inv_rA2 + inv_rB2
    cos2AB = (rA2_2 + rAB_2 - rB2_2) * inv_rA2 * inv_rAB  * 0.5
    cos2BA = (rAB_2 + rB2_2 - rA2_2) * inv_rAB * inv_rB2  * 0.5
    cosA2B = (rA2_2 - rAB_2 + rB2_2) * inv_rA2 * inv_rB2  * 0.5
    c12 = M_inv * (cos2AB - cos2BA)

    # Mixed but scalar
    t = (s1 - s2) * inv_rAB  * 0.5
    inv_t = 1 / t
    inv_t_2 = inv_t * inv_t
    inv_rAB_s = inv_rAB * inv_s
    inv_rAB_t = inv_rAB * inv_t

    P1 = rA1.size
    P2 = rA2.size
    P = P1 * P2

    # expand e1
    rA1 = np.repeat(rA1, P2)
    rB1 = np.repeat(rB1, P2)
    rA1_2 = np.repeat(rA1_2, P2)
    rB1_2 = np.repeat(rB1_2, P2)
    inv_rA1 = np.repeat(inv_rA1, P2)
    inv_rB1 = np.repeat(inv_rB1, P2)
    mu1 = np.repeat(mu1, P2)
    inv_mu1 = np.repeat(inv_mu1, P2)
    inv_mu1_2 = np.repeat(inv_mu1_2, P2)
    v1 = np.repeat(v1, P2)
    cos1AB = np.repeat(cos1AB, P2)
    cos1BA = np.repeat(cos1BA, P2)
    cosA1B = np.repeat(cosA1B, P2)
    c11 = np.repeat(c11, P2)

    # expand e2
    rA2 = np.tile(rA2, P1)
    rB2 = np.tile(rB2, P1)
    rA2_2 = np.tile(rA2_2, P1)
    rB2_2 = np.tile(rB2_2, P1)
    inv_rA2 = np.tile(inv_rA2, P1)
    inv_rB2 = np.tile(inv_rB2, P1)
    mu2 = np.tile(mu2, P1)
    inv_mu2 = np.tile(inv_mu2, P1)
    inv_mu2_2 = np.tile(inv_mu2_2, P1)
    v2 = np.tile(v2, P1)
    cos2AB = np.tile(cos2AB, P1)
    cos2BA = np.tile(cos2BA, P1)
    cosA2B = np.tile(cosA2B, P1)
    c12 = np.tile(c12, P1)

    inv_r12 = 1.0 / r12
    r12_2 = r12 * r12
    inv_r12_2 = inv_r12 * inv_r12

    v12 = v1 + v2

    cos12A = (r12_2 - rA1_2 + rA2_2) * inv_r12 * inv_rA2  * 0.5
    cos12B = (r12_2 - rB1_2 + rB2_2) * inv_r12 * inv_rB2  * 0.5
    cos1A2 = (rA1_2 + rA2_2 - r12_2) * inv_rA1 * inv_rA2  * 0.5
    cos1B2 = (rB1_2 + rB2_2 - r12_2) * inv_rB1 * inv_rB2  * 0.5
    cos21A = (r12_2 + rA1_2 - rA2_2) * inv_r12 * inv_rA1  * 0.5
    cos21B = (r12_2 + rB1_2 - rB2_2) * inv_r12 * inv_rB1  * 0.5

    # Recurring combinations
    c1 = cos12A + cos12B + cos21A + cos21B
    c2 = cosA1B + cosA2B
    c3 = (cos12A - cos21B + cos12B - cos21A)
    c4 = cos21A - cos21B
    c5 = cos12A - cos12B
    c6 = cosA2B - cosA1B
    c7 = M_inv * (cos1A2 + cos1B2)
    c8 = M_inv * (cos1AB + cos1BA + cos2AB + cos2BA)
    c9 = M_inv * (cos1AB + cos1BA - cos2AB - cos2BA)
    c10 = M_inv * (cos1A2 - cos1B2)

    # Coefficients of: [n^2, n, k*n, n*m, m^2, m, m*k, i^2-i, i, i*k, j^2-j, j, j*k, k^2 + k, k, 1]
    c_n2 = -(2.0 * M1M + c2 + c7) * inv_s_2  # n^2 - n
    c_n = - M1M * v12 * inv_s  # n
    c_n_alpha = (2.0 * (M1M * 2.0 + c2 + c7)) * inv_s  # n
    c_kn = -c1 * inv_r12 * inv_s  # k*n
    c_nm = c6 * inv_t * inv_rAB_s  # n*m
    c_nmh = c8 * inv_rAB_s  # n*(m-h)
    c_ni = (c8 - c10 * inv_mu1) * inv_rAB_s  # n*i
    c_nj = (c8 - c10 * inv_mu2) * inv_rAB_s  # n*j
    c_m2 = -0.25 * (
                2.0 *M1M + c2 - c7) * inv_t_2 * inv_rAB_2 - M_inv * inv_rAB_2 + 0.5 * c9 * inv_rAB_2 * inv_t  # m^2
    c_m = 0.5 * M1M * (v2 - v1) * inv_rAB_t + 0.25 * (
                M1M * 2.0 + c2 - c7) * inv_t_2 * inv_rAB_2 + M_inv * inv_rAB_2  # m
    c_m_alpha = - c6 * inv_rAB_t - c8 * inv_rAB
    c_mk = 0.5 * c3 * inv_r12 * inv_rAB_t  # m*k
    c_mi = (c11 + 0.5 * c10 * inv_t) * inv_mu1 * inv_rAB_2  # m*i
    c_mj = (c12 - 0.5 * c10 * inv_t) * inv_mu2 * inv_rAB_2  # m*j
    c_mijh = (0.5 * c9 * inv_t - 2.0 * M_inv) * inv_rAB_2  # m*(i+j-h)
    c_i2 = ((-M1M + cosA1B) * inv_mu1_2 * inv_rAB_2 + c11 * inv_mu1 * inv_rAB_2 - M_inv * inv_rAB_2) * np.ones_like(
        c_n)  # i^2-i
    c_i = (-M1M * (inv_rA1 - inv_rB1) + c11 * inv_rAB) * inv_mu1 * inv_rAB * np.ones_like(c_n)  # i
    c_i_alpha = (c10 * inv_mu1 - c8) * inv_rAB * np.ones_like(c_n)  # i
    c_ij = (-c7 * inv_mu1 * inv_mu2 + c11 * inv_mu1 + c12 * inv_mu2 - 2.0 * M_inv) * inv_rAB_2  # i*j
    c_ik = -c4 * inv_r12 * inv_mu1 * inv_rAB  # i*k
    c_ih = (-c11 * inv_mu1 + 2.0 * M_inv) * inv_rAB_2 * np.ones_like(c_n)  # i*h
    c_j2 = (-M1M + cosA2B) * inv_mu2_2 * inv_rAB_2 + c12 * inv_mu2 * inv_rAB_2 - M_inv * inv_rAB_2  # j^2-j
    c_j = (-M1M * (inv_rA2 - inv_rB2) + c12 * inv_rAB) * inv_mu2 * inv_rAB  # j
    c_j_alpha = (c10 * inv_mu2 - c8) * inv_rAB  # j
    c_jk = -c5 * inv_r12 * inv_mu2 * inv_rAB  # j*k
    c_jh = (-c12 * inv_mu2 + 2.0 * M_inv) * inv_rAB_2  # j*h
    c_h2 = -M_inv * inv_rAB_2 * np.ones_like(c_n)  # h^2 + h
    c_h_alpha = c8 * inv_rAB * np.ones_like(c_n)  # h
    c_k2 = - inv_r12_2  # k^2 + k
    c_k_alpha = c1 * inv_r12  # k
    c_1_alpha = M1M * v12
    c_1_alpha2 = -(2 * M1M + c2 + c7)

    c_n_beta = c8 * inv_s  # n
    c_m_beta = 0.5 * c9 * inv_rAB_t - 2. * M_inv * inv_rAB
    c_i_beta = (c11 * inv_mu1 - 2. * M_inv) * inv_rAB * np.ones_like(c_n)  # i
    c_j_beta = (c12 * inv_mu2 - 2. * M_inv) * inv_rAB  # j
    c_h_beta = 2.0 * M_inv * inv_rAB * np.ones_like(c_n)  # h
    c_1_alphabeta = -c8
    c_1_beta = M_inv * 2. * inv_rAB * np.ones_like(c_n)
    c_1_beta2 = -M_inv * np.ones_like(c_n)

    if use_delta:
        c_m_delta = - 0.5 * c3 * inv_rAB_t
        c_n_delta = c1 * inv_s  # n
        c_i_delta = c4 * inv_mu1 * inv_rAB * np.ones_like(c_n)  # i
        c_j_delta = c5 * inv_mu2 * inv_rAB  # j
        c_1_alphadelta = -c1
        c_1_delta = 2.0 * inv_r12
        c_1_delta2 = -1.0 * np.ones_like(c_n)
        c_k_delta = 2.0 * inv_r12  # k

    zero = np.zeros_like(r12)
    coef_vector_1 = np.stack(
        [c_n2, c_n, c_kn, c_nm, c_nmh, c_ni, c_nj, c_m2, c_m, c_mk, c_mi, c_mj, c_mijh, c_i2, c_i, c_ij, c_ik, c_ih,
         c_j2, c_j, c_jk, c_jh, c_h2, zero, c_k2, zero, zero], axis=1)
    coef_vector_alpha = np.stack(
        [zero, c_n_alpha, zero, zero, zero, zero, zero, zero, c_m_alpha, zero, zero, zero, zero, zero, c_i_alpha,
         zero, zero, zero, zero, c_j_alpha, zero, zero, zero, c_h_alpha, zero, c_k_alpha, c_1_alpha], axis=1)
    coef_vector_alpha2 = np.stack(
        [zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
         zero, zero, zero, zero, zero, zero, zero, zero, c_1_alpha2], axis=1)

    coef_vector_beta = np.stack(
        [zero, c_n_beta, zero, zero, zero, zero, zero, zero, c_m_beta, zero, zero, zero, zero, zero, c_i_beta, zero,
         zero, zero, zero, c_j_beta, zero, zero, zero, c_h_beta, zero, zero, c_1_beta], axis=1)
    coef_vector_alphabeta = np.stack(
        [zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
         zero, zero, zero, zero, zero, zero, zero, zero, c_1_alphabeta], axis=1)
    coef_vector_beta2 = np.stack(
        [zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
         zero, zero, zero, zero, zero, zero, zero, zero, c_1_beta2], axis=1)

    if use_delta:
        coef_vector_delta = np.stack(
            [zero, c_n_delta, zero, zero, zero, zero, zero, zero, c_m_delta, zero, zero, zero, zero, zero,
             c_i_delta, zero, zero, zero, zero, c_j_delta, zero, zero, zero, zero, zero, c_k_delta, c_1_delta],
            axis=1)
        coef_vector_alphadelta = np.stack(
            [zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
             zero, zero, zero, zero, zero, zero, zero, zero, zero, c_1_alphadelta], axis=1)
        coef_vector_delta2 = np.stack(
            [zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
             zero, zero, zero, zero, zero, zero, zero, zero, zero, c_1_delta2], axis=1)

    potential = potential_ri(rAB, rA1, rB1, rA2, rB2, r12)
    H_1_ij = coef_vector_1 @ Fij + potential[:, None]
    H_1_ji = coef_vector_1 @ Fji + potential[:, None]
    if use_delta:
        H_1_ij += (coef_vector_delta * delta + coef_vector_delta2 * delta ** 2) @ Fij
        H_1_ji += (coef_vector_delta * delta + coef_vector_delta2 * delta ** 2) @ Fji

    H_alpha_ij = coef_vector_alpha @ Fij
    H_alpha_ji = coef_vector_alpha @ Fji
    if use_delta:
        tmp = coef_vector_alphadelta * delta @ Fij
        H_alpha_ij += tmp
        H_alpha_ji += tmp
    H_alpha2 = coef_vector_alpha2 @ Fij
    H_alphabeta = coef_vector_alphabeta @ Fij
    H_beta_ij = coef_vector_beta @ Fij
    H_beta_ji = coef_vector_beta @ Fji
    H_beta2 = coef_vector_beta2 @ Fij

    return H_1_ij, H_1_ji, H_alpha_ij, H_alpha_ji, H_alpha2, H_alphabeta, H_beta_ij, H_beta_ji, H_beta2

def calc_H_alphabeta_rij(F, rAB, rA1, rB1, rA2, rB2, r12, Minv, M1M, delta):

    # rAB, s - only primitives
    inv_rAB = 1.0 / rAB
    rAB_2 = rAB * rAB
    inv_rAB_2 = inv_rAB * inv_rAB

    # e1-only primitives
    inv_rA1 = 1.0 / rA1
    inv_rB1 = 1.0 / rB1
    inv_rA1_2 = inv_rA1**2
    inv_rB1_2 = inv_rB1**2
    rA1_2 = rA1 * rA1
    rB1_2 = rB1 * rB1
    cos1AB = Minv * (rA1_2 + rAB_2 - rB1_2) * inv_rA1 * inv_rAB * 0.5
    cos1BA = Minv * (rAB_2 + rB1_2 - rA1_2) * inv_rAB * inv_rB1 * 0.5
    cosA1B = (rA1_2 - rAB_2 + rB1_2) * inv_rA1 * inv_rB1 * 0.5

    # e2-only primitives
    inv_rA2 = 1.0 / rA2
    inv_rB2 = 1.0 / rB2
    inv_rA2_2 = inv_rA2**2
    inv_rB2_2 = inv_rB2**2
    rA2_2 = rA2 * rA2
    rB2_2 = rB2 * rB2
    cos2AB = Minv * (rA2_2 + rAB_2 - rB2_2) * inv_rA2 * inv_rAB * 0.5
    cos2BA = Minv * (rAB_2 + rB2_2 - rA2_2) * inv_rAB * inv_rB2 * 0.5
    cosA2B = (rA2_2 - rAB_2 + rB2_2) * inv_rA2 * inv_rB2 * 0.5

    P1 = rA1.size
    P2 = rA2.size

    # expand e1
    rA1 = np.repeat(rA1, P2)
    rB1 = np.repeat(rB1, P2)
    rA1_2 = np.repeat(rA1_2, P2)
    rB1_2 = np.repeat(rB1_2, P2)
    inv_rA1 = np.repeat(inv_rA1, P2)
    inv_rB1 = np.repeat(inv_rB1, P2)
    inv_rA1_2 = np.repeat(inv_rA1_2, P2)
    inv_rB1_2 = np.repeat(inv_rB1_2, P2)
    cos1AB = np.repeat(cos1AB, P2)
    cos1BA = np.repeat(cos1BA, P2)
    cosA1B = np.repeat(cosA1B, P2)

    # expand e2
    rA2 = np.tile(rA2, P1)
    rB2 = np.tile(rB2, P1)
    rA2_2 = np.tile(rA2_2, P1)
    rB2_2 = np.tile(rB2_2, P1)
    inv_rA2 = np.tile(inv_rA2, P1)
    inv_rB2 = np.tile(inv_rB2, P1)
    inv_rA2_2 = np.tile(inv_rA2_2, P1)
    inv_rB2_2 = np.tile(inv_rB2_2, P1)
    cos2AB = np.tile(cos2AB, P1)
    cos2BA = np.tile(cos2BA, P1)
    cosA2B = np.tile(cosA2B, P1)

    inv_r12 = 1.0 / r12
    r12_2 = r12 * r12
    inv_r12_2 = inv_r12 * inv_r12
    rA = np.sqrt(rA1 ** 2 + rA2 ** 2)
    rB = np.sqrt(rB1 ** 2 + rB2 ** 2)
    inv_rA = 1.0 / rA
    inv_rB = 1.0 / rB
    inv_rA_2 = inv_rA**2
    inv_rB_2 = inv_rB**2

    cos12A = (r12_2 - rA1_2 + rA2_2) * inv_r12 * inv_rA2 * 0.5
    cos12B = (r12_2 - rB1_2 + rB2_2) * inv_r12 * inv_rB2 * 0.5
    cos1A2 = (rA1_2 + rA2_2 - r12_2) * inv_rA1 * inv_rA2 * 0.5
    cos1B2 = (rB1_2 + rB2_2 - r12_2) * inv_rB1 * inv_rB2 * 0.5
    cos21A = (r12_2 + rA1_2 - rA2_2) * inv_r12 * inv_rA1 * 0.5
    cos21B = (r12_2 + rB1_2 - rB2_2) * inv_r12 * inv_rB1 * 0.5

    # Coefficients
    c_m2 = -inv_rB1_2 * M1M * 0.5
    c_m = cos21B * inv_rB1 * delta
    c_i2 = -inv_rA2_2 * M1M * 0.5
    c_i = cos12A * inv_rA2 * delta
    c_n2 = -inv_rA1_2 * M1M * 0.5
    c_n = cos21A * inv_rA1 * delta
    c_j2 = -inv_rB2_2 * M1M * 0.5 # j*(j+1)
    c_j = cos12B * inv_rB2 * delta
    c_k2 = -inv_r12_2   # k*(k+1)
    c_k = 2.0 * delta * inv_r12
    c_h2 = -inv_rAB_2 * Minv * np.ones_like(c_n) # h*(h+1)
    c_ij = -cosA2B * inv_rA2 * inv_rB2
    c_ik = -cos12A * inv_rA2 * inv_r12
    c_nm = -cosA1B * inv_rA1 * inv_rB1
    c_mk = -cos21B * inv_rB1 * inv_r12
    c_nk = -cos21A * inv_rA1 * inv_r12
    c_jk = -cos12B * inv_rB2 * inv_r12
    c_ni = -cos1A2 * inv_rA1 * inv_rA2 * Minv
    c_mj = -cos1B2 * inv_rB1 * inv_rB2 * Minv
    c_ih = -cos2AB * inv_rA2 * inv_rAB
    c_jh = -cos2BA * inv_rB2 * inv_rAB
    c_nh = -cos1AB * inv_rA1 * inv_rAB
    c_1 = -delta ** 2.0 + 2.0 * delta * inv_r12

    rA1_over_rA_2 = rA1 * inv_rA_2
    rA2_over_rA_2 = rA2 * inv_rA_2
    M1M_over_rA_2 = M1M * inv_rA_2
    rB1_over_rB_2 = rB1 * inv_rB_2
    rB2_over_rB_2 = rB2 * inv_rB_2
    M1M_over_rB_2 = M1M * inv_rB_2

    c_ka = cos12A * inv_r12 * rA2_over_rA_2 + cos21A * inv_r12 * rA1_over_rA_2
    c_ab = -cosA1B * rA1_over_rA_2 * rB1_over_rB_2 - cosA2B * rA2_over_rA_2 * rB2_over_rB_2
    c_kb = (cos12B * rB2_over_rB_2 + cos21B * rB1_over_rB_2) * inv_r12
    c_mb = cos1B2 * rB1_over_rB_2 * rB2 * Minv + M1M_over_rB_2
    c_nb = cosA1B * inv_rA1 * rB1_over_rB_2
    c_ib = cosA2B * inv_rA2 * rB2_over_rB_2
    c_mh = -cos1BA * inv_rB1 * inv_rAB
    c_hb = cos1BA * inv_rAB * rB1_over_rB_2 + cos2BA * inv_rAB * rB2_over_rB_2
    c_ha = cos1AB * inv_rAB * rA1_over_rA_2 + cos2AB * inv_rAB * rA2_over_rA_2
    c_ma = cosA1B * inv_rB1 * rA1_over_rA_2
    c_ja = cosA2B * inv_rB2 * rA2_over_rA_2
    c_jb = cos1B2 * inv_rB2 * rB1_over_rB_2 * Minv + inv_rB_2 * M1M
    c_na = cos1A2 * inv_rA1 * rA2_over_rA_2 * Minv + M1M_over_rA_2
    c_ia = cos1A2 * inv_rA2 * rA1_over_rA_2 * Minv + M1M_over_rA_2
    c_a2 = -0.5 * M1M_over_rA_2 - cos1A2 * rA1_over_rA_2 * rA2_over_rA_2 * Minv
    c_a = -2.0 * cos1A2 * rA1_over_rA_2 * rA2_over_rA_2 * Minv - cos12A * delta * rA2_over_rA_2 - cos21A * delta * rA1_over_rA_2 + 2.0 * M1M_over_rA_2
    c_b2 = -inv_rB_2 * M1M * 0.5 - cos1B2 * rB1_over_rB_2 * rB2_over_rB_2 * Minv
    c_b = -2.0 * cos1B2 * rB1_over_rB_2 * rB2_over_rB_2 * Minv - cos12B * delta * rB2_over_rB_2 - cos21B * delta * rB1_over_rB_2 + 2.0 * inv_rB_2 * M1M

    zero = np.zeros_like(r12)
    coef_vector_1 = np.stack(
        [c_n2, c_n, c_nk, c_nm, c_ni, c_m2, c_m, c_mk, c_mj, c_i2, c_i, c_ij, c_ik, c_ih, c_j2, c_j, c_jk, c_jh, c_h2, zero, c_k2, c_k, c_nh, c_mh, c_1,
         c_ka, c_ab, c_kb, c_mb, c_nb, c_ib, c_hb, c_ha, c_ma, c_ja, c_jb, c_na, c_ia, c_a2, c_a, c_b2, c_b], axis=1)

    ca1 = (cos12A + cos12B + cos21A + cos21B)
    ca2 = (cos1AB + cos1BA + cos2AB + cos2BA)
    ca3 = (Minv * cos1A2 + M1M + cosA1B)
    ca4 = (Minv * cos1B2 + M1M + cosA1B)
    ca5 = (Minv * cos1A2 + M1M + cosA2B)
    ca6 = (Minv * cos1B2 + M1M + cosA2B)

    c_alpha_k = ca1 * inv_r12
    c_alpha_h = ca2 * inv_rAB
    c_alpha_n = ca3 * inv_rA1
    c_alpha_m = ca4 * inv_rB1
    c_alpha_i = ca5 * inv_rA2
    c_alpha_j = ca6 * inv_rB2
    c_alpha_a = -ca3 * rA1_over_rA_2 - ca5 * rA2_over_rA_2
    c_alpha_b = -ca4 * rB1_over_rB_2 - ca6 * rB2_over_rB_2
    c_alpha_1 = M1M * (inv_rA1 + inv_rA2 + inv_rB1 + inv_rB2) - ca1 * delta

    c_alpha2_1 = -2.0 * M1M - cosA1B - cosA2B - Minv * (cos1A2 + cos1B2)
    c_alphabeta_1 = -ca2

    c_beta_n = cos1AB * inv_rA1
    c_beta_m = cos1BA * inv_rB1
    c_beta_i = cos2AB * inv_rA2
    c_beta_j = cos2BA * inv_rB2
    c_beta_h = 2.0 * inv_rAB * Minv * np.ones_like(c_n)
    c_beta_a = -cos1AB * rA1_over_rA_2 - cos2AB * rA2_over_rA_2
    c_beta_b = -cos1BA * rB1_over_rB_2 - cos2BA * rB2_over_rB_2
    c_beta_1 = 2.0 * inv_rAB * Minv * np.ones_like(c_n)

    c_beta2_1 = -Minv * np.ones_like(c_n)

    coef_vector_alpha = np.stack(
        [zero, c_alpha_n, zero, zero, zero, zero, c_alpha_m, zero, zero, zero, c_alpha_i, zero, zero, zero, zero, c_alpha_j, zero, zero, zero, c_alpha_h, zero, c_alpha_k, zero, zero, c_alpha_1,
         zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_alpha_a, zero, c_alpha_b], axis=1)
    coef_vector_alpha2 = np.stack(
        [zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_alpha2_1,
         zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero], axis=1)
    coef_vector_alphabeta = np.stack(
        [zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_alphabeta_1,
         zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero], axis=1)
    coef_vector_beta = np.stack(
        [zero, c_beta_n, zero, zero, zero, zero, c_beta_m, zero, zero, zero, c_beta_i, zero, zero, zero, zero, c_beta_j, zero, zero, zero, c_beta_h, zero, zero, zero, zero, c_beta_1,
         zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_beta_a, zero, c_beta_b], axis=1)
    coef_vector_beta2 = np.stack(
        [zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_beta2_1,
         zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero], axis=1)

    potential = potential_ri(rAB, rA1, rB1, rA2, rB2, r12)
    H_1 = coef_vector_1 @ F + potential[:, None]
    H_alpha = coef_vector_alpha @ F
    H_alpha2 = coef_vector_alpha2 @ F
    H_alphabeta = coef_vector_alphabeta @ F
    H_beta = coef_vector_beta @ F
    H_beta2 = coef_vector_beta2 @ F

    return H_1, H_alpha, H_alpha2, H_alphabeta, H_beta, H_beta2, inv_rA, inv_rB

def calc_H_alphabeta_s12mu(Fij, Fji, rAB, rA1, rB1, rA2, rB2, r12, Minv, M1M, s1, s2, mu1, mu2, delta):

    # rAB, s1, s2 - only primitives (scalars)
    inv_rAB = 1.0 / rAB
    rAB_2 = rAB * rAB
    inv_rAB_2 = inv_rAB * inv_rAB
    inv_s1 = 1.0 / s1
    inv_s1_2 = inv_s1**2
    s1_2 = s1**2
    inv_s2 = 1.0 / s2
    inv_s2_2 = inv_s2**2
    s2_2 = s2**2

    # e1-only primitives
    inv_rA1 = 1.0 / rA1
    inv_rB1 = 1.0 / rB1
    rA1_2 = rA1 * rA1
    rB1_2 = rB1 * rB1
    mu1_2 = mu1**2
    inv_mu1 = 1 / mu1
    inv_mu1_2 = inv_mu1 * inv_mu1
    v1 = inv_rA1 + inv_rB1
    cos1AB = (rA1_2 + rAB_2 - rB1_2) * inv_rA1 * inv_rAB  * 0.5
    cos1BA = (rAB_2 + rB1_2 - rA1_2) * inv_rAB * inv_rB1  * 0.5
    cosA1B = (rA1_2 - rAB_2 + rB1_2) * inv_rA1 * inv_rB1  * 0.5

    # e2-only primitives
    inv_rA2 = 1.0 / rA2
    inv_rB2 = 1.0 / rB2
    rA2_2 = rA2 * rA2
    rB2_2 = rB2 * rB2
    mu2_2 = mu2**2
    inv_mu2 = 1.0 / mu2
    inv_mu2_2 = inv_mu2 * inv_mu2
    v2 = inv_rA2 + inv_rB2
    cos2AB = (rA2_2 + rAB_2 - rB2_2) * inv_rA2 * inv_rAB  * 0.5
    cos2BA = (rAB_2 + rB2_2 - rA2_2) * inv_rAB * inv_rB2  * 0.5
    cosA2B = (rA2_2 - rAB_2 + rB2_2) * inv_rA2 * inv_rB2  * 0.5

    P1 = rA1.size
    P2 = rA2.size
    P = P1 * P2

    # expand e1
    rA1_2 = np.repeat(rA1_2, P2)
    rB1_2 = np.repeat(rB1_2, P2)
    inv_rA1 = np.repeat(inv_rA1, P2)
    inv_rB1 = np.repeat(inv_rB1, P2)
    inv_mu1 = np.repeat(inv_mu1, P2)
    inv_mu1_2 = np.repeat(inv_mu1_2, P2)
    mu1_2 = np.repeat(mu1_2, P2)
    v1 = np.repeat(v1, P2)
    cos1AB = np.repeat(cos1AB, P2)
    cos1BA = np.repeat(cos1BA, P2)
    cosA1B = np.repeat(cosA1B, P2)

    # expand e2
    rA2_2 = np.tile(rA2_2, P1)
    rB2_2 = np.tile(rB2_2, P1)
    inv_rA2 = np.tile(inv_rA2, P1)
    inv_rB2 = np.tile(inv_rB2, P1)
    inv_mu2 = np.tile(inv_mu2, P1)
    inv_mu2_2 = np.tile(inv_mu2_2, P1)
    mu2_2 = np.tile(mu2_2, P1)
    v2 = np.tile(v2, P1)
    cos2AB = np.tile(cos2AB, P1)
    cos2BA = np.tile(cos2BA, P1)
    cosA2B = np.tile(cosA2B, P1)

    inv_r12 = 1.0 / r12
    r12_2 = r12 * r12
    inv_r12_2 = inv_r12 * inv_r12

    inv_rA1_rB1 = inv_rA1 * inv_rB1 * M1M
    inv_rA2_rB2 = inv_rA2 * inv_rB2 * M1M
    v12 = v1 + v2

    cos12A = (r12_2 - rA1_2 + rA2_2) * inv_r12 * inv_rA2  * 0.5
    cos12B = (r12_2 - rB1_2 + rB2_2) * inv_r12 * inv_rB2  * 0.5
    cos1A2 = (rA1_2 + rA2_2 - r12_2) * inv_rA1 * inv_rA2  * 0.5
    cos1B2 = (rB1_2 + rB2_2 - r12_2) * inv_rB1 * inv_rB2  * 0.5
    cos21A = (r12_2 + rA1_2 - rA2_2) * inv_r12 * inv_rA1  * 0.5
    cos21B = (r12_2 + rB1_2 - rB2_2) * inv_r12 * inv_rB1  * 0.5

    # Recurring combinations
    one = np.ones_like(r12)
    c1 = (cos1AB + cos1BA) * inv_rAB * Minv * one
    c2 = (cos2AB + cos2BA) * inv_rAB * Minv * one
    c3 = (cos21A + cos21B + cos12A + cos12B)
    c4 = (cos1A2 + cos1B2) * Minv

    c_n2 = -inv_s1_2 * M1M - cosA1B * inv_s1_2
    c_n = -inv_rA1_rB1 + (cos21A + cos21B) * delta * inv_s1
    c_m2 = -inv_s2_2 * M1M - cosA2B * inv_s2_2
    c_m = -inv_rA2_rB2 + (cos12A + cos12B) * delta * inv_s2
    c_k2 = -inv_r12_2 * one
    c_k = 2 * delta * inv_r12 * one
    c_i2 = -inv_mu1_2 * inv_rAB_2 * M1M + inv_rAB_2 * (inv_mu1 * cos1AB - inv_mu1 * cos1BA - 1) * Minv + cosA1B * inv_mu1_2 * inv_rAB_2
    c_i = inv_mu1_2 * inv_rAB_2 * M1M + inv_rA1_rB1 + cos21A * delta * inv_mu1 * inv_rAB - cos21B * delta * inv_mu1 * inv_rAB - cosA1B * inv_mu1_2 * inv_rAB_2 + inv_rAB_2 * Minv
    c_j2 = -inv_mu2_2 * inv_rAB_2 * M1M + inv_rAB_2 * (inv_mu2 * cos2AB - inv_mu2 * cos2BA - 1) * Minv + cosA2B * inv_mu2_2 * inv_rAB_2
    c_j = inv_mu2_2 * inv_rAB_2 * M1M + inv_rA2_rB2 + cos12A * delta * inv_mu2 * inv_rAB - cos12B * delta * inv_mu2 * inv_rAB - cosA2B * inv_mu2_2 * inv_rAB_2 + inv_rAB_2 * Minv
    c_h2 = -inv_rAB_2 * Minv *one
    c_ni = inv_s1 * c1
    c_mj = inv_s2 * c2
    c_nj = -inv_rAB * inv_s1 * (inv_mu2 * cos1A2 - inv_mu2 * cos1B2 - cos1AB - cos1BA) * Minv
    c_nm = -inv_s2 * c4 * inv_s1
    c_jk = -inv_r12 * inv_rAB * (cos12A - cos12B) * inv_mu2
    c_nh = -c_ni
    c_ik = -inv_mu1 * inv_r12 * (cos21A - cos21B) * inv_rAB
    c_mh = -c_mj
    c_nk = -inv_r12 * (cos21A + cos21B) * inv_s1
    c_mi = -inv_rAB * inv_s2 * (cos1A2 * inv_mu1 - inv_mu1 * cos1B2 - cos2AB - cos2BA) * Minv
    c_jh = -inv_rAB_2 * (inv_mu2 * cos2AB - inv_mu2 * cos2BA - 2) * Minv
    c_mk = -inv_r12 * (cos12A + cos12B) * inv_s2
    c_ij = -inv_rAB_2 * (cos1A2 * inv_mu1 * inv_mu2 + cos1B2 * inv_mu1 * inv_mu2 - inv_mu1 * cos1AB + inv_mu1 * cos1BA - inv_mu2 * cos2AB + inv_mu2 * cos2BA + 2) * Minv  # todo: c4
    c_ih = -inv_rAB_2 * (inv_mu1 * cos1AB - inv_mu1 * cos1BA - 2) * Minv
    c_1 = -delta * (-2 * inv_r12 + delta) * one

    c_alpha_1 = M1M * v12 - delta * c3
    c_alpha_n = 2 * inv_s1 * M1M + 2 * cosA1B * inv_s1 + inv_s1 * c4
    c_alpha_m = 2 * inv_s2 * M1M + 2 * cosA2B * inv_s2 + inv_s2 * c4
    c_alpha_i = inv_rAB * (cos1A2 * inv_mu1 - inv_mu1 * cos1B2 - cos1AB - cos1BA - cos2AB - cos2BA) * Minv
    c_alpha_j = inv_rAB * (inv_mu2 * cos1A2 - inv_mu2 * cos1B2 - cos1AB - cos1BA - cos2AB - cos2BA) * Minv
    c_alpha_h = c1 + c2
    c_alpha_k = c3 * inv_r12

    c_beta_1 = 2 * inv_rAB * Minv * one
    c_beta_n = inv_s1 * (cos1AB + cos1BA) * Minv
    c_beta_m = (cos2AB + cos2BA) * inv_s2 * Minv
    c_beta_i = inv_rAB * (inv_mu1 * cos1AB - inv_mu1 * cos1BA - 2) * Minv
    c_beta_j = inv_rAB * (inv_mu2 * cos2AB - inv_mu2 * cos2BA - 2) * Minv
    c_beta_h = 2 * inv_rAB * Minv * one

    c_alpha2_1 = -2 * M1M - (cosA1B + cosA2B) - (cos1A2 + cos1B2) * Minv
    c_alphabeta_1 = -(cos1BA + cos2AB + cos2BA + cos1AB) * Minv
    c_beta2_1 = -Minv

    zero = np.zeros_like(r12)
    coef_vector_1 = np.stack([c_n2, c_n, c_k2, c_k, c_m2, c_m, c_i2, c_i, c_j2, c_j, c_h2, zero, c_ni, c_mj, c_nj, c_nm, c_jk, c_nh, c_ik, c_mh, c_nk, c_mi, c_jh, c_mk, c_ij, c_ih, c_1], axis=1)
    coef_vector_alpha = np.stack([zero, c_alpha_n, zero, c_alpha_k, zero, c_alpha_m, zero, c_alpha_i, zero, c_alpha_j, zero, c_alpha_h, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_alpha_1], axis=1)
    coef_vector_beta = np.stack([zero, c_beta_n, zero, zero, zero, c_beta_m, zero, c_beta_i, zero, c_beta_j, zero, c_beta_h, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_beta_1], axis=1)

    potential = potential_ri_inv(inv_rAB, inv_rA1, inv_rB1, inv_rA2, inv_rB2, inv_r12)
    H_1_ij = coef_vector_1 @ Fij + potential[:, None]
    H_1_ji = coef_vector_1 @ Fji + potential[:, None]
    H_alpha_ij = coef_vector_alpha @ Fij
    H_alpha_ji = coef_vector_alpha @ Fji
    H_beta_ij = coef_vector_beta @ Fij
    H_beta_ji = coef_vector_beta @ Fji
    H_alpha2 = c_alpha2_1[:, None]
    H_alphabeta = c_alphabeta_1[:, None]
    H_beta2 = c_beta2_1

    return H_1_ij, H_1_ji, H_alpha_ij, H_alpha_ji, H_alpha2, H_alphabeta, H_beta_ij, H_beta_ji, H_beta2

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
    else:
        a_idx = zeros_like(h_idx)
        b_idx = zeros_like(h_idx)
    return h_idx, k_idx, n_idx, m_idx, i_idx, j_idx, a_idx, b_idx

def calc_AB(x1, y1, x2, y2, z2, rAB, s, s1, s2, mu1, mu2, w1, w2, W, coords, basis_idx, delta, M1M, M_inv, Fij, Fji, X=None):

        rA1 = np.sqrt((x1 + rAB/2) ** 2 + y1 ** 2)
        rB1 = np.sqrt((x1 - rAB/2) ** 2 + y1 ** 2)  # vectorized distances

        rA2 = np.sqrt((x2 + rAB / 2) ** 2 + y2 ** 2 + z2 ** 2)
        rB2 = np.sqrt((x2 - rAB / 2) ** 2 + y2 ** 2 + z2 ** 2)

        matSize = basis_idx[:, 0].size
        h_idx, k_idx, n_idx, m_idx, i_idx, j_idx, a_idx, b_idx = expand_idx(basis_idx, coords)
        # k_min = np.min(k_idx)
        k_max = np.max(k_idx)
        # n_min = np.min(n_idx)
        n_max = np.max(n_idx)
        # m_min = np.min(m_idx)
        m_max = np.max(m_idx)
        # ij_min = np.min(i_idx)
        ij_max = np.max(i_idx)
        a_min = np.min(a_idx)
        a_max = np.max(a_idx)

        # r12-dependent quantities
        dx = x1[:, None] - x2[None, :]
        dy = y1[:, None] - y2[None, :]
        dz = 0.0 - z2[None, :]
        r12 = np.sqrt(dx * dx + dy * dy + dz * dz).ravel()  # vector of r12 values for all e1, e2 positions
        r12 = np.maximum(r12, 10 ** (-8))
        r12_p = power_table(r12, k_max)

        P1 = rA1.size
        P2 = rA2.size
        P = P1 * P2

        if coords == "stmu":
            H_1_ij, H_1_ji, H_alpha_ij, H_alpha_ji, H_alpha2, H_alphabeta, H_beta_ij, H_beta_ji, H_beta2 = calc_H_alphabeta_stmu(Fij, Fji, rAB, rA1, rB1, rA2, rB2, r12, M_inv, M1M, s1, s2, s, mu1, mu2, delta)
            mu1_p = power_table(mu1, ij_max)
            mu2_p = power_table(mu2, ij_max)
            mu1_p = np.repeat(mu1_p, P2, axis=0)  # shape (P, Npow)  # Tile/repeat single-electron arrays to match the entire sample point array
            mu2_p = np.tile(mu2_p, (P1, 1))
            t = (s1 - s2) / rAB * 0.5

            # # Assemble basis functions from power matrices
            # B = np.ones((r12_p.shape[0], matSize), dtype=np.float64) * (rAB ** h_idxs ** n_idx * t ** m_idx)[None, :]


            if np.ndim(s) == 0:
                B = np.ones((r12_p.shape[0], matSize), dtype=np.float64) * (rAB ** h_idx * s ** n_idx * t ** m_idx)[None, :]
            else: # just for plotting
                s_p = power_table(s, n_max)
                t_p = power_table(t, m_max)
                B = np.ones((r12_p.shape[0], matSize), dtype=np.float64) * (rAB ** h_idx)[None, :]
                B *= s_p[:, n_idx] * t_p[:, m_idx]

            B *= r12_p[:, k_idx]
            B *= np.exp(-delta*r12)[:, None]

            mu_part_ij = mu1_p[:, i_idx] * mu2_p[:, j_idx]
            mu_part_ji = mu1_p[:, j_idx] * mu2_p[:, i_idx]

            B_ij = B * mu_part_ij
            B_ji = B * mu_part_ji

            B = B_ij + B_ji

            A_1 = H_1_ij * B_ij + H_1_ji * B_ji
            A_alpha = H_alpha_ij * B_ij + H_alpha_ji * B_ji
            A_beta = H_beta_ij * B_ij + H_beta_ji * B_ji
            A_alpha2 = H_alpha2 * (B_ij + B_ji)
            A_alphabeta = H_alphabeta * (B_ij + B_ji)
            # A_beta2 = H_beta2 * (B_ij + B_ji)

        elif coords == "s12mu":
            H_1_12, H_1_21, H_alpha_12, H_alpha_21, H_alpha2, H_alphabeta, H_beta_12, H_beta_21, H_beta2 = calc_H_alphabeta_s12mu(Fij, Fji, rAB, rA1, rB1, rA2, rB2, r12, M_inv, M1M, s1, s2, mu1, mu2, delta)

            mu1_p = power_table(mu1, ij_max)
            mu2_p = power_table(mu2, ij_max)
            mu1_p = np.repeat(mu1_p, P2, axis=0)  # shape (P, Npow)  # Tile/repeat single-electron arrays to match the entire sample point array
            mu2_p = np.tile(mu2_p, (P1, 1))

            # Assemble basis functions from power matrices
            B = np.broadcast_to(rAB**h_idx, (P, matSize)).copy()
            B *= r12_p[:, k_idx]
            B *= np.exp(-delta*r12)[:, None]

            if np.ndim(s1) == 0:
                part_12 = mu1_p[:, i_idx] * mu2_p[:, j_idx] * s1 ** n_idx * s2 ** m_idx
                part_21 = mu1_p[:, j_idx] * mu2_p[:, i_idx] * s1 ** m_idx * s2 ** n_idx
            else: # just for plotting
                part_12 = mu1_p[:, i_idx] * mu2_p[:, j_idx] * (s1[:, None] ** n_idx[None, :]) * s2 ** m_idx
                part_21 = mu1_p[:, j_idx] * mu2_p[:, i_idx] * (s1[:, None] ** m_idx[None, :]) * s2 ** n_idx

            B_12 = B * part_12
            B_21 = B * part_21

            B = B_12 + B_21

            A_1 = H_1_12 * B_12 + H_1_21 * B_21
            A_alpha = H_alpha_12 * B_12 + H_alpha_21 * B_21
            A_beta = H_beta_12 * B_12 + H_beta_21 * B_21
            A_alpha2 = H_alpha2 * B
            A_alphabeta = H_alphabeta * B
            # A_beta2 = H_beta2 * B

        elif coords == "rij":
            H_1, H_alpha, H_alpha2, H_alphabeta, H_beta, H_beta2, inv_rA, inv_rB = calc_H_alphabeta_rij(Fij, rAB, rA1, rB1, rA2, rB2, r12, M_inv, M1M, delta)

            rA1_p = power_table(rA1, n_max)
            rB1_p = power_table(rB1, n_max)
            rA2_p = power_table(rA2, n_max)
            rB2_p = power_table(rB2, n_max)
            inv_rA_p  = power_table(inv_rA, int(a_max), int(a_min))  # already full size P
            inv_rB_p  = power_table(inv_rB, int(a_max), int(a_min))

            rA1_p = np.repeat(rA1_p, P2, axis=0)  # shape (P, Npow)
            rB1_p = np.repeat(rB1_p, P2, axis=0)  # shape (P, Npow)
            rA2_p = np.tile(rA2_p, (P1, 1))
            rB2_p = np.tile(rB2_p, (P1, 1))

            # Assemble basis functions from power matrices
            B = np.broadcast_to(rAB**h_idx, (P, matSize)).copy() # rA1^n*rB1^m*rA2^i*rB2^j*r12^k*rAB^h/rA^a/rB^b
            B *= rA1_p[:, n_idx]
            B *= rA2_p[:, i_idx]
            B *= rB1_p[:, m_idx]
            B *= rB2_p[:, j_idx]
            B *= r12_p[:, k_idx]
            B *= inv_rA_p[:, a_idx]
            B *= inv_rB_p[:, b_idx]
            B *= np.exp(-delta*r12)[:, None]

            sqrtW = np.sqrt((W) * (w1[:, None] * w2[None, :]).ravel())  # (P,)
            B *= sqrtW[:, None]

            # Apply Hamiltonian prefactors
            A_1 = H_1 * B
            A_alpha = H_alpha * B
            A_alpha2 = H_alpha2 * B
            A_alphabeta = H_alphabeta * B
            A_beta = H_beta * B

            # Apply linear combinations to assemble symmetric basis
            # B, A_1, A_alpha, A_alpha2, A_alphabeta, A_beta, A_beta2 = combine_fcts(B, A_1, A_alpha, A_alpha2, A_alphabeta, A_beta, A_beta2, groups)
            B = B @ X
            A_1 = A_1 @ X
            A_alpha = A_alpha @ X
            A_alpha2 = A_alpha2 @ X
            A_alphabeta = A_alphabeta @ X
            A_beta = A_beta @ X
            # A_beta2 = A_beta2 @ X

        # if abs(s2-plot_s2_target) < 1e-8:
        #     plot_chunks = update_plot_chunks(plot_chunks, mu2, B, P1, P2, x1, y1, phi2, plot_phi_target, A_1, A_alpha, A_alpha2, A_alphabeta, A_beta, A_beta2)

        return B, A_1, A_alpha, A_beta, A_alphabeta, A_alpha2, P


def calc_F_ne(x1, y1, rAB, coords, basis_idx):
    rA1 = np.sqrt((x1 + rAB / 2) ** 2 + y1 ** 2)
    rB1 = np.sqrt((x1 - rAB / 2) ** 2 + y1 ** 2)  # vectorized distances
    s1 = rA1 + rB1
    mu1 = (rA1 - rB1) / rAB

    s2 = rAB
    mu2 = -1  #  = (rA2-rB2)/rAB -> rAB = rB2, rA2 = 0, e2 in A -> r12 = rA1
    r12 = rA1

    matSize = basis_idx[:, 0].size
    h_idx, k_idx, n_idx, m_idx, i_idx, j_idx, a_idx, b_idx = expand_idx(basis_idx, coords)
    i_max = np.max(i_idx)
    n_min = np.min(n_idx)
    n_max = np.max(n_idx)
    k_max = np.max(k_idx)

    P = rA1.size

    if coords == "s12mu":
        r12_p = power_table(r12, k_max)
        s1_p = power_table(s1, n_max, n_min)
        mu1_p = power_table(mu1, i_max)
        ni_factors_ne_1 = (n_idx + i_idx)/rAB
        ni_factors_ne_alpha = -1

        B = np.broadcast_to(rAB ** h_idx, (P, matSize)).copy()
        B *= r12_p[:, k_idx]
        # B *= np.exp(-delta * r12)[:, None] # (cancels out anyway)

        part_12 = mu1_p[:, i_idx] * mu2 ** j_idx * s1_p[:, n_idx] * s2 ** m_idx
        part_21 = mu1_p[:, j_idx] * mu2 ** i_idx * s1_p[:, m_idx] * s2 ** n_idx

        B = B * (part_12 + part_21)
        F_ne_1 = B * ni_factors_ne_1
        F_ne_alpha = B * ni_factors_ne_alpha

    else:
        print(f"Coordinate system {coords} not implemented for cusp functions")
        return None, None, None

    return B, F_ne_1, F_ne_alpha  # (combine with chosen alpha value later)

def calc_F_ee(x1, y1, rAB, coords, basis_idx, delta):

    rA1 = np.sqrt((x1 + rAB/2) ** 2 + y1 ** 2)
    rB1 = np.sqrt((x1 - rAB/2) ** 2 + y1 ** 2)  # vectorized distances
    s1 = rA1 + rB1
    mu1 = (rA1 - rB1)/rAB

    matSize = basis_idx[:, 0].size
    h_idx, k_idx, n_idx, m_idx, i_idx, j_idx, a_idx, b_idx = expand_idx(basis_idx, coords)
    ij_max = np.max(i_idx + j_idx)
    nm_min = np.min(n_idx+m_idx)
    nm_max = np.max(n_idx+m_idx)

    P = rA1.size

    if coords == "s12mu":
        s1_p = power_table(s1, nm_max, nm_min)
        mu1_p = power_table(mu1, ij_max)
        k_factors_ee = np.where(k_idx == 1, 1.0, np.where(k_idx == 0, -delta, 0.0))
        k_factors_B = np.where(k_idx == 0, 1.0, 0.0)  # only non-vanishing parts of B at r12=0

        # Assemble basis functions from power matrices
        B = np.broadcast_to(rAB**h_idx, (P, matSize)).copy()
        B *= 2 * mu1_p[:, i_idx+j_idx] * s1_p[:, n_idx+m_idx]  # mu1^i*mu2^j*s1^n*s2^m+mu1^j*mu2^i*s1^m*s2^n, but mu1=mu2 and s1=s2
        F_ee = B * k_factors_ee[None, :] # compute diff(B, r12) at r12=0
        B *= k_factors_B  # set r12 = 0 in B
    else:
        print(f"Coordinate system {coords} not implemented for cusp functions")
        return None, None

    return B, F_ee

