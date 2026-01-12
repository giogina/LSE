import numpy as np


def potential_ri(rAB, rA1, rB1, rA2, rB2, r12):
    return 1/rAB + 1/r12 - 1/rA1 - 1/rB1 - 1/rA2 - 1/rB2


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

def calc_Fij_stmu(h_idx, k_idx, n_idx, m_idx, i_idx, j_idx):

    h = h_idx.astype(np.float64)
    k = k_idx.astype(np.float64)
    n = n_idx.astype(np.float64)
    m = m_idx.astype(np.float64)
    i = i_idx.astype(np.float64)
    j = j_idx.astype(np.float64)

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
    cos1AB = (rA1_2 + rAB_2 - rB1_2) * inv_rA1 * inv_rAB * 0.5
    cos1BA = (rAB_2 + rB1_2 - rA1_2) * inv_rAB * inv_rB1 * 0.5
    cosA1B = (rA1_2 - rAB_2 + rB1_2) * inv_rA1 * inv_rB1 * 0.5
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
    cos2AB = (rA2_2 + rAB_2 - rB2_2) * inv_rA2 * inv_rAB * 0.5
    cos2BA = (rAB_2 + rB2_2 - rA2_2) * inv_rAB * inv_rB2 * 0.5
    cosA2B = (rA2_2 - rAB_2 + rB2_2) * inv_rA2 * inv_rB2 * 0.5
    c12 = M_inv * (cos2AB - cos2BA)

    # Mixed but scalar
    t = (s1 - s2) * inv_rAB * 0.5
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

    cos12A = (r12_2 - rA1_2 + rA2_2) * inv_r12 * inv_rA2 * 0.5
    cos12B = (r12_2 - rB1_2 + rB2_2) * inv_r12 * inv_rB2 * 0.5
    cos1A2 = (rA1_2 + rA2_2 - r12_2) * inv_rA1 * inv_rA2 * 0.5
    cos1B2 = (rB1_2 + rB2_2 - r12_2) * inv_rB1 * inv_rB2 * 0.5
    cos21A = (r12_2 + rA1_2 - rA2_2) * inv_r12 * inv_rA1 * 0.5
    cos21B = (r12_2 + rB1_2 - rB2_2) * inv_r12 * inv_rB1 * 0.5

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
                2 * M1M + c2 - c7) * inv_t_2 * inv_rAB_2 - M_inv * inv_rAB_2 + 0.5 * c9 * inv_rAB_2 * inv_t  # m^2
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