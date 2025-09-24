import torch


def gaussian_kernel(x, y, sigma):
    x = x.unsqueeze(1)  # [B1, 1, D]
    y = y.unsqueeze(0)  # [1, B2, D]
    return torch.exp(-((x - y) ** 2).sum(dim=2) / (2 * sigma ** 2))

def estimate_sigma_median_heuristic(X, Y, num_samples=1000):
    Z = torch.cat([X, Y], dim=0)
    n = Z.shape[0]
    idx = torch.randperm(n)[:min(num_samples, n)]
    sample = Z[idx]
    dists = torch.cdist(sample, sample)
    median = torch.median(dists[dists > 0])  # exclude zero (self-distances)
    return median.item()

def batched_biased_mmd(X, Y, batch_size=5000, sigma=None):
    assert X.shape[1] == Y.shape[1], "Dim mismatch"
    n, m = X.shape[0], Y.shape[0]

    if sigma is None:
        sigma = estimate_sigma_median_heuristic(X, Y)

    # 1. XX term
    xx_sum = 0.0
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        for j in range(0, n, batch_size):
            xbp = X[j:j+batch_size]
            K = gaussian_kernel(xb, xbp, sigma)
            xx_sum += K.sum().item()

    # 2. YY term (assume small enough to do at once)
    K_yy = gaussian_kernel(Y, Y, sigma)
    yy_sum = K_yy.sum().item()

    # 3. XY term
    xy_sum = 0.0
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        K = gaussian_kernel(xb, Y, sigma)
        xy_sum += K.sum().item()

    mmd2 = (xx_sum / (n * n)) + (yy_sum / (m * m)) - (2 * xy_sum / (n * m))
    return mmd2



# Weighted mmd for MF-ABC
@torch.no_grad()
def batched_weighted_mmd(
    X, Y, wx=None, wy=None, batch_size=5000, sigma=None, unbiased=True, normalize_weights=True
):
    """
    Weighted MMD^2 with RBF kernel (optionally a kernel mixture via sigma=list/tuple/tensor).
    - X: [n,d], Y: [m,d]
    - wx: [n], wy: [m] (importance weights). If None, uses uniform.
    - unbiased=True removes diagonal contributions in Kxx/Kyy (recommended if you have duplicates).
    - Returns a Python float.
    """
    assert X.shape[1] == Y.shape[1], "Dim mismatch"
    device = X.device
    dtype = X.dtype
    n, m = X.shape[0], Y.shape[0]

    # --- weights ---
    if wx is None:
        wx = torch.full((n,), 1.0/n, device=device, dtype=dtype)
    if wy is None:
        wy = torch.full((m,), 1.0/m, device=device, dtype=dtype)

    if normalize_weights:
        wx = wx / (wx.sum() + 1e-12)
        wy = wy / (wy.sum() + 1e-12)

    # Effective normalizers for unbiased estimate
    # sum_{i != j} wx_i wx_j = (sum wx)^2 - sum wx^2 ; since sum wx = 1 if normalized, this is 1 - sum wx^2
    norm_x = (1.0 - (wx**2).sum()).clamp(min=1e-12) if unbiased else torch.tensor(1.0, device=device, dtype=dtype)
    norm_y = (1.0 - (wy**2).sum()).clamp(min=1e-12) if unbiased else torch.tensor(1.0, device=device, dtype=dtype)

    # --- bandwidth ---
    if sigma is None:
        sigma = estimate_sigma_median_heuristic(X, Y)
    # You can also pass a list/tuple of bandwidths for a kernel mixture:
    # e.g., sigma = [0.5*sigma, sigma, 2*sigma, 4*sigma]

    # --- XX term ---
    Zx = torch.zeros((), device=device, dtype=dtype)
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        wbx = wx[i:i+batch_size]
        for j in range(0, n, batch_size):
            xbp = X[j:j+batch_size]
            wbp = wx[j:j+batch_size]
            K = gaussian_kernel(xb, xbp, sigma)  # [bi, bj]
            if unbiased and (i == j):
                # remove self-similarity terms
                K = K.clone()
                diag_len = min(K.shape[0], K.shape[1])
                K.view(-1)[:diag_len*(K.shape[1]+1):K.shape[1]+1] = 0.0
            W = wbx[:, None] * wbp[None, :]
            Zx += (K * W).sum()

    # --- YY term ---
    Zy = torch.zeros((), device=device, dtype=dtype)
    for i in range(0, m, batch_size):
        yb = Y[i:i+batch_size]
        wby = wy[i:i+batch_size]
        for j in range(0, m, batch_size):
            ybp = Y[j:j+batch_size]
            wbpy = wy[j:j+batch_size]
            K = gaussian_kernel(yb, ybp, sigma)
            if unbiased and (i == j):
                K = K.clone()
                diag_len = min(K.shape[0], K.shape[1])
                K.view(-1)[:diag_len*(K.shape[1]+1):K.shape[1]+1] = 0.0
            W = wby[:, None] * wbpy[None, :]
            Zy += (K * W).sum()

    # --- XY term ---
    Zxy = torch.zeros((), device=device, dtype=dtype)
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        wbx = wx[i:i+batch_size]
        for j in range(0, m, batch_size):
            yb = Y[j:j+batch_size]
            wby = wy[j:j+batch_size]
            K = gaussian_kernel(xb, yb, sigma)
            W = wbx[:, None] * wby[None, :]
            Zxy += (K * W).sum()

    mmd2 = (Zx / norm_x) + (Zy / norm_y) - 2.0 * Zxy
    # Numerical guard
    return float(mmd2.clamp_min(0.0).item())