"""PyTorch re-implementation of the fixes in
"A Fix for Softmax and Cross-Entropy" (Barron, Gnanapragasam, Tschannen, arXiv:2610.31642).

Everything here is computed in whatever dtype the logits arrive in, so we can
study bf16 / fp32 / fp64 behaviour simply by casting logits before the loss.
"""
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Standard (baseline) versions -- what everyone uses today.
# ----------------------------------------------------------------------------
def standard_ce(z, t):
    """Hard-target cross-entropy, mean over batch. z: [B, K], t: [B]."""
    return F.cross_entropy(z, t)


def standard_soft_ce(z_s, z_t, T=1.0):
    """Soft-target (distillation) CE: -sum_k p_k log q_k with p = softmax(z_t / T)."""
    p = torch.softmax(z_t / T, dim=-1).detach()
    return -(p * torch.log_softmax(z_s / T, dim=-1)).sum(-1).mean()


# ----------------------------------------------------------------------------
# Fix 1: log-softmax / softmax  (paper Eq. 5)
# ----------------------------------------------------------------------------
def fixed_log_softmax(z, dim=-1):
    """The argmax slot is set to the *constant* 0, so exp(0)=1 never enters a
    sum where it would swamp tiny terms; the 1 is re-added via log1p."""
    imax = z.argmax(dim=dim, keepdim=True)
    is_max = torch.zeros_like(z, dtype=torch.bool).scatter_(dim, imax, True)
    zmax = z.gather(dim, imax)
    shifted = torch.where(is_max, torch.zeros_like(z), z - zmax)
    s = torch.where(is_max, torch.zeros_like(z), shifted.exp()).sum(dim, keepdim=True)
    return shifted - torch.log1p(s)          # = shifted - softplus(-m)


def fixed_softmax(z, dim=-1):
    return fixed_log_softmax(z, dim).exp()


# ----------------------------------------------------------------------------
# Fix 2: hard-target cross-entropy  (paper Eq. 6/7)
#   m = -LSE_{j != t}(z_j - z_t),  L = softplus(-m)
# ----------------------------------------------------------------------------
def fixed_ce_per_example(z, t):
    is_t = F.one_hot(t, z.shape[-1]).bool()
    zt = z.gather(-1, t[:, None])
    shifted = torch.where(is_t, torch.zeros_like(z), z - zt)
    b = shifted.max(-1, keepdim=True).values.detach()   # >= 0 (target slot is 0)
    s = torch.where(is_t, torch.zeros_like(z), (shifted - b).exp()).sum(-1, keepdim=True)
    # log(1 + e^b * s) = b + log1p(s + expm1(-b))
    return (b + torch.log1p(s + torch.expm1(-b))).squeeze(-1)


def fixed_ce(z, t):
    return fixed_ce_per_example(z, t).mean()


# ----------------------------------------------------------------------------
# Fix 3: soft-target cross-entropy  (paper Eq. 9) -- cures Softmax Cancellation
# ----------------------------------------------------------------------------
def _fixed_soft_grad(z_s, z_t):
    p = fixed_softmax(z_t)
    q = fixed_softmax(z_s)
    delta = z_s - z_t
    d = delta - (p * delta).sum(-1, keepdim=True)
    s = d - torch.log1p((p * torch.expm1(d)).sum(-1, keepdim=True))
    u = torch.where(s.abs() < 1, p * torch.expm1(s), q - p)
    g = u - q * u.sum(-1, keepdim=True)
    return g, p


def fixed_soft_ce(z_s, z_t, T=1.0):
    """Forward value is the usual soft CE; backward uses the exact-cancellation
    gradient g via the stop-gradient trick from the paper."""
    zs, zt = z_s / T, z_t.detach() / T
    g, p = _fixed_soft_grad(zs.detach(), zt)
    log_q = fixed_log_softmax(zs.detach())
    loss = -(p * log_q).sum(-1)
    surrogate = (g * (zs - zs.detach())).sum(-1)
    return (loss + surrogate).mean()


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def margin(z, t):
    """m = z_t - LSE_{j != t} z_j, computed in fp64."""
    z = z.double()
    is_t = F.one_hot(t, z.shape[-1]).bool()
    others = z.masked_fill(is_t, float("-inf"))
    return z.gather(-1, t[:, None]).squeeze(-1) - torch.logsumexp(others, -1)


# ----------------------------------------------------------------------------
# Reproduce the paper's worked numerical examples
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=True)
    print("== Sec 2: Softmax Collapse, z=[9,-9,-9], t=0, fp32 ==")
    for name, fn in [("standard", standard_ce), ("fixed", fixed_ce)]:
        z = torch.tensor([[9., -9., -9.]], requires_grad=True)
        L = fn(z, torch.tensor([0]))
        L.backward()
        print(f"  {name:9s} loss={L.item():.4e} grad={z.grad[0].tolist()}  sum={z.grad.sum().item():.3e}")
    print("  paper fixed: loss 3.046e-8, grad [-3.046e-8, 1.523e-8, 1.523e-8]")

    print("\n== Sec 3: Softmax Cancellation, z_T=[2,1,0], z_S=z_T+[-1e-7,1e-7,0], fp32 ==")
    zT = torch.tensor([[2., 1., 0.]])
    dz = torch.tensor([[-1e-7, 1e-7, 0.]])
    exact = (torch.softmax((zT + dz).double(), -1) - torch.softmax(zT.double(), -1))[0]  # fp32-rounded z_S
    print(f"  exact fp64 : {exact.tolist()}")
    for name, fn in [("standard", standard_soft_ce), ("fixed", fixed_soft_ce)]:
        zS = (zT + dz).clone().requires_grad_(True)
        fn(zS, zT).backward()
        print(f"  {name:9s}  : {zS.grad[0].tolist()}")
    print("  paper: standard [-5.960e-8, 1.452e-8, -1.123e-9], fixed [-4.596e-8, 4.144e-8, 4.513e-9]")

    print("\n== Identical student/teacher logits -> gradient should be exactly 0 ==")
    torch.manual_seed(0)
    for dt in [torch.bfloat16, torch.float32]:
        zT = (torch.randn(4096, 53) * 3).to(dt)
        for name, fn in [("standard", standard_soft_ce), ("fixed", fixed_soft_ce)]:
            zS = zT.clone().requires_grad_(True)
            fn(zS, zT).backward()
            print(f"  {str(dt):15s} {name:9s} max|grad| = {zS.grad.abs().max().item():.3e}")

    print("\n== Fixed CE agrees with standard CE where standard is fine ==")
    z = torch.randn(256, 31, dtype=torch.float64)
    t = torch.randint(0, 31, (256,))
    print("  max |diff| loss:", (fixed_ce_per_example(z, t) - F.cross_entropy(z, t, reduction='none')).abs().max().item())
    print("  max |diff| logsoftmax:", (fixed_log_softmax(z) - torch.log_softmax(z, -1)).abs().max().item())
