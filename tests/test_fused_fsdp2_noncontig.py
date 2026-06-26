"""
T5 — exercise the non-contiguous local-shard copy-back branch of the fused
FSDP2 fix in gefen.py::_step_automatic.

Two cases, both fused=True on CUDA:

A. Plain non-contiguous param (p_local = p, is_contiguous() == False) vs an
   identical contiguous param. Same grad/seed -> identical update. Proves the
   copy-back branch produces the same result as the direct branch and that the
   update actually lands on the param's storage.

B. Real DTensor whose to_local() shard is *non-contiguous* (built via
   DTensor.from_local on a strided slice). Single-rank process group. Runs a
   full optimizer.step() and asserts the local shard storage was updated to
   match a plain-tensor oracle. Covers grad.to_local() unwrap + p.to_local()
   non-contiguous + copy-back, end to end.
"""
import os
import torch

from gefen import Gefen

DEV = "cuda"
LR = 5e-5


def make_noncontig(values: torch.Tensor) -> torch.Tensor:
    """Return a non-contiguous tensor holding `values` (shape preserved)."""
    m, n = values.shape
    parent = torch.empty(m, 2 * n, device=values.device, dtype=values.dtype)
    view = parent[:, :n]  # stride (2n, 1) -> non-contiguous
    view.copy_(values)
    assert not view.is_contiguous()
    return view


def case_A():
    torch.manual_seed(0)
    M, N = 64, 96  # numel 6144; period prediction will pick a divisor
    init = torch.randn(M, N, device=DEV, dtype=torch.bfloat16)
    grad = torch.randn(M, N, device=DEV, dtype=torch.bfloat16) * 0.01

    # contiguous reference param
    p_c = torch.nn.Parameter(init.clone())
    # non-contiguous param with identical values (leaf Parameter, non-contig storage)
    p_nc = torch.nn.Parameter(make_noncontig(init.clone()).detach())

    before_nc = p_nc.detach().clone()

    opt_c = Gefen([p_c], lr=LR, fused=True)
    opt_nc = Gefen([p_nc], lr=LR, fused=True)

    # identical grads -> identical learned codebook/period -> identical update.
    # opt.step() learns the codebook on the first step (required for fused path).
    p_c.grad = grad.clone()
    p_nc.grad = grad.clone()
    opt_c.step()
    opt_nc.step()

    changed = not torch.equal(p_nc.detach(), before_nc)
    nc_contig = p_nc.is_contiguous()
    max_abs = (p_c.detach().float() - p_nc.detach().float()).abs().max().item()
    ok = changed and (max_abs == 0.0)
    print(f"[A] p_nc.is_contiguous()={nc_contig} (expect False -> copy-back branch)")
    print(f"[A] param changed={changed}  max|contig-noncontig|={max_abs:.3e}")
    print(f"[A] {'PASS' if ok else 'FAIL'}")
    return ok and (not nc_contig)


def case_B():
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Shard, init_device_mesh

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    mesh = init_device_mesh("cuda", (1,))

    torch.manual_seed(1)
    M, N = 48, 64  # numel 3072
    init = torch.randn(M, N, device=DEV, dtype=torch.bfloat16)
    grad = torch.randn(M, N, device=DEV, dtype=torch.bfloat16) * 0.01

    # --- DTensor param with a NON-contiguous local shard ---
    local_nc = make_noncontig(init.clone())
    p = torch.nn.Parameter(
        DTensor.from_local(local_nc, mesh, [Shard(0)], run_check=False)
    )
    grad_dt = DTensor.from_local(grad.clone(), mesh, [Shard(0)], run_check=False)
    p.grad = grad_dt

    opt = Gefen([p], lr=LR, fused=True)
    opt.step()

    updated_local = p.to_local()

    # --- plain-tensor oracle: same single fused step on a contiguous param ---
    p_ref = torch.nn.Parameter(init.clone())
    opt_ref = Gefen([p_ref], lr=LR, fused=True)
    p_ref.grad = grad.clone()
    opt_ref.step()

    changed = not torch.equal(updated_local.float(), init.float())
    local_was_noncontig = not local_nc.is_contiguous()
    max_abs = (updated_local.float() - p_ref.detach().float()).abs().max().item()
    ok = changed and (max_abs == 0.0)
    print(f"[B] local shard non-contiguous on entry={local_was_noncontig}")
    print(f"[B] DTensor param changed={changed}  max|dtensor-oracle|={max_abs:.3e}")
    print(f"[B] {'PASS' if ok else 'FAIL'}")

    dist.destroy_process_group()
    return ok and local_was_noncontig


if __name__ == "__main__":
    a = case_A()
    b = case_B()
    print("\nT5 OVERALL:", "PASS" if (a and b) else "FAIL")
    raise SystemExit(0 if (a and b) else 1)
