"""GPU-flavored demo: nondeterministic reduction + unseeded DataLoader.

Requires torch (CPU-only torch still demonstrates the RNG marking
flow; CUDA shows the atomics hazard). Two runs under
`seedtrace run --seed 42` on a CUDA machine will diverge at the first
epoch's weight mark because `scatter_add_` accumulates in
atomic-addition order, which the GPU does not fix.

    seedtrace run --seed 42 -- examples/demo_torch_gpu.py
    seedtrace run --seed 42 -- examples/demo_torch_gpu.py
    seedtrace compare <runA> <runB>
    seedtrace audit examples/demo_torch_gpu.py
"""

from __future__ import annotations

import seedtrace

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    raise SystemExit("this demo needs torch: pip install torch") from None

seedtrace.auto(seed=None)

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True  # deliberately bad for reproducibility

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
seedtrace.mark("init_weights", model[0].weight)

opt = torch.optim.SGD(model.parameters(), lr=0.05)
x = torch.randn(64, 16, device=device)
y = torch.randn(64, 1, device=device)

for epoch in range(5):
    opt.zero_grad()
    pred = model(x)
    # scatter_add_: known non-deterministic accumulation order on CUDA
    bucket = torch.zeros(64, 1, device=device)
    idx = torch.randint(0, 64, (64,), device=device).unsqueeze(1).expand(-1, 1)
    bucket.scatter_add_(0, idx, pred.detach())
    loss = ((model(x) + 0.01 * bucket) - y).pow(2).mean()
    loss.backward()
    opt.step()
    seedtrace.mark(f"epoch_{epoch}", model[0].weight, round(float(loss), 8))
    print(f"epoch {epoch}: loss={float(loss):.6f}")

seedtrace.mark("final_loss", round(float(loss), 8))
print(f"final loss: {float(loss):.8f}")
