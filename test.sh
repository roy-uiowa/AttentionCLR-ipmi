cd /Users/troy3/myProjects/attentionCLR
python - <<'PY'
import itertools
import torch.nn as nn

def block(in_c, out_c, k):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, padding=k // 2), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, k, padding=k // 2), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )

def build(channels, k, pools, dense, num_classes=21):
    ch = [3] + list(channels)
    layers = []
    for i, (a, b) in enumerate(zip(ch[:-1], ch[1:])):
        layers.append(block(a, b, k))
        layers.append(nn.MaxPool2d(pools[i]))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(channels[-1], dense))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(dense, num_classes))
    return nn.Sequential(*layers)

def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

target = 303_437

variants = []
channel_sets = {
    "8/16/32/64":       [8, 16, 32, 64],
    "16/32/64/128":     [16, 32, 64, 128],
    "32/64/128/256":    [32, 64, 128, 256],
    "8/16/32/64/128":   [8, 16, 32, 64, 128],
    "16/32/64/128/256": [16, 32, 64, 128, 256],
    "8/16/32/64/128/256": [8, 16, 32, 64, 128, 256],
}
for ch_name, ch in channel_sets.items():
    n_blocks = len(ch)
    for k in (3, 5, 7):
        for dense in (256, 512, 1024):
            # pool sizes: try to reach 1x1 from 250 with n_blocks pools
            if n_blocks == 4:
                pool_opts = [(4, 4, 5, 3), (4, 4, 5, 1), (5, 5, 5, 2), (3, 3, 3, 3)]
            elif n_blocks == 5:
                pool_opts = [(4, 4, 4, 2, 1), (3, 3, 3, 3, 3)]
            else:  # 6
                pool_opts = [(3, 3, 3, 3, 2, 1), (4, 4, 4, 4, 1, 1)]
            for pools in pool_opts:
                try:
                    n = count(build(ch, k, pools, dense))
                    variants.append((abs(n - target), n, ch_name, f"k={k}", f"dense={dense}", str(pools)))
                except Exception:
                    pass

variants.sort()
print(f"{'|Δ|':>8} {'params':>9}  channels / kernel / dense / pools")
print(f"{'-'*70}")
for delta, n, ch_name, k, dense, pools in variants[:15]:
    print(f"{delta:>8d} {n:>9d}  {ch_name}  {k}  {dense}  {pools}")
print(f"\ntarget: {target}")
print(f"current: {count(build([8,16,32,64], 3, [4,4,5,3], 256))}")
PY
