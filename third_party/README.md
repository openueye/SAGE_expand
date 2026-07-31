# Local third-party dependencies

`diff-gaussian-rasterization-w-depth` is a tracked source dependency with its upstream attribution.

`SPNet` is intentionally not tracked. Its upstream repository does not provide a licence granting
redistribution, so each user must obtain and verify it locally at the locked revision:

```bash
git clone https://github.com/Wang-xjtu/SPNet.git third_party/SPNet
git -C third_party/SPNet checkout b836bd044517b33d3737094acd6a1f09c2362f04
```

SAGE requires this local clone for SPNet execution. Set `SAGE_SPNET_ROOT` only to use another verified clone.
