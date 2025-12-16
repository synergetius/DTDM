# DTDM



## Environment

Windows 11, WSL Ubuntu 22.04.05.

```bash
conda create -n dtdm-env python=3.10
conda activate dtdm-env
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install einops
pip install matplotlib
pip install pandas
```

