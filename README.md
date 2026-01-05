# IR-MambaAD

<p align="center">
  <img src="./doc/overview.jpg" width="900" alt="Overview">
</p>

```bash
git clone <your-repo-url> IR-MambaAD
cd IR-MambaAD
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy scikit-learn imageio matplotlib
python run_quick_model_test.py
````

```
IR-MambaAD/
├─ datasets/
│  ├─ __init__.py
│  ├─ InfraredAD.py        # Generic infrared dataset loader (radiometric-aware)
│  ├─ MulSenAD.py          # Multi-sensor (thermal + RGB) loader
│  ├─ ThermoSolarPV.py     # PV thermal dataset loader with optional annotations
│  ├─ transforms.py        # Dict-style data transforms (joint crop/flip, noise, normalize)
│  ├─ dataloaders.py       # Collate & DataLoader helpers
│  └─ utils.py             # Compute mean/std, visualize samples
├─ model/
│  ├─ __init__.py
│  ├─ ops.py               # Haar DWT / IDWT helpers
│  ├─ mwfm.py              # Multi-scale Wavelet Feature Modulation (MWFM)
│  ├─ encoder.py           # ResNet34-based encoder + Half-FPN
│  ├─ mamba_ssm.py         # Prototype SelectiveSSM (naive, correct)
│  ├─ hpg_mamba.py         # HPG-Mamba stage (projects tokens, predicts alpha, runs SSM)
│  ├─ decoder.py           # Cascaded decoder chaining HPG-Mamba stages
│  ├─ model.py             # IRMambaAD top-level wiring MWFM->Encoder->Decoder
│  └─ losses.py            # Reconstruction map loss helper
├─ train.py                # Training script (recon mode implemented; pretrain stub provided)
├─ test.py                 # Evaluation/inference script (saves maps, computes AUROC/AP)
├─ run_quick_model_test.py # Quick forward-pass sanity check
├─ README.md               # This file
└─ doc/overview.jpg        # Optional overview figure (placeholder)
```

Download datasets: MulSenAD, ThermoSolarPV, InfraredAD.

```
dataset_root/
  images/
    train/
    val/        
    test/
  masks/        
    train/
    val/
    test/
```

Training:

```bash
python train.py \
  --data-root /path/to/dataset \
  --exp-dir ./exp/run1 \
  --mode recon \
  --epochs 50 \
  --batch-size 8 \
  --lr 3e-4 \
  --device cuda
```

Important flags: `--data-root` (dataset root), `--exp-dir` (checkpoints/logs), `--mode` (`recon` or `pretrain` stub), `--resume` (path to checkpoint). The training script uses AMP on CUDA automatically.

Testing / evaluation:

```bash
python test.py \
  --data-root /path/to/dataset \
  --checkpoint ./exp/run1/best_checkpoint.pth \
  --out-dir ./results \
  --save-maps
```


