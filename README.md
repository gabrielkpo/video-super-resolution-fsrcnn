# Video super-resolution with FSRCNN

Research project on **upscaling low-resolution video frames with small, fast models**, in the
spirit of "Green AI": send fewer pixels over the network and rebuild the rest on the device.
The project compares classical interpolation, frequency-domain methods, denoising, sparse coding
and a lightweight CNN (FSRCNN), and adds a temporal block filter to skip unchanged regions.

**Authors:** Gabriel Kpodoh and Thomas Delacretaz-Martinon (equal contribution).
**Supervisor:** Thomas Oberlin. IATI research project (image processing and AI), ENSEEIHT.

## Idea

Streaming video is energy hungry. If a client can rebuild a x4 upscale cheaply, the network only
carries a quarter of the pixels per axis-pair. The project is worth it only if

```
E_transport(HD) > E_transport(SD) + E_inference(device)
```

so every method is measured on quality (PSNR, SSIM) **and** inference time.

## What is in this repository

| Path | Content |
|---|---|
| `methods/level1_interpolation.py` | Bilinear, bicubic, Lanczos, nearest (baselines) |
| `methods/level2_frequency.py` | DWT and FFT zero-padding upscaling |
| `methods/level3_denoising.py` | Bilateral, NLM, BM3D as post-processing |
| `methods/level4_ml_classic.py` | Sparse-coding super-resolution (dictionary + OMP) |
| `methods/level6_cnn.py` | FSRCNN (Dong et al., ECCV 2016) model and loader |
| `train/` | FSRCNN fine-tuning (DIV2K to Vimeo-90K), sparse-coding training, Vimeo benchmark, checkpoints |
| `workflow/` | SAD + FSRCNN: temporal filter on 16x16 blocks, FSRCNN only on blocks that changed |
| `pipeline/` | Frame extraction, degradation, metrics, video reconstruction |
| `notebook_comparatif.ipynb` | Side-by-side comparison of the methods |
| `config.py` | Paths and hyper-parameters |

Not implemented (listed in the original plan): SRCNN and ESPCN, post-training quantisation and
pruning.

## Results recorded so far

Measured on 30 Vimeo-90K test images, x4, PSNR and SSIM on the Y channel
(source: outputs saved in `notebook_comparatif.ipynb`):

| Method | PSNR Y (dB) | SSIM Y | Time (ms/frame) |
|---|---|---|---|
| Bilinear | 24.74 | 0.8195 | 0.23 |
| Bicubic | 25.75 | 0.8498 | 0.36 |
| Lanczos | 25.86 | 0.8500 | 0.83 |
| Bicubic + bilateral | 24.53 | 0.7623 | 0.87 |
| Bicubic + NLM | 24.76 | 0.7864 | 34.17 |

Denoising as post-processing **lowers** quality here (-1.2 and -1.0 dB versus bicubic).

**FSRCNN and SAD+FSRCNN numbers are not recorded in this repository.** The scripts produce them
(`train/benchmark_vimeo.py`, `workflow/compare_sad_vs_fsrcnn.py`) but the datasets live on an
external disk and the results folder is git-ignored. They will be added here once re-run, rather
than quoted from memory.

## Reproduce

```bash
pip install -r requirements.txt

# Level 1 baselines on an image
python methods/level1_interpolation.py

# FSRCNN vs bicubic on the Vimeo-90K test split (needs the dataset, see below)
python train/benchmark_vimeo.py --max-clips 100

# SAD + FSRCNN vs FSRCNN alone
python -m workflow.compare_sad_vs_fsrcnn
```

Paths to the datasets are set at the top of `train/benchmark_vimeo.py` and `config.py`
(an external disk in the original setup); adapt them to your machine.

## Data and checkpoints

- Datasets are **not** included: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and
  [Vimeo-90K](http://toflow.csail.mit.edu/) have their own terms of use (research use).
- Small trained weights are committed: `train/fsrcnn_div2k.ckpt` (pre-trained on DIV2K),
  `train/fsrcnn_vimeo_best.ckpt` (fine-tuned on Vimeo-90K), `train/sparse_sr.joblib`.

## Development

```bash
pip install pytest ruff
ruff check .
pytest
```

Pull requests need a Conventional Commits title; CI runs ruff, pytest and a secret scan.

## References

- Dong et al., *Accelerating the Super-Resolution Convolutional Neural Network* (FSRCNN, ECCV 2016)
- Dong et al., *Learning a Deep Convolutional Network for Image Super-Resolution* (SRCNN, 2014)
- Xue et al., *Video Enhancement with Task-Oriented Flow* (Vimeo-90K, IJCV 2019)
- Agustsson and Timofte, *NTIRE 2017 Challenge on Single Image Super-Resolution* (DIV2K)

## License

MIT, see [LICENSE](LICENSE).
