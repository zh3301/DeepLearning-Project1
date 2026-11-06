# Deep Learning Project 1 — Efficient ResNet for CIFAR-10

NYU Tandon School of Engineering · Deep Learning · Fall 2026

This project compares custom Residual Neural Networks (ResNets) trained **from scratch** for CIFAR-10 image classification, under a strict budget of **5 million trainable parameters**. The final selected submission was **Experiment 3 (Deeper ResNet)**.

## Final result

- **Best public Kaggle accuracy:** **82.50%** (Experiment 3)
- **Final model parameters:** **3,958,346** (under 5 million)
- **Final model's best validation accuracy:** **93.26%** (epoch 46 of 50)
- **Training data:** permitted CIFAR-10 training images only (45,000 train / 5,000 validation)
- **Competition:** [NYU Tandon Deep Learning F26](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl)

The public leaderboard score is reported here; it is **not** the private leaderboard score. The validation and competition scores are calculated on different image sets.

## Project files

| File | Purpose |
| --- | --- |
| `Project1_Exp1_Baseline.ipynb` | Original baseline training, validation, predictions, and TTA exploration (Google Colab) |
| `Project1_Exp2_Augmentation.ipynb` | Training and evaluation with added color augmentation and random erasing (Kaggle Notebook) |
| `Project1_Exp3_DeeperResNet.ipynb` | Final deeper-architecture experiment, training, evaluation, and prediction generation (Kaggle Notebook) |
| `results.csv` | Comparison of validation and public Kaggle results |
| `requirements.txt` | Python libraries used by the notebooks |

**Repository contents:** The Exp2 and Exp3 Kaggle notebooks were exported as `.ipynb` files. The archive files `Experiment2_backup.zip` and `Experiment3_backup.zip` are kept privately as checkpoints and results, and are not included in this public repository.

## Dataset

The permitted dataset contains 50,000 labeled CIFAR-10 training images (32 × 32 RGB, 10 categories). We use the same reproducible 45,000/5,000 split in all experiments, generated with NumPy seed 42. The separate competition evaluation set contains 2,000 unlabeled images.

For Kaggle notebooks, attach the **NYU Tandon Deep Learning F26** competition input. The notebooks expect files like:

```text
/kaggle/input/competitions/cifar-10-res-net-architecture-challenge-dl/
    train_images/
    test_images/
    train_labels.csv
    test.csv
    sample_submission.csv
```

Experiment 1 was developed in Google Colab using the official CIFAR-10 training dataset and verified that its ordering and labels matched the Kaggle-provided training data. Its original notebook also contains Colab-specific Google Drive paths, Kaggle authentication via Colab Secrets, and exploratory setup cells. It is retained as an experiment log rather than a one-click Kaggle notebook.

No external training datasets or pretrained model weights were used.

## Model architecture

Each model uses a 3 × 3 convolutional stem, residual `BasicBlock`s with two 3 × 3 convolutions per block and identity/projection skip connections, three residual stages with channel widths **64 → 128 → 256**, global average pooling, and a 10-class linear classifier.

| Model | Residual blocks by stage | Trainable parameters |
| --- | --- | ---: |
| Experiment 1 / 2 | (2, 2, 2) | 2,777,674 |
| Experiment 3 (final) | **(2, 2, 3)** | **3,958,346** |

All models satisfy the 5,000,000-parameter limit.

## Training settings

| Setting | Value |
| --- | --- |
| Train / validation | 45,000 / 5,000 |
| Training epochs | 50 |
| Batch size (train / validation) | 128 / 256 |
| Loss | Cross-entropy |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0001 |
| LR scheduler | CosineAnnealingLR (`T_max=50`) |
| Precision | PyTorch automatic mixed precision on CUDA |
| Initialization | From scratch; no pretrained weights |

**Experiment 1:** RandomCrop(32, padding=4), RandomHorizontalFlip, tensor conversion, and normalization using CIFAR-10 training-set mean/std.

**Experiments 2 and 3:** same transforms plus ColorJitter (brightness/contrast/saturation = 0.2; hue = 0.04; applied with probability 0.8) and RandomErasing (probability 0.15, scale = 0.02–0.12).

**TTA trial:** for an Experiment 1 prediction-only comparison, average logits of original and horizontally flipped images; this did not improve the public Kaggle score.

## Experimental results

| Experiment | Best validation accuracy | Public Kaggle accuracy |
| --- | ---: | ---: |
| Exp1 Baseline | 92.54% | 80.90% |
| Exp1 + horizontal-flip TTA | 92.90% | 80.70% |
| Exp2 Augmentation | **93.38%** | 82.00% |
| **Exp3 Deeper ResNet (final)** | 93.26% | **82.50%** |

Exp2 achieved the highest internal validation accuracy, while Exp3 achieved the best public competition score. A higher validation score did not necessarily lead to a higher public leaderboard score. Hidden evaluation labels were never accessed; public leaderboard scores were used only as feedback for selecting among official submissions.

Detailed results are in [`results.csv`](results.csv). The original experiment backups also include per-epoch training histories, retained privately for verification.

## Reproduce the experiment

1. Open the corresponding `.ipynb` in Kaggle (Exp2 or Exp3) and attach the competition data under **Input**.
2. Enable a compatible GPU runtime (a T4 GPU was used in the experiments).
3. Run the notebook cells in order: data loading and transforms → model definition → training → test prediction. Training starts with randomly initialized weights.
4. The code creates the best validation checkpoint and logs training/validation accuracy per epoch. The Exp3 notebook writes these under `/kaggle/working/experiment3/`.
5. After training, run the prediction cell on `test_images` and generate `submission.csv` in the format `id,label` with a predicted integer class from 0 to 9 for each of the 2,000 test-image IDs.
6. Upload the resulting CSV to the competition using **Submit Prediction**. Public scores may vary between runs, and the private leaderboard is evaluated separately.

The final model is reproducible from the notebook source and permitted training data; **pretrained weight files are deliberately not distributed in this public repository**. The submitted model checkpoints and full training logs were saved separately for course staff verification if requested.

## References

1. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” *CVPR*, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
2. A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” 2009. [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
3. [PyTorch](https://pytorch.org/) and [Torchvision](https://pytorch.org/vision/stable/index.html) documentation for neural-network training and image transformations.

## Course submission

Include a link to this **public GitHub repository on the first page** of the project report. Submit the report (PDF, at most eight pages including figures, tables and citations) on NYU Brightspace according to the course deadline. Include all team members and the registered Kaggle team name in the report.
