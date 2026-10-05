# FibroinESM

Homology-independent identification of insect silk proteins with a protein language model classifier.

FibroinESM is a fine-tuned protein language model classifier built on frozen ESM-2 embeddings (esm2_t33_650M_UR50D). A joint classification head predicts whether a protein is a silk protein and, for silk proteins, whether it is a heavy-chain fibroin, a light-chain fibroin or a P25-type protein. The model was trained on 1,976 heavy-chain, 45 light-chain and 51 P25 reference proteins against 41,216 non-silk proteins, with the first 20 transformer layers frozen, and was used to screen 121 insect proteomes.

This repository accompanies the study of webspinner silk evolution. If you use this code, please cite the associated publication (to be updated upon acceptance).

## Repository structure

```
FibroinESM/
├── README.md                 project description, installation and usage
├── requirements.txt          python dependencies
├── .gitignore
├── fibroinesm/
│   ├── silk_model.py               model definition, training and prediction CLI
│   ├── extract_top1_HC.py          top-1 heavy-chain extraction with biological validation rules
│   └── ablation_study.py           benchmarking of ProtBERT, ESM-1b and ESM-2 backbones
└── data/                           training data description (see below)
```

## Installation

```bash
conda create -n fibroinesm python=3.10
conda activate fibroinesm
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training and large-scale screening.

## Usage

### 1. Train

```bash
python fibroinesm/silk_model.py train \
    --train_heavy train_data/train_heavy.fasta \
    --train_light train_data/train_light.fasta \
    --train_p25 train_data/train_p25.fasta \
    --train_neg train_data/train_neg.fasta \
    --epochs 10 --batch_size 32 --freeze_layers 20 \
    --model_save_dir ./ESM-2
```

### 2. Screen proteomes

```bash
python fibroinesm/silk_model.py predict \
    --predict_dir ./proteomes \
    --output ./all_predict_results \
    --model_dir ./ESM-2 \
    --batch_size 32 --strict --max_silk_ratio 0.02
```

### 3. Extract the single top heavy chain per species

Applies the biological validation rules described in the Methods (heavy-chain class, minimum length 200 amino acids, single highest-scoring candidate per species).

```bash
python fibroinesm/extract_top1_HC.py \
    --pred_dir ./all_predict_results \
    --pep_dir ./proteomes \
    --out ./top1_HC.fa \
    --stats ./top1_HC_stats.csv
```

### 4. Reproduce the model benchmarking

```bash
python fibroinesm/ablation_study.py
```

Trains equivalently fine-tuned ProtBERT and ESM-1b variants on identical splits and writes metrics and figures to `./ablation_results/`.

## Data

The reference training set (1,976 heavy-chain, 45 light-chain, 51 P25 and 41,216 non-silk proteins) was compiled from published silk gene annotations and UniProt entries. The input FASTA files are not redistributed here because of their size; they are available from the corresponding author upon request, and the 121 screened proteomes are listed in Supplementary Table 1 of the associated publication.

## License

MIT (or replace with your preferred license before making the repository public)
