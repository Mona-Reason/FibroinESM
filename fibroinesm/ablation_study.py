                      

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments
import pandas as pd
import numpy as np
from collections import Counter
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from silk_model import (
    read_fasta, SilkDataset, train_collator, compute_metrics,
    BIO_CONSTRAINTS, CLASS_NAMES,
    FocalLoss, ConfidencePenaltyLoss, SilkTrainer
)

BASE_DATA = "/home/rst/data01/reason/FibroinESM/00_data"
TRAIN_LIGHT = os.path.join(BASE_DATA, "train_data", "train_light.fasta")
TRAIN_HEAVY = os.path.join(BASE_DATA, "train_data", "train_heavy.fasta")
TRAIN_P25   = os.path.join(BASE_DATA, "train_data", "train_p25.fasta")
TRAIN_NEG   = os.path.join(BASE_DATA, "train_data", "train_neg.fasta")

TEST_LIGHT  = os.path.join(BASE_DATA, "test_data", "test_light.fasta")
TEST_HEAVY  = os.path.join(BASE_DATA, "test_data", "test_heavy.fasta")
TEST_P25    = os.path.join(BASE_DATA, "test_data", "test_p25.fasta")
TEST_NEG    = os.path.join(BASE_DATA, "test_data", "test_neg.fasta")

MODEL_VARIANTS = {
    'ProtBERT': {
        'model_name': 'Rostlab/prot_bert_bfd',
        'hidden_size': 1024,
        'freeze_layers': 20,
    },
    'ESM-1b': {
        'model_name': 'facebook/esm1b_t33_650M_UR50S',
        'hidden_size': 1280,
        'freeze_layers': 20,
    },
    'ESM-2': {
        'model_name': 'facebook/esm2_t33_650M_UR50D',
        'hidden_size': 1280,
        'freeze_layers': 20,
    },
}

RESULTS_DIR = "./ablation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, "ablation_study_results.csv")

class SilkModelAblation(nn.Module):
    
    def __init__(self, model_name, hidden_size, freeze_layers):
        super().__init__()
        self.esm = AutoModel.from_pretrained(model_name)
        self._freeze_layers(freeze_layers)
        self.esm_head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(),
        )
        self.bio_head = nn.Sequential(
            nn.Linear(38, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(),
        )
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.classifier = nn.Linear(64, 4)

    def _freeze_layers(self, num_freeze):
        for param in self.esm.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.esm.encoder.layer):
            if i < num_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"  Frozen first {num_freeze} layers")

    def forward(self, input_ids, attention_mask, enhanced_features, labels=None, **kwargs):
        esm_output = self.esm(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
        cls = esm_output.last_hidden_state[:, 0, :]
        esm_feat = self.esm_head(cls)
        bio_feat = self.bio_head(enhanced_features)
        alpha = torch.clamp(self.fusion_alpha, 0.3, 0.7)
        fused = esm_feat * alpha + bio_feat * (1 - alpha)
        logits = self.classifier(fused)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return logits

def load_data():
    print("Loading data...")
    train_seqs = {1: read_fasta(TRAIN_LIGHT), 2: read_fasta(TRAIN_HEAVY),
                  3: read_fasta(TRAIN_P25),   0: read_fasta(TRAIN_NEG)}
    test_seqs  = {1: read_fasta(TEST_LIGHT),  2: read_fasta(TEST_HEAVY),
                  3: read_fasta(TEST_P25),    0: read_fasta(TEST_NEG)}

    train_all, train_labels = [], []
    for cls, seqs in train_seqs.items():
        train_all.extend(seqs)
        train_labels.extend([cls] * len(seqs))

    target_counts = {0: min(sum(1 for l in train_labels if l == 0), 50000),
                     1: max(sum(1 for l in train_labels if l == 1), 1000),
                     2: max(sum(1 for l in train_labels if l == 2), 2000),
                     3: max(sum(1 for l in train_labels if l == 3), 1000)}
    seq_by_class = {i: [] for i in range(4)}
    for seq, lbl in zip(train_all, train_labels):
        seq_by_class[lbl].append(seq)
    random.seed(42)
    balanced, balanced_labels = [], []
    for cls, target in target_counts.items():
        if len(seq_by_class[cls]) > 0:
            if target > len(seq_by_class[cls]):
                sampled = seq_by_class[cls] + random.choices(seq_by_class[cls], k=target - len(seq_by_class[cls]))
            else:
                sampled = random.sample(seq_by_class[cls], k=target)
            balanced.extend(sampled)
            balanced_labels.extend([cls] * target)
    combined = list(zip(balanced, balanced_labels))
    random.shuffle(combined)
    train_all, train_labels = [x[0] for x in combined], [x[1] for x in combined]

    test_all, test_labels = [], []
    for cls, seqs in test_seqs.items():
        test_all.extend(seqs)
        test_labels.extend([cls] * len(seqs))

    print(f"  Train: {len(train_all)} (N={train_labels.count(0)}, L={train_labels.count(1)}, H={train_labels.count(2)}, P={train_labels.count(3)})")
    print(f"  Test:  {len(test_all)}  (N={test_labels.count(0)}, L={test_labels.count(1)}, H={test_labels.count(2)}, P={test_labels.count(3)})")
    return train_all, train_labels, test_all, test_labels

def plot_training_curves(log_history, model_name, save_path):
    
    train_epochs, train_losses = [], []
    eval_epochs, eval_losses, eval_heavy_f1 = [], [], []
    for entry in log_history:
        if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
            train_epochs.append(entry["epoch"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and "epoch" in entry:
            eval_epochs.append(entry["epoch"])
            eval_losses.append(entry["eval_loss"])
        if "eval_heavy_f1" in entry and "epoch" in entry:
            eval_heavy_f1.append(entry["eval_heavy_f1"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    if train_epochs:
        axes[0].plot(train_epochs, train_losses, 'b-o', markersize=3, label='Train Loss')
    if eval_epochs:
        axes[0].plot(eval_epochs, eval_losses, 'r-s', markersize=3, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name}: Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if eval_epochs and eval_heavy_f1:
        axes[1].plot(eval_epochs, eval_heavy_f1, 'g-^', markersize=4, label='Heavy F1')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Heavy F1')
        axes[1].set_title(f'{model_name}: Heavy F1')
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    if eval_epochs:
        eval_acc = [entry.get('eval_accuracy', None) for entry in log_history if 'eval_accuracy' in entry]
        if eval_acc:
            axes[2].plot(eval_epochs[:len(eval_acc)], eval_acc, 'm-D', markersize=3, label='Overall Acc')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Accuracy')
            axes[2].set_title(f'{model_name}: Overall Accuracy')
            axes[2].set_ylim(0.5, 1.05)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved training curves: {save_path}")

def plot_ablation_bar(results_df, save_path):
    metrics = ['Overall_Acc', 'Silk_Acc', 'Heavy_F1', 'Heavy_Recall', 'Heavy_Precision']
    labels = ['Overall Acc', 'Silk Acc', 'Heavy F1', 'Heavy Recall', 'Heavy Precision']
    models = results_df['Model'].tolist()
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        vals = [results_df.loc[i, m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=model, color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation Study: Backbone Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, model in enumerate(models):
        vals = [results_df.loc[i, m] for m in metrics]
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved bar chart: {save_path}")

def plot_ablation_radar(results_df, save_path):
    
    categories = ['Overall\nAcc', 'Silk\nAcc', 'Type\nAcc', 'Heavy\nF1', 'Heavy\nRecall', '1-FPR\nSilk']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    markers = ['o', 's', '^']

    for i, row in results_df.iterrows():
        values = [row['Overall_Acc'], row['Silk_Acc'], row['Type_Acc'],
                  row['Heavy_F1'], row['Heavy_Recall'], 1 - row['FPR_Silk']]
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2, marker=markers[i],
                markersize=6, label=row['Model'])
        ax.fill(angles, values, color=colors[i], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('Ablation Study: Multi-metric Radar Comparison', fontsize=14,
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved radar chart: {save_path}")

def plot_fpr_comparison(results_df, save_path):
    
    fig, ax = plt.subplots(figsize=(8, 5))
    models = results_df['Model'].tolist()
    x = np.arange(len(models))
    width = 0.35
    fpr_silk = results_df['FPR_Silk'].tolist()
    fpr_heavy = results_df['FPR_Heavy'].tolist()

    ax.bar(x - width/2, fpr_silk, width, label='Silk FPR', color='#e67e22', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, fpr_heavy, width, label='Heavy FPR', color='#9b59b6', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('False Positive Rate', fontsize=12)
    ax.set_title('Ablation Study: False Positive Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, (fs, fh) in enumerate(zip(fpr_silk, fpr_heavy)):
        ax.text(i - width/2, fs + 0.001, f'{fs:.4f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, fh + 0.001, f'{fh:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved FPR chart: {save_path}")

def train_and_eval(model_variant, train_seqs, train_labels, test_seqs, test_labels, output_dir):
    
    name = model_variant['model_name'].split('/')[-1]
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    train_ds = SilkDataset(train_seqs, train_labels, max_length=512)
    test_ds  = SilkDataset(test_seqs,  test_labels,  max_length=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = SilkModelAblation(model_variant['model_name'],
                              model_variant['hidden_size'],
                              model_variant['freeze_layers'])
    model.to(device)

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if "esm" in n and p.requires_grad], "lr": 1e-5},
        {"params": [p for n, p in param_optimizer if "esm" not in n], "lr": 1e-4},
    ]
    from torch.optim import AdamW
    optimizer = AdamW(optimizer_grouped_parameters)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="epoch",
        learning_rate=1e-5,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="heavy_f1",
        greater_is_better=True,
        seed=42,
        save_total_limit=2,
        save_safetensors=False,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    trainer = SilkTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )
    trainer.label_names = ["labels"]
    trainer.data_collator = train_collator

    
    latest_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
                       if d.startswith('checkpoint-') and os.path.isdir(os.path.join(output_dir, d))]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            print(f"  Resuming from checkpoint: {latest_checkpoint}")

    
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    
    test_results = trainer.evaluate(test_ds)
    print(f"\n  Test Results for {name}:")
    for k, v in test_results.items():
        print(f"    {k}: {v:.4f}")

    
    curve_path = os.path.join(RESULTS_DIR, f"Fig_Ablation_TrainingCurves_{model_variant['model_name'].split('/')[-1]}.png")
    plot_training_curves(trainer.state.log_history, name, curve_path)

    return test_results

def run_ablation():
    
    
    for path in [TRAIN_LIGHT, TRAIN_HEAVY, TRAIN_P25, TRAIN_NEG,
                 TEST_LIGHT, TEST_HEAVY, TEST_P25, TEST_NEG]:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)

    train_seqs, train_labels, test_seqs, test_labels = load_data()

    results = []
    for variant_name, config in MODEL_VARIANTS.items():
        out_dir = f"./ablation_{variant_name}"
        
        
        if os.path.exists(RESULTS_CSV):
            existing = pd.read_csv(RESULTS_CSV)
            if variant_name in existing['Model'].values:
                print(f"\nSkipping {variant_name} (already completed)")
                row = existing[existing['Model'] == variant_name].iloc[0]
                results.append(row.to_dict())
                continue
        
        try:
            metrics = train_and_eval(config, train_seqs, train_labels, test_seqs, test_labels, out_dir)
            results.append({
                'Model': variant_name,
                'BaseModel': config['model_name'],
                'HiddenSize': config['hidden_size'],
                'Overall_Acc':  metrics.get('eval_accuracy', 0),
                'Silk_Acc':     metrics.get('eval_silk_accuracy', 0),
                'Type_Acc':     metrics.get('eval_type_accuracy', 0),
                'Heavy_Recall': metrics.get('eval_heavy_recall', 0),
                'Heavy_Precision': metrics.get('eval_heavy_precision', 0),
                'Heavy_F1':     metrics.get('eval_heavy_f1', 0),
                'FPR_Silk':     metrics.get('eval_fpr_silk', 0),
                'FPR_Heavy':    metrics.get('eval_fpr_heavy', 0),
            })
        except Exception as e:
            print(f"ERROR training {variant_name}: {e}")
            import traceback
            traceback.print_exc()
    
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n{'='*60}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    
    md_path = RESULTS_CSV.replace('.csv', '.md')
    with open(md_path, 'w') as f:
        f.write("| Model | Overall Acc | Silk Acc | Type Acc | Heavy F1 | Heavy Recall | Heavy Precision | Silk FPR | Heavy FPR |\n")
        f.write("|-------|------------|----------|----------|----------|--------------|-----------------|----------|-----------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['Model']} | {row['Overall_Acc']:.4f} | {row['Silk_Acc']:.4f} | "
                   f"{row['Type_Acc']:.4f} | {row['Heavy_F1']:.4f} | {row['Heavy_Recall']:.4f} | "
                   f"{row['Heavy_Precision']:.4f} | {row['FPR_Silk']:.4f} | {row['FPR_Heavy']:.4f} |\n")
    print(f"\n  Markdown table: {md_path}")

    
    if len(df) > 0:
        plot_ablation_bar(df, os.path.join(RESULTS_DIR, "Fig_Ablation_Performance.png"))
        plot_ablation_radar(df, os.path.join(RESULTS_DIR, "Fig_Ablation_Radar.png"))
        plot_fpr_comparison(df, os.path.join(RESULTS_DIR, "Fig_Ablation_FPR.png"))

    print(f"\n{'='*60}")
    print("All ablation results & figures saved to:", RESULTS_DIR)
    print(f"{'='*60}")

if __name__ == '__main__':
    run_ablation()
