import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import logging
import random
import re
import shutil
from typing import List, Tuple, Dict, Optional
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from Bio import SeqIO
from matplotlib.backends.backend_pdf import PdfPages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: 'Non-Silk', 1: 'Light-Chain', 2: 'Heavy-Chain', 3: 'P25'}
SILK_CLASSES = [1, 2, 3]
CLASS_ABBR = {0: 'Non', 1: 'L', 2: 'H', 3: 'P25'}

MODEL_CONFIG = {
    'model_name': "facebook/esm2_t33_650M_UR50D",
    'hidden_size': 1280,
    'max_length': 512,
}

BIO_CONSTRAINTS = {
    'heavy': {
        'min_length': 200,
        'max_length': 8000,
        'min_ga_ratio': 0.15,
        'min_gas_ratio': 0.30,
        'min_gagags_repeats': 1,
        'min_gagagy_repeats': 0,
        'min_tyr_count': 2,
        'max_other_ratio': 0.55,
    },
    'light': {
        'min_length': 100,
        'max_length': 500,
        'min_cys_count': 1,
        'max_cys_count': 10,
        'max_ga_ratio': 0.30,
    },
    'p25': {
        'min_length': 150,
        'max_length': 400,
        'min_nglyco_sites': 0,
        'max_ga_ratio': 0.25,
        'min_leu_ratio': 0.03,
    }
}

PREDICTION_CONFIG = {
    'silk_threshold': 0.90,
    'type_threshold': 0.90,
    'joint_min_confidence': 0.90,
    'absolute_min_confidence': 0.95,
}

def read_fasta(file_path: str) -> List[Tuple[str, str]]:
    sequences = []
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            seq_id = record.id
            sequence = str(record.seq).replace("*", "")
            if len(sequence) > 0:
                sequences.append((seq_id, sequence))
        logger.info(f"Successfully read FASTA file: {file_path}, total {len(sequences)} sequences")
    except Exception as e:
        logger.error(f"Failed to read FASTA file: {file_path}, error: {str(e)}")
        raise
    return sequences

def calculate_enhanced_features(sequence: str) -> List[float]:
    if len(sequence) == 0:
        return [0.0] * 38

    seq_len = len(sequence)
    aa_counts = Counter(sequence)

    g_count = aa_counts.get("G", 0)
    a_count = aa_counts.get("A", 0)
    s_count = aa_counts.get("S", 0)
    y_count = aa_counts.get("Y", 0)
    v_count = aa_counts.get("V", 0)
    c_count = aa_counts.get("C", 0)
    n_count = aa_counts.get("N", 0)
    l_count = aa_counts.get("L", 0)
    gas_total = g_count + a_count + s_count

    features = []
    features.append(gas_total / seq_len)
    features.append(g_count / seq_len)
    features.append(a_count / seq_len)
    features.append(s_count / seq_len)
    features.append(gas_total / max(1, seq_len - gas_total))
    features.append(y_count / seq_len)
    features.append(v_count / seq_len)
    features.append(c_count / seq_len)
    features.append(n_count / seq_len)

    dipeptide_counts = Counter(sequence[i] + sequence[i+1] for i in range(len(sequence)-1))
    total_dipeptides = max(1, seq_len - 1)

    features.append(dipeptide_counts.get("GA", 0) / total_dipeptides)
    features.append(dipeptide_counts.get("GS", 0) / total_dipeptides)
    features.append(dipeptide_counts.get("AG", 0) / total_dipeptides)
    features.append(dipeptide_counts.get("SG", 0) / total_dipeptides)
    features.append(dipeptide_counts.get("GY", 0) / total_dipeptides)
    features.append(dipeptide_counts.get("YG", 0) / total_dipeptides)
    features.append(dipeptide_counts.get("GV", 0) / total_dipeptides)
    features.append(dipeptide_counts.get("VG", 0) / total_dipeptides)

    gagags_count = sequence.count('GAGAGS')
    features.append(gagags_count / max(1, seq_len // 6))
    features.append(min(gagags_count, 500) / 500.0)

    gagagy_count = sequence.count('GAGAGY')
    features.append(gagagy_count / max(1, seq_len // 6))
    features.append(min(gagagy_count, 200) / 200.0)

    heavy_patterns = ['GAGVGY', 'GVGAGY', 'GAGYGA', 'GAGAGAGA', 'GAGAGVGY', 'GVGAGAGY', 'GAGYGAG', 'GAGAGYG']
    heavy_repeats = sum(sequence.count(p) for p in heavy_patterns)
    features.append(heavy_repeats / max(1, seq_len // 6))
    features.append(min(heavy_repeats, 300) / 300.0)

    max_ga_stretch = 0
    current = 0
    for i in range(len(sequence) - 1):
        if sequence[i:i+2] in ['GA', 'AG']:
            current += 1
            max_ga_stretch = max(max_ga_stretch, current)
        else:
            current = 0
    features.append(max_ga_stretch / seq_len)
    features.append(min(max_ga_stretch, 100) / 100.0)

    block_size = 50
    gas_rich_blocks = 0
    for i in range(0, len(sequence) - block_size, block_size):
        block = sequence[i:i+block_size]
        gas_ratio = sum(1 for aa in block if aa in 'GAS') / len(block)
        if gas_ratio > 0.7:
            gas_rich_blocks += 1
    features.append(gas_rich_blocks / max(1, seq_len // block_size))

    max_alternating_stretch = 0
    current_stretch = 0
    for i in range(len(sequence) - 1):
        pair = sequence[i] + sequence[i+1]
        if pair in {"GA", "GS", "AG", "SG", "GY", "YG", "GV", "VG"}:
            current_stretch += 1
            max_alternating_stretch = max(max_alternating_stretch, current_stretch)
        else:
            current_stretch = 0
    features.append(max_alternating_stretch / max(1, seq_len - 1))

    triplet_counts = Counter(sequence[i:i+3] for i in range(len(sequence)-2))
    max_trip = max(1, len(sequence) - 2)
    features.append(1.0 - min(triplet_counts.get("GGG", 0) / max_trip, 1.0))
    features.append(1.0 - min(triplet_counts.get("AAA", 0) / max_trip, 1.0))
    features.append(1.0 - min(triplet_counts.get("SSS", 0) / max_trip, 1.0))
    features.append(1.0 - min(triplet_counts.get("LLL", 0) / max_trip, 1.0))
    features.append(1.0 - min(sequence.count("GLGL") * 4 / max(1, len(sequence)), 1.0))

    cc_pattern = sequence.count('CC')
    features.append(cc_pattern / max(1, seq_len // 10))

    n_glyco = sum(1 for i in range(len(sequence)-2) if sequence[i] == 'N' and sequence[i+2] in 'ST')
    features.append(n_glyco / max(1, seq_len // 10))

    features.append(min(seq_len, 10000) / 10000.0)
    features.append(1.0 if seq_len > 3000 else 0.0)
    features.append(1.0 if 150 <= seq_len <= 400 else 0.0)
    features.append(1.0 if 180 <= seq_len <= 350 else 0.0)

    return features

def quick_bio_filter(sequence: str) -> Tuple[bool, str]:
    seq_len = len(sequence)
    ga_ratio = sum(1 for aa in sequence if aa in 'GA') / max(1, seq_len)
    seq_len = len(sequence)

    if seq_len >= 100 and ga_ratio >= 0.15:
        return True, "potential_heavy"

    if 80 <= seq_len <= 500:
        c_count = sequence.count('C')
        if c_count >= 1:
            return True, "potential_light"

    return False, "fails_quick_filter"

def strict_biological_validation(sequence: str, predicted_class: int, confidence: float = None, joint_prob: np.ndarray = None, type_prob: np.ndarray = None) -> Tuple[bool, str, int]:
    seq_len = len(sequence)
    aa_counts = Counter(sequence)

    if predicted_class == 2:
        constraints = BIO_CONSTRAINTS['heavy']

        if seq_len < constraints['min_length']:
            return False, f"heavy_too_short_{seq_len}", 0
        if seq_len > constraints['max_length']:
            return False, f"heavy_too_long_{seq_len}", 0

        ga_count = aa_counts.get('G', 0) + aa_counts.get('A', 0)
        ga_ratio = ga_count / seq_len
        if ga_ratio < constraints['min_ga_ratio']:
            return False, f"heavy_low_ga_ratio_{ga_ratio:.3f}", 0

        gas_count = ga_count + aa_counts.get('S', 0)
        gas_ratio = gas_count / seq_len
        if gas_ratio < constraints['min_gas_ratio']:
            return False, f"heavy_low_gas_ratio_{gas_ratio:.3f}", 0

        gagags_count = sequence.count('GAGAGS')
        gagagy_count = sequence.count('GAGAGY')
        if gagags_count < constraints['min_gagags_repeats'] and gagagy_count < constraints['min_gagagy_repeats']:
            return False, f"heavy_low_repeats_GS{gagags_count}_GY{gagagy_count}", 0

        if aa_counts.get('Y', 0) < constraints['min_tyr_count']:
            return False, f"heavy_low_tyr_{aa_counts.get('Y', 0)}", 0

        other_count = seq_len - gas_count - aa_counts.get('Y', 0) - aa_counts.get('V', 0)
        other_ratio = other_count / seq_len
        if other_ratio > constraints['max_other_ratio']:
            return False, f"heavy_high_other_ratio_{other_ratio:.3f}", 0

        return True, "heavy_pass", 2

    elif predicted_class == 1:
        constraints = BIO_CONSTRAINTS['light']

        if seq_len < constraints['min_length']:
            return False, f"light_too_short_{seq_len}", 0
        if seq_len > constraints['max_length']:
            return False, f"light_too_long_{seq_len}", 0

        cys_count = aa_counts.get('C', 0)
        if cys_count < constraints['min_cys_count']:
            return False, f"light_low_cys_{cys_count}", 0
        if cys_count > constraints['max_cys_count']:
            return False, f"light_high_cys_{cys_count}", 0

        ga_ratio = (aa_counts.get('G', 0) + aa_counts.get('A', 0)) / seq_len
        if ga_ratio > constraints['max_ga_ratio']:
            return False, f"light_high_ga_ratio_{ga_ratio:.3f}", 0

        return True, "light_pass", 1

    elif predicted_class == 3:
        constraints = BIO_CONSTRAINTS['p25']

        if seq_len < constraints['min_length']:
            return False, f"p25_too_short_{seq_len}", 0
        if seq_len > constraints['max_length']:
            return False, f"p25_too_long_{seq_len}", 0

        n_glyco = sum(1 for i in range(len(sequence)-2) if sequence[i] == 'N' and sequence[i+2] in 'ST')
        if n_glyco < constraints['min_nglyco_sites']:
            return False, f"p25_low_glyco_{n_glyco}", 0

        ga_ratio = (aa_counts.get('G', 0) + aa_counts.get('A', 0)) / seq_len
        if ga_ratio > constraints['max_ga_ratio']:
            return False, f"p25_high_ga_ratio_{ga_ratio:.3f}", 0

        leu_ratio = aa_counts.get('L', 0) / seq_len
        if leu_ratio < constraints['min_leu_ratio']:
            return False, f"p25_low_leu_ratio_{leu_ratio:.3f}", 0

        return True, "p25_pass", 3

    return True, "non_silk", 0

def has_forbidden_repeats(sequence: str) -> Tuple[bool, str]:
\
\
\
\
\
\
       
    if len(sequence) < 8:
        return False, ""
    
    for aa in ['G', 'A', 'S']:
        if aa * 4 not in sequence:
            continue
        max_repeat = 0
        current = 0
        for char in sequence:
            if char == aa:
                current += 1
                max_repeat = max(max_repeat, current)
            else:
                current = 0
        if max_repeat >= 4:
            return True, f"quad_{aa}_repeat_{max_repeat}"
    
    non_gas = set('CDEFHIKLMNPQRTVWY')
    
    for i in range(len(sequence) - 5):
        x, y = sequence[i], sequence[i+1]
        if x != y and x in non_gas and y in non_gas:
            pattern = sequence[i:i+6]
            if (pattern[0] == pattern[2] == pattern[4] and 
                pattern[1] == pattern[3] == pattern[5] and
                pattern[0] != pattern[1]):
                return True, f"non_gas_di_repeat_{x}{y}"
    
    return False, ""

def apply_top20_filters(sequence: str, predicted_label: str) -> Tuple[bool, str]:
\
\
\
       
    if len(sequence) == 0:
        return False, "empty_sequence"
    
    seq_len = len(sequence)
    aa_counts = Counter(sequence)
    
    if predicted_label == "Heavy-Chain":
        if seq_len < 200:
            return False, f"heavy_too_short_{seq_len}"
    
    if predicted_label == "Light-Chain":
        branched = ['V', 'I', 'L']
        branched_count = sum(aa_counts.get(aa, 0) for aa in branched)
        branched_ratio = branched_count / seq_len if seq_len > 0 else 0
        
        hydrophobic = ['V', 'I', 'L', 'F', 'M', 'W', 'A', 'P', 'C', 'G']
        hydro_count = sum(aa_counts.get(aa, 0) for aa in hydrophobic)
        hydro_ratio = hydro_count / seq_len if seq_len > 0 else 0
        
        if branched_count < 3 and branched_ratio < 0.02:
            return False, f"light_low_branched_{branched_count}"
        if hydro_ratio < 0.15:
            return False, f"light_low_hydrophobic_{hydro_ratio:.3f}"
    
    if predicted_label == "P25":
        n_glyco = sum(1 for i in range(len(sequence)-2) if sequence[i] == 'N' and sequence[i+2] in 'ST')
        cys_count = aa_counts.get('C', 0)
        
        if n_glyco < 1:
            return False, "p25_no_glycosylation"
        if cys_count < 2:
            return False, f"p25_low_cys_globular_{cys_count}"
    
    return True, "pass"

def extract_gene_number(seq_id: str) -> int:
\
\
\
       
    numbers = re.findall(r'\d+', seq_id)
    if not numbers:
        return 0
    last_num = int(numbers[-1])
    if last_num <= 10 and len(numbers) > 1:
        return int(numbers[-2])
    return last_num

def cluster_and_select(candidates: List[Dict], max_per_cluster: int = 3, cluster_threshold: int = 10) -> List[Dict]:
\
\
\
\
       
    if not candidates:
        return []
    
    candidates_by_conf = sorted(candidates, key=lambda x: -x['confidence'])
    best = candidates_by_conf[0]
    best_num = extract_gene_number(best['sequence_id'])
    
    cluster = [best]
    for c in candidates_by_conf[1:]:
        num = extract_gene_number(c['sequence_id'])
        if abs(num - best_num) <= cluster_threshold:
            cluster.append(c)
    
    cluster.sort(key=lambda x: extract_gene_number(x['sequence_id']))
    
    selected = cluster[:max_per_cluster]
    
    if len(candidates) > len(selected):
        removed_ids = [c['sequence_id'] for c in candidates if c not in selected]
        logger.info(f"    [Gene Cluster] Center: {best['sequence_id']} (gene {best_num}), "
                   f"kept {len(selected)} near genes, removed {len(removed_ids)} distant: {removed_ids[:3]}...")
    
    return selected

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        valid_mask = targets != self.ignore_index
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        if len(targets) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ConfidencePenaltyLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='mean')
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        penalty = -self.beta * entropy.mean()
        return ce_loss + penalty

class SilkDataset(Dataset):
    def __init__(self, sequences: List[Tuple[str, str]], labels: Optional[List[int]] = None,
                 max_length: int = 512):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_id, sequence = self.sequences[idx]

        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        features = calculate_enhanced_features(sequence)

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "enhanced_features": torch.tensor(features, dtype=torch.float32),
            "seq_id": seq_id,
        }

        if self.labels is not None:
            lbl_t = torch.tensor(self.labels[idx], dtype=torch.long)
            item["labels"] = lbl_t
            item["label"] = lbl_t
            item["label_ids"] = item["labels"]
        item["input"] = item["input_ids"]
        return item

class SilkModelV3(nn.Module):
    def __init__(self, model_name: str = None, hidden_size: int = None):
        super().__init__()

        if model_name is None:
            model_name = MODEL_CONFIG['model_name']
        if hidden_size is None:
            hidden_size = MODEL_CONFIG['hidden_size']

        self.esm = AutoModel.from_pretrained(model_name)
        self._freeze_layers(20)

        self.esm_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        self.bio_head = nn.Sequential(
            nn.Linear(38, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
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
        logger.info(f"Frozen first {num_freeze} layers of ESM-2")

    def forward(self, input_ids, attention_mask, enhanced_features, labels=None, **kwargs):
        esm_output = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False
        )
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

def load_v2_to_v3(model_dir: str, device: torch.device):
    model = SilkModelV3()
    
    model_path = os.path.join(model_dir, "best_model", "pytorch_model.bin")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "pytorch_model.bin")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    key_mapping = {
        "joint_classifier.weight": "classifier.weight",
        "joint_classifier.bias": "classifier.bias",
    }
    
    new_state_dict = {}
    skipped_keys = []
    
    for key, value in state_dict.items():
        if key in key_mapping:
            new_state_dict[key_mapping[key]] = value
            logger.info(f"Mapped: {key} -> {key_mapping[key]}")
        elif key.startswith("silk_classifier.") or key.startswith("type_classifier."):
            skipped_keys.append(key)
        else:
            new_state_dict[key] = value
    
    if skipped_keys:
        logger.info(f"Skipped {len(skipped_keys)} untrained keys: {skipped_keys[:3]}...")
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if missing:
        logger.warning(f"Missing keys after mapping: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    
    model.to(device)
    model.eval()
    
    return model

def compute_universal_score(sequence, heavy_prob, silk_prob, repeat_density=0.0):
    seq_len = len(sequence)
    g = sequence.count('G')
    a = sequence.count('A')
    s = sequence.count('S')
    ga_count = g + a
    gas_count = g + a + s
    ga_ratio = ga_count / max(1, seq_len)
    gas_ratio = gas_count / max(1, seq_len)

    repeats = sum(1 for i in range(seq_len - 4) if sum(1 for aa in sequence[i:i+5] if aa in 'GA') >= 3)
    repeat_frac = repeats / max(1, seq_len)

    if seq_len >= 25:
        nterm_hydro = sum(1 for aa in sequence[5:25] if aa in 'VILFAMW')
        has_signal = 1.0 if nterm_hydro >= 5 else 0.0
    else:
        has_signal = 0.0

    ga_score = min(ga_ratio / 0.40, 1.0)
    gas_score = min(gas_ratio / 0.60, 1.0)
    repeat_score = min(repeat_frac / 0.10, 1.0)
    density_score = min(repeat_density / 0.1, 1.0) if repeat_density > 0 else 0.0

    if 300 < seq_len < 2000:
        length_score = 1.0
    elif seq_len <= 300:
        length_score = max(0.0, (seq_len - 100) / 200.0)
    else:
        length_score = max(0.0, 1.0 - (seq_len - 2000) / 6000.0)

    bio_composite = (ga_score * 0.15 + gas_score * 0.25 +
                     repeat_score * 0.25 + length_score * 0.15 +
                     density_score * 0.10 + has_signal * 0.10)

    return round(heavy_prob * 0.25 + bio_composite * 0.75, 4)

def hierarchical_predict_v3(
    sequence: str,
    joint_prob: np.ndarray,
    repeat_density: float,
    threshold: float = None,
    strict_mode: bool = True,
    universal_mode: bool = False
) -> Tuple[int, float, Dict]:
    if threshold is None:
        threshold = PREDICTION_CONFIG['silk_threshold']

    best_class = int(np.argmax(joint_prob))
    confidence = joint_prob[best_class]

    if best_class > 0 and confidence < threshold:
        if strict_mode:
            best_class = 0
            confidence = joint_prob[0]
        else:
            sorted_probs = np.sort(joint_prob)[::-1]
            if sorted_probs[1] > confidence * 0.5:
                second_class = int(np.argsort(joint_prob)[-2])
                if second_class > 0:
                    best_class = second_class
                    confidence = joint_prob[best_class]

    if best_class > 0:
        is_valid, rule_reason, corrected_class = strict_biological_validation(
            sequence, best_class, confidence
        )
        
        if not is_valid and confidence < 0.99:
            best_class = corrected_class
            confidence = joint_prob[best_class] if best_class < len(joint_prob) else confidence
            rule_applied = True
        else:
            rule_applied = False
    else:
        is_valid = True
        rule_reason = "non_silk"
        rule_applied = False

    if best_class > 0 and confidence < PREDICTION_CONFIG['absolute_min_confidence']:
        best_class = 0
        confidence = joint_prob[0]
        rule_applied = True

    details = {
        'joint_probabilities': {CLASS_NAMES[i]: joint_prob[i] for i in range(4)},
        'rule_validation': f"joint|{rule_reason}" if rule_applied else "joint",
        'rule_corrected': rule_applied,
        'level': 'L0_joint_direct',
        'biological_pass': is_valid,
        'repeat_density': repeat_density
    }
    
    return best_class, confidence, details

def strict_filter_predictions(results_df: pd.DataFrame,
                              silk_threshold: float = None,
                              type_threshold: float = None) -> pd.DataFrame:
    if silk_threshold is None:
        silk_threshold = PREDICTION_CONFIG['silk_threshold']
    if type_threshold is None:
        type_threshold = PREDICTION_CONFIG['type_threshold']

    original_count = len(results_df)
    silk_count = len(results_df[results_df['prediction'] != 'Non-Silk'])

    mask = (
        (results_df['confidence'] > PREDICTION_CONFIG['absolute_min_confidence']) &
        (
            ((results_df['prediction'] == 'Light-Chain') &
             (results_df['seq_length'] >= BIO_CONSTRAINTS['light']['min_length']) &
             (results_df['seq_length'] <= BIO_CONSTRAINTS['light']['max_length'])) |
            ((results_df['prediction'] == 'Heavy-Chain') &
             (results_df['seq_length'] >= BIO_CONSTRAINTS['heavy']['min_length']) &
             (results_df['seq_length'] <= BIO_CONSTRAINTS['heavy']['max_length'])) |
            ((results_df['prediction'] == 'P25') &
             (results_df['seq_length'] >= BIO_CONSTRAINTS['p25']['min_length']) &
             (results_df['seq_length'] <= BIO_CONSTRAINTS['p25']['max_length']))
        ) &
        (results_df['biological_pass'] == True)
    )

    filtered = results_df[mask].copy()
    rejected = results_df[~mask].copy()
    rejected['prediction'] = 'Non-Silk'
    rejected['prediction_abbr'] = 'Non'
    rejected['confidence'] = 1.0 - rejected.get('confidence', 0.5)

    final_df = pd.concat([filtered, rejected], ignore_index=True)

    new_silk_count = len(final_df[final_df['prediction'] != 'Non-Silk'])
    logger.info(f"Strict filter: {silk_count} -> {new_silk_count} silk proteins ({original_count} total)")
    logger.info(f"  Rejected: {silk_count - new_silk_count} low-confidence or bio-failed predictions")

    return final_df

def species_level_filter(results_df: pd.DataFrame, max_silk_ratio: float = 0.02) -> pd.DataFrame:
    results_df['species'] = results_df['sequence_id'].apply(
        lambda x: x.split('_')[0] if '_' in x else x.split('|')[0] if '|' in x else 'unknown'
    )

    species_counts = results_df['species'].value_counts()

    for species in species_counts.index:
        species_df = results_df[results_df['species'] == species]
        species_total = len(species_df)
        species_silk = len(species_df[species_df['prediction'] != 'Non-Silk'])
        species_ratio = species_silk / species_total if species_total > 0 else 0

        logger.info(f"Species {species}: {species_silk}/{species_total} silk predictions ({species_ratio:.2%})")

        if species_ratio > max_silk_ratio:
            logger.warning(f"Species {species} has abnormally high silk ratio ({species_ratio:.2%}), applying emergency filter")
            species_mask = (results_df['species'] == species) & (results_df['prediction'] != 'Non-Silk')
            expected_silk = max(1, int(species_total * max_silk_ratio))
            species_silk_df = results_df[species_mask].copy()
            if len(species_silk_df) > expected_silk:
                keep_ids = species_silk_df.nlargest(expected_silk, 'confidence')['sequence_id'].tolist()
                drop_mask = species_mask & ~results_df['sequence_id'].isin(keep_ids)
                results_df.loc[drop_mask, 'prediction'] = 'Non-Silk'
                results_df.loc[drop_mask, 'prediction_abbr'] = 'Non'
                results_df.loc[drop_mask, 'confidence'] = 1.0 - results_df.loc[drop_mask, 'silk_probability']
                results_df.loc[drop_mask, 'rule_validation'] = 'species_emergency_filter'

    return results_df.drop(columns=['species'])

def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    preds = p.predictions.argmax(axis=1)
    labels = p.label_ids

    accuracy = accuracy_score(labels, preds)

    silk_preds = (preds > 0).astype(int)
    silk_labels = (labels > 0).astype(int)
    silk_accuracy = accuracy_score(silk_preds, silk_labels)

    silk_mask = labels > 0
    if silk_mask.sum() > 0:
        type_preds = preds[silk_mask] - 1
        type_labels = labels[silk_mask] - 1
        type_accuracy = accuracy_score(type_labels, type_preds)
    else:
        type_accuracy = 0.0

    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)

    heavy_recall = recall_score(labels == 2, preds == 2, zero_division=0)
    heavy_precision = precision_score(labels == 2, preds == 2, zero_division=0)
    heavy_f1 = f1_score(labels == 2, preds == 2, zero_division=0)

    fp_silk = ((preds > 0) & (labels == 0)).sum()
    tn_silk = ((preds == 0) & (labels == 0)).sum()
    fpr_silk = fp_silk / max(1, fp_silk + tn_silk)

    fp_heavy = ((preds == 2) & (labels != 2)).sum()
    fpr_heavy = fp_heavy / max(1, len(labels))

    return {
        "accuracy": accuracy,
        "silk_accuracy": silk_accuracy,
        "type_accuracy": type_accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "heavy_recall": heavy_recall,
        "heavy_precision": heavy_precision,
        "heavy_f1": heavy_f1,
        "fpr_silk": fpr_silk,
        "fpr_heavy": fpr_heavy,
    }

def plot_training_history(trainer, save_dir):
    import os
    log_history = trainer.state.log_history
    train_epochs, train_losses = [], []
    eval_epochs, eval_losses = [], []
    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            train_epochs.append(entry["epoch"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and "epoch" in entry:
            eval_epochs.append(entry["epoch"])
            eval_losses.append(entry["eval_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if train_epochs:
        axes[0].plot(train_epochs, train_losses, "b-o", markersize=4)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(True, alpha=0.3)
    if eval_epochs:
        axes[1].plot(eval_epochs, eval_losses, "r-s", markersize=4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Validation Loss")
        axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_history.png"), dpi=150)
    fig.savefig(os.path.join(save_dir, "training_history.pdf"))
    plt.close(fig)
    logger.info(f"Training history plots saved to {save_dir}")

def plot_test_results(all_labels, all_predictions, all_joint_probs, output_dir):
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    import os

    cm = confusion_matrix(all_labels, all_predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[CLASS_NAMES[i] for i in range(4)],
                yticklabels=[CLASS_NAMES[i] for i in range(4)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    fig.savefig(os.path.join(output_dir, "confusion_matrix.pdf"))
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {output_dir}")

    n_classes = 4
    y_true_bin = np.eye(n_classes)[np.array(all_labels)]
    y_score = np.array([p for p in all_joint_probs])
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    fig.savefig(os.path.join(output_dir, "roc_curves.pdf"))
    plt.close(fig)
    logger.info(f"ROC curves saved to {output_dir}")

    metrics = {
        "Accuracy": accuracy_score(all_labels, all_predictions),
        "Heavy Recall": recall_score(np.array(all_labels) == 2, np.array(all_predictions) == 2, zero_division=0),
        "Heavy Precision": precision_score(np.array(all_labels) == 2, np.array(all_predictions) == 2, zero_division=0),
        "Heavy F1": f1_score(np.array(all_labels) == 2, np.array(all_predictions) == 2, zero_division=0),
        "Silk FPR": sum((np.array(all_predictions) > 0) & (np.array(all_labels) == 0)) / len(all_labels),
    }
    colors_list = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#95a5a6"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors_list[:len(metrics)])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Test Performance Metrics")
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "test_metrics.png"), dpi=150)
    fig.savefig(os.path.join(output_dir, "test_metrics.pdf"))
    plt.close(fig)
    logger.info(f"Test metrics chart saved to {output_dir}")
    
    n_classes = 4
    categories = [CLASS_NAMES[i] for i in range(n_classes)]
    radar_data = {
        "Precision": [precision_score(np.array(all_labels) == i, np.array(all_predictions) == i, zero_division=0) for i in range(n_classes)],
        "Recall": [recall_score(np.array(all_labels) == i, np.array(all_predictions) == i, zero_division=0) for i in range(n_classes)],
        "F1-Score": [f1_score(np.array(all_labels) == i, np.array(all_predictions) == i, zero_division=0) for i in range(n_classes)],
    }

    n_metrics = len(radar_data)
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 5),
                             subplot_kw=dict(polar=True))
    if n_metrics == 1:
        axes = [axes]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for ax, (name, values), color in zip(axes, radar_data.items(), colors):
        vals = values + values[:1]
        ax.plot(angles, vals, "o-", color=color, linewidth=2)
        ax.fill(angles, vals, alpha=0.1, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
        ax.set_title(name, pad=20, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "radar_chart.png"), dpi=150)
    fig.savefig(os.path.join(output_dir, "radar_chart.pdf"))
    plt.close(fig)
    logger.info(f"Radar chart saved to {output_dir}")

def train_collator(features):
    batch = {}
    for key in features[0].keys():
        if isinstance(features[0][key], torch.Tensor):
            batch[key] = torch.stack([f[key] for f in features])
        else:
            batch[key] = [f[key] for f in features]
    return batch

class SilkTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels", None)
        if labels is None:
            labels = inputs.get("label", None)
        
        outputs = model(**inputs)
        
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
            logits = outputs.get("logits")
        else:
            logits = outputs
            loss = None
        
        if loss is None and labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        if return_outputs:
            return (loss, {"logits": logits})
        return loss

def train_model(
    train_light_file: str,
    train_heavy_file: str,
    train_p25_file: str,
    train_neg_file: str,
    val_light_file: Optional[str] = None,
    val_heavy_file: Optional[str] = None,
    val_p25_file: Optional[str] = None,
    val_neg_file: Optional[str] = None,
    model_save_dir: str = "./checkpoints_v3",
    epochs: int = 20,
    batch_size: int = 8,
    max_len: int = 512,
    lr_bert: float = 1e-5,
    lr_head: float = 1e-4,
    fp16: bool = True,
    validation_split: float = 0.2,
    freeze_layers: int = 20
) -> None:
    logger.info("Loading training data...")

    light_sequences = read_fasta(train_light_file)
    heavy_sequences = read_fasta(train_heavy_file)
    p25_sequences = read_fasta(train_p25_file)
    neg_sequences = read_fasta(train_neg_file)

    train_sequences = light_sequences + heavy_sequences + p25_sequences + neg_sequences
    train_labels = ([1] * len(light_sequences) +
                   [2] * len(heavy_sequences) +
                   [3] * len(p25_sequences) +
                   [0] * len(neg_sequences))

    logger.info(f"Training set - Light: {len(light_sequences)}, Heavy: {len(heavy_sequences)}, P25: {len(p25_sequences)}, Negative: {len(neg_sequences)}")

    if val_light_file and val_heavy_file and val_p25_file and val_neg_file:
        val_light = read_fasta(val_light_file)
        val_heavy = read_fasta(val_heavy_file)
        val_p25 = read_fasta(val_p25_file)
        val_neg = read_fasta(val_neg_file)

        val_sequences = val_light + val_heavy + val_p25 + val_neg
        val_labels = ([1] * len(val_light) +
                     [2] * len(val_heavy) +
                     [3] * len(val_p25) +
                     [0] * len(val_neg))
        logger.info(f"Validation set - Light: {len(val_light)}, Heavy: {len(val_heavy)}, P25: {len(val_p25)}, Negative: {len(val_neg)}")
    else:
        logger.info(f"No validation set provided, using {validation_split*100}% of training data for validation")
        train_sequences, val_sequences, train_labels, val_labels = train_test_split(
            train_sequences, train_labels, test_size=validation_split,
            random_state=42, stratify=train_labels
        )
        logger.info(f"After split - Training size: {len(train_sequences)}, Validation size: {len(val_sequences)}")

    train_count = Counter(train_labels)
    logger.info(f"Before balancing: {dict(train_count)}")

    target_counts = {
        0: min(train_count[0], 50000),
        1: max(train_count[1], 1000),
        2: max(train_count[2], 2000),
        3: max(train_count[3], 1000),
    }

    seq_by_class = {i: [] for i in range(4)}
    for seq, label in zip(train_sequences, train_labels):
        seq_by_class[label].append(seq)

    random.seed(42)
    balanced_sequences = []
    balanced_labels = []
    for cls, target in target_counts.items():
        if len(seq_by_class[cls]) > 0:
            if target > len(seq_by_class[cls]):
                sampled = seq_by_class[cls] + random.choices(seq_by_class[cls], k=target - len(seq_by_class[cls]))
            else:
                sampled = random.sample(seq_by_class[cls], k=target)
            balanced_sequences.extend(sampled)
            balanced_labels.extend([cls] * target)

    combined = list(zip(balanced_sequences, balanced_labels))
    random.shuffle(combined)
    train_sequences, train_labels = [x[0] for x in combined], [x[1] for x in combined]

    logger.info(f"After balancing: {Counter(train_labels)}")

    train_dataset = SilkDataset(train_sequences, train_labels, max_length=max_len)
    val_dataset = SilkDataset(val_sequences, val_labels, max_length=max_len)

    model = SilkModelV3()
    model._freeze_layers(freeze_layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if "esm" in n and p.requires_grad],
            "lr": lr_bert
        },
        {
            "params": [p for n, p in param_optimizer if "esm" not in n],
            "lr": lr_head
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    training_args = TrainingArguments(
        output_dir=model_save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(model_save_dir, "logs"),
        logging_strategy="epoch",
        learning_rate=lr_bert,
        fp16=fp16,
        load_best_model_at_end=True,
        metric_for_best_model="heavy_f1",
        greater_is_better=True,
        seed=42,
        save_total_limit=3,
        save_safetensors=False,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    trainer = SilkTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )
    
    trainer.label_names = ["labels"]
    
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Gradient accumulation: 4, Effective batch: {batch_size * 4}")
    trainer.data_collator = train_collator
    trainer.train()

    logger.info("Training completed, saving best model...")
    
    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint and os.path.exists(os.path.join(best_checkpoint, "pytorch_model.bin")):
        best_model_path = os.path.join(model_save_dir, "best_model")
        os.makedirs(best_model_path, exist_ok=True)
        
        shutil.copy(os.path.join(best_checkpoint, "pytorch_model.bin"),
                    os.path.join(best_model_path, "pytorch_model.bin"))
        logger.info(f"Copied best checkpoint from {best_checkpoint} to {best_model_path}")
    else:
        best_model_path = os.path.join(model_save_dir, "best_model")
        os.makedirs(best_model_path, exist_ok=True)
        torch.save(trainer.model.state_dict(), os.path.join(best_model_path, "pytorch_model.bin"))
        logger.info(f"Saved trainer.model to {best_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    tokenizer.save_pretrained(best_model_path)

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_CONFIG['model_name'])
    config.save_pretrained(best_model_path)

    logger.info(f"Best model saved to: {best_model_path}")

def test_model(
    test_light_file: str,
    test_heavy_file: str,
    test_p25_file: str,
    test_neg_file: str,
    model_dir: str,
    output_dir: str,
    batch_size: int = 8,
    threshold: float = None,
    strict: bool = True
) -> Dict:
    if threshold is None:
        threshold = PREDICTION_CONFIG['silk_threshold']

    logger.info("Loading test data...")

    light_sequences = read_fasta(test_light_file)
    heavy_sequences = read_fasta(test_heavy_file)
    p25_sequences = read_fasta(test_p25_file)
    neg_sequences = read_fasta(test_neg_file)

    test_sequences = light_sequences + heavy_sequences + p25_sequences + neg_sequences
    test_labels = ([1] * len(light_sequences) +
                  [2] * len(heavy_sequences) +
                  [3] * len(p25_sequences) +
                  [0] * len(neg_sequences))

    logger.info(f"Test set - Light: {len(light_sequences)}, Heavy: {len(heavy_sequences)}, P25: {len(p25_sequences)}, Negative: {len(neg_sequences)}")

    test_dataset = SilkDataset(test_sequences, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = load_v2_to_v3(model_dir, device)

    all_predictions = []
    all_joint_probs = []
    all_labels = []
    all_seq_ids = []
    all_details = []
    all_repeat_densities = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            enhanced_features = batch["enhanced_features"].to(device)
            labels = batch["labels"].to(device)
            seq_ids = batch["seq_id"]

            logits = model(input_ids, attention_mask, enhanced_features)
            joint_probs = torch.softmax(logits, dim=1).cpu().numpy()

            for i in range(len(seq_ids)):
                sequence = dict(test_sequences).get(seq_ids[i], "")
                
                final_class, confidence, details = hierarchical_predict_v3(
                    sequence, joint_probs[i], 0.0, threshold, strict_mode=strict
                )

                all_predictions.append(final_class)
                all_joint_probs.append(joint_probs[i])
                all_details.append(details)
                all_repeat_densities.append(0.0)

            all_labels.extend(labels.cpu().numpy())
            all_seq_ids.extend(seq_ids)

    accuracy = accuracy_score(all_labels, all_predictions)

    silk_true = (np.array(all_labels) > 0).astype(int)
    silk_pred = (np.array(all_predictions) > 0).astype(int)
    silk_accuracy = accuracy_score(silk_true, silk_pred)

    heavy_recall = recall_score(np.array(all_labels) == 2, np.array(all_predictions) == 2, zero_division=0)
    heavy_precision = precision_score(np.array(all_labels) == 2, np.array(all_predictions) == 2, zero_division=0)
    heavy_f1 = f1_score(np.array(all_labels) == 2, np.array(all_predictions) == 2, zero_division=0)

    fp_heavy = ((np.array(all_predictions) == 2) & (np.array(all_labels) != 2)).sum()
    fpr_heavy = fp_heavy / len(all_labels)

    fp_silk = ((np.array(all_predictions) > 0) & (np.array(all_labels) == 0)).sum()
    fpr_silk = fp_silk / len(all_labels)

    class_report = classification_report(
        all_labels, all_predictions,
        target_names=[CLASS_NAMES[i] for i in range(4)],
        zero_division=0
    )

    results_df = pd.DataFrame({
        'sequence_id': all_seq_ids,
        'true_label': all_labels,
        'predicted_label': all_predictions,
        'confidence': [d['joint_probabilities'][CLASS_NAMES[p]] for d, p in zip(all_details, all_predictions)],
        'rule_reason': [d['rule_validation'] for d in all_details],
        'rule_corrected': [d['rule_corrected'] for d in all_details],
        'biological_pass': [d.get('biological_pass', False) for d in all_details],
        'repeat_density': all_repeat_densities,
        'correct': [1 if t == p else 0 for t, p in zip(all_labels, all_predictions)]
    })
    results_df['true_label_name'] = results_df['true_label'].map(CLASS_NAMES)
    results_df['predicted_label_name'] = results_df['predicted_label'].map(CLASS_NAMES)

    results_path = os.path.join(output_dir, 'test_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Detailed prediction results saved to: {results_path}")

    missed_heavy = results_df[(results_df['true_label'] == 2) & (results_df['predicted_label'] != 2)]
    if len(missed_heavy) > 0:
        logger.warning(f"!!! {len(missed_heavy)} TRUE HEAVY CHAIN proteins were missed !!!")
        for _, row in missed_heavy.head(10).iterrows():
            logger.warning(f"  Missed: {row['sequence_id']}")
            logger.warning(f"    Predicted as: {row['predicted_label_name']}")
            logger.warning(f"    Confidence: {row['confidence']:.4f}")
            logger.warning(f"    Rule reason: {row['rule_reason']}")
    else:
        logger.info("All true heavy chain proteins correctly identified!")

    false_heavy = results_df[(results_df['true_label'] != 2) & (results_df['predicted_label'] == 2)]
    logger.warning(f"!!! {len(false_heavy)} FALSE HEAVY CHAIN predictions !!!")
    if len(false_heavy) > 0:
        for _, row in false_heavy.head(10).iterrows():
            logger.warning(f"  False: {row['sequence_id']} (true={row['true_label_name']})")
            logger.warning(f"    Confidence: {row['confidence']:.4f}")
            logger.warning(f"    Rule reason: {row['rule_reason']}")

    logger.info(" " + "="*60)
    logger.info("Test Results:")
    logger.info(f"Accuracy:        {accuracy:.4f}")
    logger.info(f"Silk Accuracy:   {silk_accuracy:.4f}")
    logger.info(f"Heavy Recall:    {heavy_recall:.4f}  <-- CRITICAL")
    logger.info(f"Heavy Precision: {heavy_precision:.4f}")
    logger.info(f"Heavy F1:        {heavy_f1:.4f}")
    logger.info(f"Heavy FPR:       {fpr_heavy:.4f}  <-- KEY METRIC")
    logger.info(f"Silk FPR:        {fpr_silk:.4f}")
    logger.info(f"Missed Heavy:    {len(missed_heavy)}")
    logger.info(f"False Heavy:     {len(false_heavy)}")
    logger.info(" Classification Report:")
    logger.info(class_report)
    logger.info("="*60)

    try:
        plot_test_results(all_labels, all_predictions, all_joint_probs, output_dir)
    except Exception as e:
        logger.warning(f"Failed to plot test results: {e}")

    return {
        'accuracy': accuracy,
        'silk_accuracy': silk_accuracy,
        'heavy_recall': heavy_recall,
        'heavy_precision': heavy_precision,
        'heavy_f1': heavy_f1,
        'fpr_heavy': fpr_heavy,
        'fpr_silk': fpr_silk,
        'classification_report': class_report,
        'results_df': results_df,
        'missed_heavy': missed_heavy,
        'false_heavy': false_heavy
    }

def predict(
    predict_dir: str,
    output_path: str,
    model_dir: str,
    batch_size: int = 8,
    threshold: float = None,
    strict: bool = True,
    strict_silk_threshold: float = None,
    strict_type_threshold: float = None,
    max_silk_ratio: float = 0.02,
    apply_species_filter: bool = True,
    diagnose: bool = False,
    universal: bool = False,
    separate_output: bool = False,
    silk_fasta: Optional[str] = None
) -> None:
    if threshold is None:
        threshold = PREDICTION_CONFIG["silk_threshold"]
    if strict_silk_threshold is None:
        strict_silk_threshold = PREDICTION_CONFIG["silk_threshold"]
    if strict_type_threshold is None:
        strict_type_threshold = PREDICTION_CONFIG["type_threshold"]

    logger.info(f"Loading prediction data...")
    logger.info(f"Using thresholds: silk={threshold}, type={strict_type_threshold}")

    fasta_files = [os.path.join(predict_dir, f) for f in os.listdir(predict_dir) if f.endswith((".fa", ".fasta"))]
    if not fasta_files:
        logger.error(f"No FASTA files found in {predict_dir}")
        return
    logger.info(f"Found {len(fasta_files)} FASTA files")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_v2_to_v3(model_dir, device)
    logger.info(f"Using device: {device}")

    os.makedirs(output_path, exist_ok=True)
    all_results = []

    for fasta_file in fasta_files:
        try:
            seqs = read_fasta(fasta_file)
            fname = os.path.basename(fasta_file)
            if not seqs: continue
            species = os.path.splitext(fname)[0].replace(".pep", "")

                                                                        
            _ABBR_MAP = {
                "Eosembia_varians": "Evar", "Oligotoma_humbertiana": "Ohum",
                "Bombyx_mori": "Bmor", "Bombyx_mandarina": "Bman",
                "Apis_mellifera": "Amel", "Apis_cerana": "Acer",
                "Tribolium_castaneum": "Tcas", "Drosophila_melanogaster": "Dmel",
                "Locusta_migratoria": "Lmig", "Plutella_xylostella": "Pxyl",
                "Cotesia_vestalis": "Cves", "Acyrthosiphon_pisum": "Apis",
                "Rhodnius_prolixus": "Rpro", "Rhyacophila_brunnea": "Rbru",
                "Limnephilus_lunatus": "Llun", "Tenodera_sinensis": "Tsin",
                "Clitarchus_hookeri": "Choo", "Blattella_germanica": "Bger",
            }
            def _auto_abbr(name):
                parts = name.split("_")
                if len(parts) >= 2:
                    return parts[0][0].upper() + parts[1][:3].lower()
                return name[:4]
            sp_abbr = _ABBR_MAP.get(species, _auto_abbr(species))
            sid_to_seq = {sid: s for sid, s in seqs}
            ds = SilkDataset(seqs, max_length=MODEL_CONFIG["max_length"])
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
            results = []

            with torch.no_grad():
                for batch in dl:
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    ef = batch["enhanced_features"].to(device)
                    sids = batch["seq_id"]
                    logits = model(ids, mask, ef)
                    jp = torch.softmax(logits, 1).cpu().numpy()
                    
                    for i in range(len(sids)):
                        seq = sid_to_seq.get(sids[i], "")
                        
                        hs = strict if not universal else False
                        fc, conf, det = hierarchical_predict_v3(seq, jp[i], 0.0, threshold, strict_mode=hs, universal_mode=universal)
                        
                        if CLASS_NAMES[fc] != "Non-Silk" and conf < 0.90:
                            fc = 0
                            conf = det["joint_probabilities"]["Non-Silk"] if "joint_probabilities" in det else (1.0 - conf)
                        
                        bv, br, bc = strict_biological_validation(seq, fc)
                        if not bv and conf < 0.99:
                            fc = bc
                            conf = 1.0 - det.get("joint_probabilities", {}).get("Non-Silk", 0.5)
                            det["rule_validation"] = "bio_filter|" + br
                            det["biological_pass"] = False
                        else:
                            det["biological_pass"] = bv
                        
                        results.append({
                            "sequence_id": sids[i],
                            "prediction": CLASS_NAMES[fc],
                            "confidence": conf,
                            "seq_length": len(seq),
                            "source_file": fname
                        })

            species_csv = os.path.join(output_path, species + "_predictions.csv")
            pd.DataFrame(results).to_csv(species_csv, index=False)
            logger.info("Saved: " + species_csv)
            
                                                                    
            cls_seqs = [r for r in results if r["prediction"] == "Heavy-Chain" and r["confidence"] > 0.90]
            if cls_seqs:
                cls_seqs.sort(key=lambda x: -x['confidence'])
                top20 = cls_seqs[:20]
                
                filtered_top = []
                for r in top20:
                    seq = sid_to_seq.get(r["sequence_id"], "")
                    has_repeat, repeat_reason = has_forbidden_repeats(seq)
                    if has_repeat:
                        logger.info(f"  [Top20 Filter] Removed {r['sequence_id']}: {repeat_reason}")
                        continue
                    passes_filter, filter_reason = apply_top20_filters(seq, "Heavy-Chain")
                    if not passes_filter:
                        logger.info(f"  [Top20 Filter] Removed {r['sequence_id']}: {filter_reason}")
                        continue
                    filtered_top.append(r)
                
                                                                                            
                if filtered_top:
                    top1 = filtered_top[0]
                    fp = os.path.join(output_path, sp_abbr + "HC.fa")
                    with open(fp, "w") as f:
                        seq = sid_to_seq.get(top1["sequence_id"], "")
                        if seq:
                            header = ">" + sp_abbr + "HC source=" + top1["sequence_id"] + " confidence=" + str(top1["confidence"])
                            f.write(header + "\n" + seq + "\n")
                    logger.info("  Heavy-Chain top-1: " + sp_abbr + "HC -> " + fp)
            all_results.extend(results)
        except Exception as e:
            logger.error("Error: " + str(fasta_file) + ": " + str(e))

    if all_results:
        df = pd.DataFrame(all_results)
        
        df['biological_pass'] = True
        df['heavy_probability'] = 0.0
        df['light_probability'] = 0.0
        df['p25_probability'] = 0.0
        df['silk_probability'] = 0.0
        
        df = strict_filter_predictions(df)
        
        if apply_species_filter:
            df = species_level_filter(df, max_silk_ratio=max_silk_ratio)
        
        all_results = df.to_dict('records')
        
        df.to_csv(os.path.join(output_path, "all_predictions.csv"), index=False)
        summary = []
        for fname in fasta_files:
            sp = os.path.splitext(os.path.basename(fname))[0].replace(".pep", "")
            sd = df[df["source_file"] == os.path.basename(fname)]
            if len(sd) == 0: continue
            h = sd[sd["prediction"] == "Heavy-Chain"]
            l = sd[sd["prediction"] == "Light-Chain"]
            p25 = sd[sd["prediction"] == "P25"]
            th = h.nlargest(1, "confidence") if len(h) > 0 else pd.DataFrame()
            summary.append({
                "species": sp,
                "total": len(sd),
                "heavy": len(h),
                "light": len(l),
                "p25": len(p25),
                "top_heavy": th.iloc[0]["sequence_id"] if len(th) > 0 else "",
                "top_heavy_prob": round(th.iloc[0]["confidence"], 4) if len(th) > 0 else 0
            })
        pd.DataFrame(summary).to_csv(os.path.join(output_path, "silk_summary.csv"), index=False)
        logger.info("Summary saved to " + output_path + "/silk_summary.csv")
        cc = Counter(r["prediction"] for r in all_results)
        logger.info("\n" + "=" * 60)
        logger.info("Prediction: " + str(len(df)) + " seqs, " + str(len(fasta_files)) + " species")
        for cn in [CLASS_NAMES[i] for i in range(4)]:
            logger.info("  " + cn + ": " + str(cc.get(cn, 0)))
        logger.info("=" * 60)
    else:
        logger.warning("No results to summarize!")

def main():
    parser = argparse.ArgumentParser(description="Insect Silk Classification Model V3 (ESM-2 + Simplified Architecture)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--train_light", type=str, required=True)
    train_parser.add_argument("--train_heavy", type=str, required=True)
    train_parser.add_argument("--train_p25", type=str, required=True)
    train_parser.add_argument("--train_neg", type=str, required=True)
    train_parser.add_argument("--val_light", type=str, default=None)
    train_parser.add_argument("--val_heavy", type=str, default=None)
    train_parser.add_argument("--val_p25", type=str, default=None)
    train_parser.add_argument("--val_neg", type=str, default=None)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--max_len", type=int, default=512)
    train_parser.add_argument("--lr_bert", type=float, default=1e-5)
    train_parser.add_argument("--lr_head", type=float, default=1e-4)
    train_parser.add_argument("--fp16", action="store_true", default=True)
    train_parser.add_argument("--model_save_dir", type=str, default="./checkpoints_v3")
    train_parser.add_argument("--validation_split", type=float, default=0.2)
    train_parser.add_argument("--freeze_layers", type=int, default=20)

    test_parser = subparsers.add_parser("test", help="Test the model")
    test_parser.add_argument("--test_light", type=str, required=True)
    test_parser.add_argument("--test_heavy", type=str, required=True)
    test_parser.add_argument("--test_p25", type=str, required=True)
    test_parser.add_argument("--test_neg", type=str, required=True)
    test_parser.add_argument("--model_dir", type=str, default="./checkpoints_v3")
    test_parser.add_argument("--output_dir", type=str, default="./test_results_v3")
    test_parser.add_argument("--batch_size", type=int, default=8)
    test_parser.add_argument("--threshold", type=float, default=None)
    test_parser.add_argument("--no_strict", action="store_true", help="Disable strict mode in hierarchical prediction")

    predict_parser = subparsers.add_parser("predict", help="Make predictions on new data")
    predict_parser.add_argument("--predict_dir", type=str, required=True)
    predict_parser.add_argument("--output", type=str, required=True)
    predict_parser.add_argument("--model_dir", type=str, default="./checkpoints_v3")
    predict_parser.add_argument("--batch_size", type=int, default=8)
    predict_parser.add_argument("--separate", action="store_true")
    predict_parser.add_argument("--silk_fasta", type=str, default=None)
    predict_parser.add_argument("--threshold", type=float, default=None)
    predict_parser.add_argument("--strict", action="store_true", default=True)
    predict_parser.add_argument("--no_strict", action="store_true")
    predict_parser.add_argument("--strict_silk_threshold", type=float, default=None)
    predict_parser.add_argument("--strict_type_threshold", type=float, default=None)
    predict_parser.add_argument("--max_silk_ratio", type=float, default=0.02)
    predict_parser.add_argument("--no_species_filter", action="store_true")
    predict_parser.add_argument("--diagnose", action="store_true", help="Diagnosis mode: output raw probabilities without any filtering")
    predict_parser.add_argument("--universal", action="store_true", help="Universal mode: relaxed thresholds + continuous bio-informed scoring for cross-order insect silk prediction")

    args = parser.parse_args()

    if args.command == "train":
        for f in [args.train_light, args.train_heavy, args.train_p25, args.train_neg]:
            if not os.path.exists(f):
                logger.error(f"File not found: {f}")
                return
        for f in [args.val_light, args.val_heavy, args.val_p25, args.val_neg]:
            if f and not os.path.exists(f):
                logger.error(f"Validation file not found: {f}")
                return

        os.makedirs(args.model_save_dir, exist_ok=True)

        train_model(
            train_light_file=args.train_light,
            train_heavy_file=args.train_heavy,
            train_p25_file=args.train_p25,
            train_neg_file=args.train_neg,
            val_light_file=args.val_light,
            val_heavy_file=args.val_heavy,
            val_p25_file=args.val_p25,
            val_neg_file=args.val_neg,
            model_save_dir=args.model_save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_len=args.max_len,
            lr_bert=args.lr_bert,
            lr_head=args.lr_head,
            fp16=args.fp16,
            validation_split=args.validation_split,
            freeze_layers=args.freeze_layers
        )

    elif args.command == "test":
        for f in [args.test_light, args.test_heavy, args.test_p25, args.test_neg]:
            if not os.path.exists(f):
                logger.error(f"File not found: {f}")
                return

        os.makedirs(args.output_dir, exist_ok=True)

        test_model(
            test_light_file=args.test_light,
            test_heavy_file=args.test_heavy,
            test_p25_file=args.test_p25,
            test_neg_file=args.test_neg,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            threshold=args.threshold,
            strict=not args.no_strict
        )

    elif args.command == "predict":
        if not os.path.exists(args.predict_dir):
            logger.error(f"Prediction directory not found: {args.predict_dir}")
            return

        if args.separate:
            os.makedirs(args.output, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

        strict_mode = not args.no_strict if args.no_strict else args.strict

        predict(
            predict_dir=args.predict_dir,
            output_path=args.output,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            separate_output=args.separate,
            silk_fasta=args.silk_fasta,
            threshold=args.threshold,
            strict=strict_mode,
            strict_silk_threshold=args.strict_silk_threshold,
            strict_type_threshold=args.strict_type_threshold,
            max_silk_ratio=args.max_silk_ratio,
            apply_species_filter=not args.no_species_filter,
            diagnose=args.diagnose,
            universal=args.universal
        )

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
