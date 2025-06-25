import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. Load & preprocess data
# ------------------------------------------------
df = pd.read_csv("all_circuits_features.csv")
le = LabelEncoder()
df["gate_label"] = le.fit_transform(df["gate_type"].astype(str))

feat_cols = [
    "fan_in","fan_out","dist_to_output",
    "is_primary_input","is_primary_output","is_internal","is_key_gate",
    "degree_centrality","betweenness_centrality",
    "closeness_centrality","clustering_coefficient",
    "avg_fan_in_neighbors","avg_fan_out_neighbors"
]
df = df.dropna(subset=feat_cols)
df[feat_cols] = StandardScaler().fit_transform(df[feat_cols].astype(float))

nodes = df["node"].tolist()
node2idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)

# ------------------------------------------------
# 2. Build the DGL graph
# ------------------------------------------------
edges = []
src_cands = df[df["fan_out"] > 0]["node"].tolist()
for _, row in df.iterrows():
    tgt = node2idx[row["node"]]
    for src in src_cands[: int(row["fan_in"])]:
        edges.append((node2idx[src], tgt))
edges = list(set(edges))
src, dst = zip(*edges) if edges else ([], [])
graph = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=N)
graph = dgl.add_self_loop(graph)
graph.ndata["feat"]  = torch.tensor(df[feat_cols].values, dtype=torch.float32)
graph.ndata["label"] = torch.tensor(df["gate_label"].values, dtype=torch.long)

# ------------------------------------------------
# 3. Define tasks by gate‐type groups
# ------------------------------------------------
group_defs = [
    ["and","or"], ["nand","nor"],
    ["xor","xnor"], ["buf","not"]
]
all_types = set(df["gate_type"])
valid_groups, train_masks, test_masks = [], [], []
for grp in group_defs:
    present = [g for g in grp if g in all_types]
    if not present: continue
    valid_groups.append(present)
    idxs = [node2idx[n] for n in df[df["gate_type"].isin(present)]["node"]]
    strat = df["gate_label"].iloc[idxs]
    tr, te = train_test_split(idxs, test_size=0.2, random_state=42, stratify=strat)
    tm = torch.zeros((N,), dtype=torch.bool); tm[tr] = True
    vm = torch.zeros((N,), dtype=torch.bool); vm[te] = True
    train_masks.append(tm); test_masks.append(vm)

# ------------------------------------------------
# 4. CoPE model with BatchNorm & Dropout
# ------------------------------------------------
class CoPEModel(nn.Module):
    def __init__(self, in_feats, hid, num_cls, drop=0.5):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.bn1   = nn.BatchNorm1d(hid)
        self.conv2 = dglnn.GraphConv(hid,     hid, allow_zero_in_degree=True)
        self.bn2   = nn.BatchNorm1d(hid)
        self.drop  = nn.Dropout(drop)
        self.classifier   = nn.Linear(hid, num_cls)
        self.pretext_head = nn.Linear(hid, in_feats)

    def encode(self, g, x):
        h = F.relu(self.bn1(self.conv1(g, x)))
        h = self.drop(h)
        h = F.relu(self.bn2(self.conv2(g, h)))
        return self.drop(h)

    def forward(self, g, x):
        return self.classifier(self.encode(g, x))

    def reconstruct(self, g, x):
        return self.pretext_head(self.encode(g, x))

# ------------------------------------------------
# 5. CoPE training with snapshotting
# ------------------------------------------------
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model    = CoPEModel(len(feat_cols), hid=128,
                     num_cls=len(le.classes_), drop=0.4).to(device)
graph    = graph.to(device)
feats    = graph.ndata["feat"]

opt_pre = optim.Adam(
    list(model.conv1.parameters()) + list(model.bn1.parameters()) +
    list(model.conv2.parameters()) + list(model.bn2.parameters()) +
    list(model.pretext_head.parameters()),
    lr=5e-4, weight_decay=1e-5
)
opt_cls = optim.Adam(
    list(model.conv1.parameters()) + list(model.bn1.parameters()) +
    list(model.conv2.parameters()) + list(model.bn2.parameters()) +
    list(model.classifier.parameters()),
    lr=5e-3, weight_decay=1e-5
)

results      = {}
mask_ratio   = 0.2
pre_epochs   = 50
cls_epochs   = 100
num_cls      = len(le.classes_)

# snapshot lists
param_snapshots     = []
pretext_snapshots   = []

# global self‐supervised warm‐up (no snapshot)
print("Global self-supervised pretraining...")
for epoch in range(1, pre_epochs+1):
    model.train()
    masked = feats.clone()
    dims   = np.random.choice(feats.shape[1],
                              int(mask_ratio * feats.shape[1]), replace=False)
    masked[:, dims] = 0
    recon    = model.reconstruct(graph, masked)
    loss_pre = F.mse_loss(recon[:, dims], feats[:, dims])
    opt_pre.zero_grad(); loss_pre.backward(); opt_pre.step()
    if epoch in (1, 25, pre_epochs):
        print(f"  [Global Pre] {epoch}/{pre_epochs}, loss: {loss_pre:.4f}")

# stage‐wise CoPE
for stage, (tr_mask, te_mask, grp) in enumerate(
    zip(train_masks, test_masks, valid_groups), start=1):

    tr_idxs = tr_mask.nonzero().squeeze().tolist()
    te_idxs = te_mask.nonzero().squeeze().tolist()

    print(f"\n=== Stage {stage}: CoPE on {grp} ===")

    # 1) task‐specific self‐supervised pretraining
    for epoch in range(1, pre_epochs+1):
        model.train()
        masked = feats.clone()
        dims   = np.random.choice(feats.shape[1],
                                  int(mask_ratio * feats.shape[1]), replace=False)
        masked[tr_idxs][:, dims] = 0
        recon    = model.reconstruct(graph, masked)
        loss_pre = F.mse_loss(recon[tr_idxs][:, dims], feats[tr_idxs][:, dims])
        opt_pre.zero_grad(); loss_pre.backward(); opt_pre.step()
        if epoch in (1, 25, pre_epochs):
            print(f"  [Pre{stage}] {epoch}/{pre_epochs}, loss: {loss_pre:.4f}")

    # 2) supervised fine‐tuning
    label_tr = graph.ndata["label"][tr_mask].to(device)
    counts   = torch.bincount(label_tr, minlength=num_cls).float()
    weights  = (label_tr.numel() / (counts * num_cls)).clamp(max=10.0)
    weights[counts == 0] = 0.0
    criterion = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(1, cls_epochs+1):
        model.train()
        logits = model(graph, feats)
        loss   = criterion(logits[tr_idxs], label_tr)
        opt_cls.zero_grad(); loss.backward(); opt_cls.step()
        if epoch in (1, 50, cls_epochs):
            print(f"  [Cls{stage}] {epoch}/{cls_epochs}, loss: {loss:.4f}")

    # 3) evaluation
    model.eval()
    with torch.no_grad():
        logits  = model(graph, feats).cpu().numpy()
        preds   = logits.argmax(axis=1)
        results[stage] = {}
        for i in range(stage):
            idxs = test_masks[i].nonzero().squeeze().cpu().numpy()
            acc  = accuracy_score(
                graph.ndata["label"][idxs].cpu().numpy(), preds[idxs]
            )
            results[stage][f"group_{i+1}"] = acc
            print(f"  → Acc group {i+1} {valid_groups[i]}: {acc*100:.2f}%")

        # confusion
        cm = confusion_matrix(
            graph.ndata["label"][te_idxs].cpu().numpy(),
            preds[te_idxs],
            labels=le.transform(grp)
        )
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=grp, yticklabels=grp)
        plt.title(f"Confusion — {grp}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.show()

    # --- snapshots after stage ---
    # 1) full parameters θ^(stage)
    param_snapshots.append({
        name: param.clone().detach().cpu()
        for name, param in model.state_dict().items()
    })
    # 2) pretext‐head parameters f_self^(stage)
    pretext_snapshots.append({
        name: param.clone().detach().cpu()
        for name, param in model.pretext_head.state_dict().items()
    })

# ------------------------------------------------
# 6. Results summary
# ------------------------------------------------
df_res = pd.DataFrame(results).T
df_res.index.name = "Stage"
print("\n=== Test accuracies per stage & group ===")
print(df_res.applymap(lambda v: f"{v*100:5.2f}%"))

# ------------------------------------------------
# 7. Interpretability Dashboard for CoPE
# ------------------------------------------------

# A) Parameter‐Drift vs Forgetting
drift_records = []
for t in range(2, len(param_snapshots)+1):
    θ_t = param_snapshots[t-1]
    for k in range(1, t):
        θ_k = param_snapshots[k-1]
        sq   = 0.0
        for name in θ_k:
            d   = θ_t[name] - θ_k[name]
            sq += d.pow(2).sum().item()
        drift = np.sqrt(sq)
        delta = results[k][f"group_{k}"] - results[t][f"group_{k}"]
        drift_records.append({
            "from_task": k, "to_task": t,
            "param_drift": drift, "delta_acc": delta
        })
drift_df = pd.DataFrame(drift_records)
drift_df["from_to"] = drift_df.apply(lambda r: f"{r.from_task}→{r.to_task}", axis=1)
print("\n=== CoPE: Param Drift vs Forgetting ===")
print(drift_df[["from_to","param_drift","delta_acc"]])
print(f"Correlation: {drift_df.param_drift.corr(drift_df.delta_acc):.4f}")

# B) CoPE‐Shift vs Forgetting (pretext‐head drift)
cope_records = []
for t in range(2, len(pretext_snapshots)+1):
    head_t = pretext_snapshots[t-1]
    for k in range(1, t):
        head_k = pretext_snapshots[k-1]
        sq     = 0.0
        for name in head_k:
            d   = head_t[name] - head_k[name]
            sq += d.pow(2).sum().item()
        shift = np.sqrt(sq)
        delta = results[k][f"group_{k}"] - results[t][f"group_{k}"]
        cope_records.append({
            "from_task": k, "to_task": t,
            "cope_shift": shift, "delta_acc": delta
        })
cope_df = pd.DataFrame(cope_records)
cope_df["from_to"] = cope_df.apply(lambda r: f"{r.from_task}→{r.to_task}", axis=1)
print("\n=== CoPE: CoPE‐Shift vs Forgetting ===")
print(cope_df[["from_to","cope_shift","delta_acc"]])
print(f"Correlation: {cope_df.cope_shift.corr(cope_df.delta_acc):.4f}")
