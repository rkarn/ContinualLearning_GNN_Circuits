import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import dgl.nn as dglnn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# 1. Load & preprocess data
# -------------------------
df = pd.read_csv("all_circuits_features.csv")

# encode gate types → integers
le = LabelEncoder()
df["gate_label"] = le.fit_transform(df["gate_type"].astype(str))

# normalize features
feat_cols = [
    "fan_in","fan_out","dist_to_output",
    "is_primary_input","is_primary_output","is_internal","is_key_gate",
    "degree_centrality","betweenness_centrality",
    "closeness_centrality","clustering_coefficient",
    "avg_fan_in_neighbors","avg_fan_out_neighbors"
]
df = df.dropna(subset=feat_cols)
df[feat_cols] = df[feat_cols].astype(float)
df[feat_cols] = StandardScaler().fit_transform(df[feat_cols])

# map node name → index
nodes = df["node"].tolist()
node2idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)

# -------------------------
# 2. Build DGL graph
# -------------------------
edges = []
src_cands = df[df["fan_out"] > 0]["node"].tolist()
for _, row in df.iterrows():
    tgt = node2idx[row["node"]]
    k   = int(row["fan_in"])
    for s in src_cands[:k]:
        edges.append((node2idx[s], tgt))
edges = list(set(edges))
if edges:
    src, dst = zip(*edges)
else:
    src, dst = [], []
graph = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=N)
graph = dgl.add_self_loop(graph)
graph.ndata["feat"]  = torch.tensor(df[feat_cols].values, dtype=torch.float32)
graph.ndata["label"] = torch.tensor(df["gate_label"].values, dtype=torch.long)

# -------------------------
# 3. Define tasks by gate groups
# -------------------------
group_defs = [
    ["and","or"],
    ["nand","nor"],
    ["xor","xnor"],
    ["buf","not"]
]
valid_groups = []
train_masks, test_masks = [], []

all_types = set(df["gate_type"])
for grp in group_defs:
    # keep only gates actually in dataset
    present = [g for g in grp if g in all_types]
    if not present:
        continue
    valid_groups.append(present)
    idxs = [node2idx[n] for n in df[df["gate_type"].isin(present)]["node"]]
    tr, te = train_test_split(idxs, test_size=0.2, random_state=42)
    tm = torch.zeros((N,), dtype=torch.bool)
    vm = torch.zeros((N,), dtype=torch.bool)
    tm[tr] = True
    vm[te] = True
    train_masks.append(tm)
    test_masks.append(vm)

# -------------------------
# 4. Define GCN model
# -------------------------
class GCN(nn.Module):
    def __init__(self, in_feats, hid, num_cls):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid, num_cls, allow_zero_in_degree=True)
    def forward(self, g, x):
        h = torch.relu(self.conv1(g, x))
        return self.conv2(g, h)

# -------------------------
# 5. EWC helper
# -------------------------
class EWC:
    def __init__(self, model, graph, mask, device):
        self.device = device
        self.params = {
            n: p.clone().detach().to(device)
            for n, p in model.named_parameters() if p.requires_grad
        }
        self.fisher = self._compute_fisher(model, graph, mask)

    def _compute_fisher(self, model, graph, mask):
        fisher = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in model.named_parameters() if p.requires_grad
        }
        model.zero_grad()
        logits = model(graph, graph.ndata["feat"].to(self.device))
        logp   = torch.log_softmax(logits, dim=1)
        labels = graph.ndata["label"].to(self.device)
        idxs   = mask.nonzero().squeeze()
        loss   = 0
        for i in idxs:
            i = i.item()
            loss += -logp[i, labels[i]]
        loss = loss / len(idxs)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad:
                fisher[n] = p.grad.data.clone()**2
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.params[n])**2).sum()
        return loss

# -------------------------
# 6. Training w/ EWC
# -------------------------
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model      = GCN(len(feat_cols), 64, len(le.classes_)).to(device)
opt        = optim.Adam(model.parameters(), lr=1e-2)
criterion  = nn.CrossEntropyLoss()
graph      = graph.to(device)

ewc_list   = []
lambda_ewc = 1000.0
results    = {}

for stage, (tr_mask, te_mask, grp) in enumerate(
        zip(train_masks, test_masks, valid_groups), start=1):
    print(f"\n=== Stage {stage}: learning {grp} ===")
    tr_mask = tr_mask.to(device)
    te_mask = te_mask.to(device)

    # train current group only
    model.train()
    for epoch in range(1, 51):
        logits   = model(graph, graph.ndata["feat"])
        loss_c   = criterion(logits[tr_mask], graph.ndata["label"][tr_mask])
        if ewc_list:
            pen = sum([m.penalty(model) for m in ewc_list])
            loss = loss_c + (lambda_ewc/2) * pen
        else:
            loss = loss_c

        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch in (1, 10, 50):
            print(f"  Epoch {epoch:02d}, loss: {loss.item():.4f}")

    # compute Fisher for this task
    ewc_list.append(EWC(model, graph, tr_mask, device))

    # evaluation
    model.eval()
    with torch.no_grad():
        logits = model(graph, graph.ndata["feat"]).cpu().numpy()
        preds  = logits.argmax(axis=1)
        results[stage] = {}
        for i in range(stage):
            mask  = test_masks[i]
            idxs  = mask.nonzero().squeeze().cpu().numpy()
            y_true = graph.ndata["label"][idxs].cpu().numpy()
            y_pred = preds[idxs]
            acc = accuracy_score(y_true, y_pred)
            results[stage][f"group_{i+1}"] = acc
            print(f"  → Acc group {i+1} {valid_groups[i]}: {acc*100:.2f}%")

        # confusion matrix for current group
        cm_idxs = te_mask.nonzero().squeeze().cpu().numpy()
        cm_true = graph.ndata["label"][cm_idxs].cpu().numpy()
        cm_pred = preds[cm_idxs]
        # labels = integer codes of this group's gates
        lbls = le.transform(grp)
        cm = confusion_matrix(cm_true, cm_pred, labels=lbls)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                    xticklabels=grp, yticklabels=grp)
        plt.title(f"Confusion — {grp}")
        plt.xlabel("Pred"); plt.ylabel("True")
        plt.show()

# -------------------------
# 7. Results summary
# -------------------------
df_res = pd.DataFrame(results).T
df_res.index.name = "Stage"
print("\n=== Test accuracies per stage & group ===")
print(df_res.applymap(lambda v: f"{v*100:5.2f}%"))

# -------------------------
# 8. Interpretability Dashboard for EWC (Two Plots with Arrows)
# -------------------------
import seaborn as sns
import matplotlib.pyplot as plt

# 8.1 Ensure 'from_to' label exists
if "from_to" not in drift_df.columns:
    drift_df["from_to"] = drift_df.apply(
        lambda r: f"{int(r['from_task'])}→{int(r['to_task'])}", axis=1
    )

# 8.2 Print the drift DataFrame and correlations
print("\n=== EWC Interpretability: Drift vs Forgetting ===")
print(drift_df[["from_to","param_drift","ewc_drift","delta_acc"]])

corr_param = drift_df["param_drift"].corr(drift_df["delta_acc"])
corr_ewc   = drift_df["ewc_drift"].corr(drift_df["delta_acc"])
print(f"\nCorrelation (param drift vs ΔAcc): {corr_param:.4f}")
print(f"Correlation (ewc drift   vs ΔAcc): {corr_ewc:.4f}")