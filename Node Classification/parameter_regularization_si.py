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

# ------------------------------------------------
# 1. Load & preprocess data
# ------------------------------------------------
df = pd.read_csv("all_circuits_features.csv")

# encode gate types → integer labels
le = LabelEncoder()
df["gate_label"] = le.fit_transform(df["gate_type"].astype(str))

# normalize features
feat_cols = [
    "fan_in", "fan_out", "dist_to_output",
    "is_primary_input", "is_primary_output",
    "is_internal", "is_key_gate",
    "degree_centrality", "betweenness_centrality",
    "closeness_centrality", "clustering_coefficient",
    "avg_fan_in_neighbors", "avg_fan_out_neighbors"
]
df = df.dropna(subset=feat_cols)
df[feat_cols] = df[feat_cols].astype(float)
df[feat_cols] = StandardScaler().fit_transform(df[feat_cols])

# map node name → index
nodes = df["node"].tolist()
node2idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)

# ------------------------------------------------
# 2. Build DGL graph once
# ------------------------------------------------
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

# ------------------------------------------------
# 3. Define tasks by gate-type groups
# ------------------------------------------------
group_defs = [
    ["and",  "or" ],
    ["nand", "nor"],
    ["xor",  "xnor"],
    ["buf",  "not"]
]
all_types = set(df["gate_type"])
valid_groups, train_masks, test_masks = [], [], []

for grp in group_defs:
    present = [g for g in grp if g in all_types]
    if not present:
        continue
    valid_groups.append(present)
    idxs = [node2idx[n] for n in df[df["gate_type"].isin(present)]["node"]]
    tr, te = train_test_split(idxs, test_size=0.2, random_state=42)
    tm = torch.zeros((N,), dtype=torch.bool)
    vm = torch.zeros((N,), dtype=torch.bool)
    tm[tr] = True; vm[te] = True
    train_masks.append(tm)
    test_masks.append(vm)

# ------------------------------------------------
# 4. Define 2-layer GCN
# ------------------------------------------------
class GCN(nn.Module):
    def __init__(self, in_feats, hid, num_cls):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid, num_cls, allow_zero_in_degree=True)

    def forward(self, g, x):
        h = torch.relu(self.conv1(g, x))
        return self.conv2(g, h)

# ------------------------------------------------
# 5. Synaptic Intelligence (SI) implementation
# ------------------------------------------------
class SynapticIntelligence:
    def __init__(self, model, xi=0.1):
        self.xi = xi
        # total importance ω_i
        self.omega = {
            n: torch.zeros_like(p.data) 
            for n, p in model.named_parameters() if p.requires_grad
        }
        # snapshot of parameters after last task
        self.theta_old = {
            n: p.data.clone().detach() 
            for n, p in model.named_parameters() if p.requires_grad
        }

    def begin_task(self, model):
        # snapshot at start of new task
        self.prev_theta = {
            n: p.data.clone().detach() 
            for n, p in model.named_parameters() if p.requires_grad
        }
        # path integral accumulator
        self.path_omega = {
            n: torch.zeros_like(p.data) 
            for n, p in model.named_parameters() if p.requires_grad
        }

    def accumulate(self, model):
        # call after backward() before optimizer.step()
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            grad = p.grad.data
            delta = (p.data - self.prev_theta[n])
            self.path_omega[n] += - grad * delta
            # update prev_theta for next step
            self.prev_theta[n] = p.data.clone().detach()

    def end_task(self, model):
        # compute and accumulate omega for this task
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            delta = p.data - self.theta_old[n]
            # avoid division by zero
            denom = delta.pow(2) + self.xi
            omega_task = self.path_omega[n] / denom
            self.omega[n] += omega_task
            # update theta_old for next tasks
            self.theta_old[n] = p.data.clone().detach()

    def penalty(self, model):
        # regularization loss Σ ω_i (θ_i - θ_old_i)²
        loss = 0
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            loss += (self.omega[n] * (p - self.theta_old[n]).pow(2)).sum()
        return loss

# ------------------------------------------------
# 6. Training loop w/ SI regularization
# ------------------------------------------------
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model      = GCN(in_feats=len(feat_cols), hid=64, num_cls=len(le.classes_)).to(device)
opt        = optim.Adam(model.parameters(), lr=1e-2)
criterion  = nn.CrossEntropyLoss()
graph      = graph.to(device)

si = SynapticIntelligence(model, xi=0.1)
lambda_si = 1000.0
results    = {}
# Before the training loop, initialize lists to hold per‐task snapshots
theta_snapshots = []
omega_snapshots = []

for stage, (tr_mask, te_mask, grp) in enumerate(
        zip(train_masks, test_masks, valid_groups), start=1):
    print(f"\n=== Stage {stage}: learning {grp} ===")
    tr_mask = tr_mask.to(device)
    te_mask = te_mask.to(device)

    # prepare SI
    si.begin_task(model)

    # train current task only
    model.train()
    for epoch in range(1, 51):
        logits = model(graph, graph.ndata["feat"])
        loss_c = criterion(logits[tr_mask], graph.ndata["label"][tr_mask])
        # add SI regularization from previous tasks
        if stage > 1:
            loss = loss_c + lambda_si * si.penalty(model)
        else:
            loss = loss_c

        opt.zero_grad()
        loss.backward()
        # accumulate path‐integral
        si.accumulate(model)
        opt.step()

        if epoch in (1, 10, 50):
            print(f"  Epoch {epoch:02d}, loss: {loss.item():.4f}")

    # finish SI for this task
    si.end_task(model)
    
    # ------------------------------------------------------------
    # After finishing task `stage`, snapshot θ_old and Ω
    theta_snapshots.append({
        name: tensor.clone().detach().cpu()
        for name, tensor in si.theta_old.items()
    })
    omega_snapshots.append({
        name: tensor.clone().detach().cpu()
        for name, tensor in si.omega.items()
    })
    # ------------------------------------------------------------

    # evaluate on all seen tasks
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
            print(f"  → Acc on group {i+1} {valid_groups[i]}: {acc*100:.2f}%")

        # confusion matrix for current group
        cm_idxs = te_mask.nonzero().squeeze().cpu().numpy()
        cm_true = graph.ndata["label"][cm_idxs].cpu().numpy()
        cm_pred = preds[cm_idxs]
        lbls = le.transform(grp)
        cm = confusion_matrix(cm_true, cm_pred, labels=lbls)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=grp, yticklabels=grp)
        plt.title(f"Confusion — {grp}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.show()

# ------------------------------------------------
# 7. Summary of results
# ------------------------------------------------
df_res = pd.DataFrame(results).T
df_res.index.name = "Stage"
print("\n=== Test accuracies per stage & group ===")
print(df_res.applymap(lambda v: f"{v*100:5.2f}%"))

# -------------------------
# 8. Interpretability Dashboard for SI
# -------------------------
import pandas as pd
import numpy as np

# Build a table of drifts and forgetting
records = []
num_tasks = len(theta_snapshots)
for t in range(2, num_tasks + 1):
    θ_t = theta_snapshots[t - 1]
    for k in range(1, t):
        θ_k = theta_snapshots[k - 1]
        Ω_k = omega_snapshots[k - 1]

        # 1) L2 parameter drift ‖θ^(t) − θ^(k)‖₂
        sq = 0.0
        for name in θ_k:
            d = (θ_t[name] - θ_k[name])
            sq += (d.pow(2)).sum().item()
        param_drift = np.sqrt(sq)

        # 2) SI-weighted drift Σ Ω_i^(k) · (Δθ_i)²
        si_drift = 0.0
        for name in θ_k:
            d = (θ_t[name] - θ_k[name])
            si_drift += (Ω_k[name] * d.pow(2)).sum().item()

        # 3) Forgetting ΔAcc = Acc_k(k) − Acc_k(t)
        acc_kk = results[k][f"group_{k}"]
        acc_kt = results[t][f"group_{k}"]
        delta_acc = acc_kk - acc_kt

        records.append({
            "from_task": k,
            "to_task":   t,
            "param_drift": param_drift,
            "si_drift":    si_drift,
            "delta_acc":   delta_acc
        })

si_df = pd.DataFrame(records)
si_df["from_to"] = si_df.apply(lambda r: f"{int(r['from_task'])}→{int(r['to_task'])}", axis=1)

# Print the SI interpretability table
print("\n=== SI Interpretability: Drift vs Forgetting ===")
print(si_df[["from_to","param_drift","si_drift","delta_acc"]])

# Compute correlations
corr_param = si_df["param_drift"].corr(si_df["delta_acc"])
corr_si    = si_df["si_drift"].corr(si_df["delta_acc"])
print(f"\nCorrelation (param drift vs ΔAcc):        {corr_param:.4f}")
print(f"Correlation (SI-weighted drift vs ΔAcc):  {corr_si:.4f}")

# (Optional) Scatterplots can be added similarly to EWC but are omitted per request.
