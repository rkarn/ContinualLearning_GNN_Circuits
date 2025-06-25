import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 2. Build single DGL graph
# ------------------------------------------------
edges = []
src_cands = df[df["fan_out"] > 0]["node"].tolist()
for _, row in df.iterrows():
    tgt = node2idx[row["node"]]
    for s in src_cands[: int(row["fan_in"])]:
        edges.append((node2idx[s], tgt))
edges = list(set(edges))
src, dst = zip(*edges) if edges else ([], [])
g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=N)
g = dgl.add_self_loop(g)
g.ndata["feat"]  = torch.tensor(df[feat_cols].values, dtype=torch.float32)
g.ndata["label"] = torch.tensor(df["gate_label"].values, dtype=torch.long)

# ------------------------------------------------
# 3. Define tasks by gate‐type groups
# ------------------------------------------------
group_defs = [
    ["and","or"],
    ["nand","nor"],
    ["xor","xnor"],
    ["buf","not"]
]
all_types = set(df["gate_type"])
valid_groups, train_masks, test_masks = [], [], []
for grp in group_defs:
    present = [gtype for gtype in grp if gtype in all_types]
    if not present:
        continue
    valid_groups.append(present)
    idxs = [node2idx[n] for n in df[df["gate_type"].isin(present)]["node"]]
    tr, te = train_test_split(idxs, test_size=0.2, random_state=42)
    tm = torch.zeros((N,), dtype=torch.bool); tm[tr] = True
    vm = torch.zeros((N,), dtype=torch.bool); vm[te] = True
    train_masks.append(tm); test_masks.append(vm)

# ------------------------------------------------
# 4. Define 2‐layer GCN model
# ------------------------------------------------
class GCN(nn.Module):
    def __init__(self, in_feats, hid, num_cls):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid,   num_cls, allow_zero_in_degree=True)
    def forward(self, graph, x):
        h = F.relu(self.conv1(graph, x))
        return self.conv2(graph, h)

# ------------------------------------------------
# 5. iCaRL‐style replay + distillation setup
# ------------------------------------------------
memory_per_class = 20
temperature      = 2.0
lambda_distill   = 1.0
kl_loss          = nn.KLDivLoss(reduction='batchmean')
exemplar_sets    = {}

def update_exemplars(cls, cand_idxs):
    np.random.shuffle(cand_idxs)
    exemplar_sets[cls] = cand_idxs[:memory_per_class]

def get_all_exemplars():
    return [idx for idxs in exemplar_sets.values() for idx in idxs]

# ------------------------------------------------
# 6. Training loop w/ iCaRL + snapshot collection
# ------------------------------------------------
device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model             = GCN(len(feat_cols), 64, len(le.classes_)).to(device)
opt               = optim.Adam(model.parameters(), lr=1e-2)
criterion         = nn.CrossEntropyLoss()
g                 = g.to(device)
results           = {}
param_snapshots   = []   # list of state_dicts (CPU tensors)
logits_snapshots  = []   # list of logits (N×C) after each stage
memory_snapshots  = []   # list of exemplar idx lists after each stage
old_model         = None

for stage, (tr_mask, te_mask, grp) in enumerate(
        zip(train_masks, test_masks, valid_groups), start=1):
    print(f"\n=== Stage {stage}: learning {grp} ===")
    tr_mask = tr_mask.to(device)
    te_mask = te_mask.to(device)
    new_idxs = tr_mask.nonzero().squeeze().tolist()

    # frozen copy for distillation
    if stage > 1:
        old_model = copy.deepcopy(model).eval().to(device)

    exemplar_idxs = get_all_exemplars()
    train_idxs    = new_idxs + exemplar_idxs
    train_idxs_t  = torch.tensor(train_idxs, dtype=torch.long, device=device)

    if old_model and exemplar_idxs:
        with torch.no_grad():
            old_logits = old_model(g, g.ndata["feat"])
            old_soft   = F.softmax(old_logits[exemplar_idxs] / temperature, dim=1)

    model.train()
    for epoch in range(1, 51):
        logits_out = model(g, g.ndata["feat"])
        ce_loss    = criterion(logits_out[train_idxs_t], g.ndata["label"][train_idxs_t])
        if old_model and exemplar_idxs:
            cur_logp      = F.log_softmax(logits_out[exemplar_idxs]/temperature, dim=1)
            distill_loss = kl_loss(cur_logp, old_soft) * (temperature**2)
            loss         = ce_loss + lambda_distill * distill_loss
        else:
            loss = ce_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch in (1, 10, 50):
            print(f"  Epoch {epoch:02d}, loss: {loss.item():.4f}")

    # update exemplars
    for gate in grp:
        cls = le.transform([gate])[0]
        cand_idxs = [i for i in new_idxs if g.ndata["label"][i].item() == cls]
        update_exemplars(cls, cand_idxs)

    # evaluation
    model.eval()
    with torch.no_grad():
        logits_full = model(g, g.ndata["feat"]).cpu()
        preds       = logits_full.argmax(dim=1).numpy()
        results[stage] = {}
        for i in range(stage):
            idxs = test_masks[i].nonzero().squeeze().cpu().numpy()
            y_true = g.ndata["label"][idxs].cpu().numpy()
            y_pred = preds[idxs]
            acc     = accuracy_score(y_true, y_pred)
            results[stage][f"group_{i+1}"] = acc
            print(f"  → Acc group {i+1} {valid_groups[i]}: {acc*100:.2f}%")

        cm = confusion_matrix(
            g.ndata["label"][te_mask].cpu().numpy(),
            preds[te_mask.nonzero().squeeze().cpu().numpy()],
            labels=[le.transform([gtype])[0] for gtype in grp]
        )
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=grp, yticklabels=grp)
        plt.title(f"Confusion — {grp}")
        plt.xlabel("Pred"); plt.ylabel("True")
        plt.show()

    # snapshots after stage
    param_snapshots.append({
        name: tensor.clone().detach().cpu()
        for name, tensor in model.state_dict().items()
    })
    logits_snapshots.append(logits_full)
    memory_snapshots.append(list(get_all_exemplars()))

# ------------------------------------------------
# 7. Summarize results
# ------------------------------------------------
df_res = pd.DataFrame(results).T
df_res.index.name = "Stage"
print("\n=== Test accuracies per stage & group ===")
print(df_res.applymap(lambda v: f"{v*100:5.2f}%"))

# ------------------------------------------------
# 8. Interpretability Dashboard for iCaRL
# ------------------------------------------------
records = []
num_tasks = len(param_snapshots)

for t in range(2, num_tasks + 1):
    state_t  = param_snapshots[t-1]
    logits_t = logits_snapshots[t-1]
    for k in range(1, t):
        state_k   = param_snapshots[k-1]
        logits_k  = logits_snapshots[k-1]
        M_k       = memory_snapshots[k-1]
        N_k       = len(M_k)

        # parameter drift
        sq = 0.0
        for name in state_k:
            d = state_t[name] - state_k[name]
            sq += d.pow(2).sum().item()
        param_drift = np.sqrt(sq)

        # distillation loss
        if N_k > 0:
            diffs = logits_t[M_k] - logits_k[M_k]
            distill_loss = diffs.pow(2).sum(dim=1).mean().item()
        else:
            distill_loss = np.nan

        # forgetting
        acc_kk = results[k][f"group_{k}"]
        acc_kt = results[t][f"group_{k}"]
        delta_acc = acc_kk - acc_kt

        records.append({
            "from_task":    k,
            "to_task":      t,
            "param_drift":  param_drift,
            "distill_loss": distill_loss,
            "delta_acc":    delta_acc
        })

ica_df = pd.DataFrame(records)
ica_df["from_to"] = ica_df.apply(lambda r: f"{int(r['from_task'])}→{int(r['to_task'])}", axis=1)

print("\n=== iCaRL Interpretability: Drift vs Forgetting ===")
print(ica_df[["from_to","param_drift","distill_loss","delta_acc"]])

corr_param   = ica_df["param_drift"].corr(ica_df["delta_acc"])
corr_distill = ica_df["distill_loss"].corr(ica_df["delta_acc"])
print(f"\nCorrelation (param drift vs ΔAcc):   {corr_param:.4f}")
print(f"Correlation (distill loss vs ΔAcc):   {corr_distill:.4f}")
