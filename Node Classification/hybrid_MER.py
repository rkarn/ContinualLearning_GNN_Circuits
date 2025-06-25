import copy
import random
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
node2idx = {n:i for i,n in enumerate(nodes)}
N = len(nodes)

# ------------------------------------------------
# 2. Build single DGL graph
# ------------------------------------------------
edges = []
src_cands = df[df["fan_out"]>0]["node"].tolist()
for _, row in df.iterrows():
    tgt = node2idx[row["node"]]
    k   = int(row["fan_in"])
    for s in src_cands[:k]:
        edges.append((node2idx[s], tgt))
edges = list(set(edges))
src, dst = zip(*edges) if edges else ([],[])
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
    tr, te = train_test_split(idxs, test_size=0.2, random_state=42)
    tm = torch.zeros((N,), dtype=torch.bool); tm[tr]=True
    vm = torch.zeros((N,), dtype=torch.bool); vm[te]=True
    train_masks.append(tm); test_masks.append(vm)

# ------------------------------------------------
# 4. Define GCN
# ------------------------------------------------
class GCN(nn.Module):
    def __init__(self, in_feats, hid, num_cls):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid,   num_cls, allow_zero_in_degree=True)
    def forward(self, g, x):
        h = torch.relu(self.conv1(g, x))
        return self.conv2(g, h)

# ------------------------------------------------
# 5. Meta-Experience Replay (MER) training
# ------------------------------------------------
memory_buffer    = []        # stores exemplars across tasks
memory_per_task  = 200       # max samples per task
mer_batch_size   = 64        # for both new and memory
inner_lr         = 0.005     # lr for inner update
meta_lr          = 0.01      # lr for meta-update

device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model          = GCN(len(feat_cols), 64, len(le.classes_)).to(device)
opt            = optim.SGD(model.parameters(), lr=meta_lr)
criterion      = nn.CrossEntropyLoss()
graph          = graph.to(device)

results        = {}
param_snapshots = []  # to store θ^(t) after each stage

for stage, (tr_mask, te_mask, grp) in enumerate(
    zip(train_masks, test_masks, valid_groups), 1):
    print(f"\n=== Stage {stage}: learning {grp} ===")
    tr_mask = tr_mask.to(device)
    te_mask = te_mask.to(device)

    new_idxs = tr_mask.nonzero().squeeze().tolist()

    # TRAIN current task with MER
    model.train()
    for epoch in range(1, 51):
        # sample minibatch from new task
        new_batch = random.sample(new_idxs, min(mer_batch_size, len(new_idxs)))
        # sample minibatch from memory
        mem_batch = (random.sample(memory_buffer, mer_batch_size)
                     if len(memory_buffer)>=mer_batch_size else memory_buffer.copy())

        # --- MER meta-update ---
        opt.zero_grad()
        # 1) Compute grad on new batch
        logits_new = model(graph, graph.ndata["feat"])
        loss_new   = criterion(logits_new[new_batch],
                               graph.ndata["label"][new_batch].to(device))
        grads_new  = torch.autograd.grad(loss_new, model.parameters(), create_graph=True)

        # 2) Inner update: θ′ = θ − α ∇L_new
        theta_orig = [p.data.clone() for p in model.parameters()]
        for p, g in zip(model.parameters(), grads_new):
            p.data.sub_(inner_lr * g.data)

        # 3) Compute loss on memory batch with updated params
        if mem_batch:
            logits_mem = model(graph, graph.ndata["feat"])
            loss_mem   = criterion(logits_mem[mem_batch],
                                   graph.ndata["label"][mem_batch].to(device))
            # 4) Meta-grad: d loss_mem / d θ
            grads_meta = torch.autograd.grad(loss_mem, model.parameters())
        else:
            grads_meta = grads_new  # fallback

        # restore original θ
        for p, v in zip(model.parameters(), theta_orig):
            p.data.copy_(v)

        # 5) Apply meta-gradient update
        for p, g_meta in zip(model.parameters(), grads_meta):
            p.grad = g_meta.data.clone()
        opt.step()

        if epoch in (1, 10, 50):
            print(f"  Epoch {epoch:02d}, loss_new: {loss_new.item():.4f}")

    # update memory buffer for this task
    sampled = random.sample(new_idxs, min(memory_per_task, len(new_idxs)))
    memory_buffer.extend(sampled)

    # EVALUATE on all seen tasks
    model.eval()
    with torch.no_grad():
        logits = model(graph, graph.ndata["feat"]).cpu().numpy()
        preds  = logits.argmax(axis=1)
        results[stage] = {}
        for i in range(stage):
            idxs  = test_masks[i].nonzero().squeeze().cpu().numpy()
            acc   = accuracy_score(graph.ndata["label"][idxs].cpu().numpy(), preds[idxs])
            results[stage][f"group_{i+1}"] = acc
            print(f"  → Acc on group {i+1} {valid_groups[i]}: {acc*100:.2f}%")

        cm_idxs = te_mask.nonzero().squeeze().cpu().numpy()
        cm = confusion_matrix(
            graph.ndata["label"][cm_idxs].cpu().numpy(),
            preds[cm_idxs],
            labels=le.transform(grp)
        )
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=grp, yticklabels=grp)
        plt.title(f"Confusion — {grp}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.show()

    # snapshot parameters θ^(stage)
    param_snapshots.append({
        name: tensor.clone().detach().cpu()
        for name, tensor in model.state_dict().items()
    })

# ------------------------------------------------
# 6. Summarize results
# ------------------------------------------------
df_res = pd.DataFrame(results).T
df_res.index.name = "Stage"
print("\n=== Test accuracies per stage & group ===")
print(df_res.applymap(lambda v: f"{v*100:5.2f}%"))

# ------------------------------------------------
# 7. Interpretability Dashboard for MER
# ------------------------------------------------
records = []
num_tasks = len(param_snapshots)

for t in range(2, num_tasks+1):
    state_t = param_snapshots[t-1]
    for k in range(1, t):
        state_k = param_snapshots[k-1]

        # Meta‐drift: Σ_i (θ_i^(t) − θ_i^(k))²
        meta_drift = 0.0
        for name in state_k:
            diff = state_t[name] - state_k[name]
            meta_drift += diff.pow(2).sum().item()

        # Forgetting ΔAcc_k^(t)
        acc_kk = results[k][f"group_{k}"]
        acc_kt = results[t][f"group_{k}"]
        delta_acc = acc_kk - acc_kt

        records.append({
            "from_task":  k,
            "to_task":    t,
            "meta_drift": meta_drift,
            "delta_acc":  delta_acc
        })

mer_df = pd.DataFrame(records)
mer_df["from_to"] = mer_df.apply(lambda r: f"{int(r['from_task'])}→{int(r['to_task'])}", axis=1)

# Print the dashboard
print("\n=== MER Interpretability: Meta-Drift vs Forgetting ===")
print(mer_df[["from_to","meta_drift","delta_acc"]])

# Correlation
corr_md = mer_df["meta_drift"].corr(mer_df["delta_acc"])
print(f"\nCorrelation (meta_drift vs ΔAcc): {corr_md:.4f}")

# ------------------------------------------------
# 9. Additional: Parameter Drift vs Forgetting for MER
# ------------------------------------------------
import numpy as np
import pandas as pd

drift_records = []
# param_snapshots was built in your MER loop
for t in range(2, len(param_snapshots) + 1):
    state_t = param_snapshots[t-1]
    for k in range(1, t):
        state_k = param_snapshots[k-1]

        # compute L2 parameter drift ‖θ^(t) − θ^(k)‖₂
        sq = 0.0
        for name in state_k:
            diff = state_t[name] - state_k[name]
            sq += diff.pow(2).sum().item()
        param_drift = np.sqrt(sq)

        # compute forgetting ΔAcc_k^(t)
        acc_kk = results[k][f"group_{k}"]
        acc_kt = results[t][f"group_{k}"]
        delta_acc = acc_kk - acc_kt

        drift_records.append({
            "from_task":   k,
            "to_task":     t,
            "param_drift": param_drift,
            "delta_acc":   delta_acc
        })

drift_df = pd.DataFrame(drift_records)
drift_df["from_to"] = drift_df.apply(
    lambda r: f"{int(r['from_task'])}→{int(r['to_task'])}", axis=1
)

print("\n=== MER Interpretability: Param Drift vs Forgetting ===")
print(drift_df[["from_to","param_drift","delta_acc"]])

corr_pd = drift_df["param_drift"].corr(drift_df["delta_acc"])
print(f"\nCorrelation (param drift vs ΔAcc): {corr_pd:.4f}")
