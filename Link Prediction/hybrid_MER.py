import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. Load & preprocess
# ---------------------------------------
df = pd.read_csv("all_circuits_features.csv")
feat_cols = [
    "fan_in","fan_out","dist_to_output","is_primary_input",
    "is_primary_output","is_internal","is_key_gate",
    "degree_centrality","betweenness_centrality",
    "closeness_centrality","clustering_coefficient",
    "avg_fan_in_neighbors","avg_fan_out_neighbors"
]
df = df.dropna(subset=feat_cols)
df[feat_cols] = StandardScaler().fit_transform(df[feat_cols].astype(float))

nodes   = df["node"].tolist()
nid2idx = {n:i for i,n in enumerate(nodes)}
N       = len(nodes)

# build base graph
edges = set()
src_cands = df[df["fan_out"]>0]["node"].tolist()
for _,r in df.iterrows():
    u = nid2idx[r["node"]]
    k = int(r["fan_in"])
    for vname in src_cands[:k]:
        v = nid2idx[vname]
        edges.add((u,v))
edges = np.array(list(edges))

g = dgl.graph((torch.tensor(edges[:,0]), torch.tensor(edges[:,1])), num_nodes=N)
g = dgl.add_self_loop(g)
g.ndata["feat"] = torch.tensor(df[feat_cols].values, dtype=torch.float32)

# ---------------------------------------
# 2. Link‐pred tasks by gate‐type
# ---------------------------------------
group_defs = [
    ["and","or"], ["nand","nor"],
    ["xor","xnor"], ["buf","not"]
]
tasks = []
for grp in group_defs:
    idxs = [nid2idx[n] for n in df[df["gate_type"].isin(grp)]["node"]]
    pos = edges[np.isin(edges[:,0], idxs) & np.isin(edges[:,1], idxs)]
    if len(pos)<10:  # if too few, sample random
        pos = edges[np.random.choice(len(edges), 20, replace=False)]
    n = len(pos)
    perm = np.random.permutation(n)
    split = int(0.8*n)
    pos_tr, pos_te = pos[perm[:split]], pos[perm[split:]]
    neg = []
    pos_set = set(map(tuple, pos.tolist()))
    while len(neg)<n:
        u,v = random.choice(idxs), random.choice(idxs)
        if u!=v and (u,v) not in pos_set:
            neg.append((u,v))
    neg = np.array(neg)
    neg_tr, neg_te = neg[perm[:split]], neg[perm[split:]]
    tasks.append({
        "name": grp,
        "train":  np.vstack([np.column_stack([pos_tr, np.ones(len(pos_tr))]),
                             np.column_stack([neg_tr, np.zeros(len(neg_tr))])]),
        "test":   np.vstack([np.column_stack([pos_te, np.ones(len(pos_te))]),
                             np.column_stack([neg_te, np.zeros(len(neg_te))])])
    })

# ---------------------------------------
# 3. GCN encoder
# ---------------------------------------
class GCNEnc(nn.Module):
    def __init__(self, in_feats, hid):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid, hid,   allow_zero_in_degree=True)
    def forward(self, g, x):
        h = torch.relu(self.conv1(g, x))
        return self.conv2(g, h)

# ---------------------------------------
# 4. MER hyperparams
# ---------------------------------------
memory_buffer   = []   # list of [u,v,label]
memory_per_task = 200
mer_batch_size  = 64
inner_lr        = 0.005
meta_lr         = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GCNEnc(len(feat_cols), 64).to(device)
opt    = optim.SGD(model.parameters(), lr=meta_lr)
criterion = nn.BCEWithLogitsLoss()

results         = {}
param_snaps     = []
memory_snaps    = []

# ---------------------------------------
# 5. Training w/ MER
# ---------------------------------------
for stage, task in enumerate(tasks, start=1):
    print(f"\n=== Stage {stage}: {task['name']} ===")
    train_data = task["train"].copy()
    # combine with memory
    if memory_buffer:
        train_data = np.vstack([train_data, memory_buffer])
    model.train()
    for epoch in range(1, 51):
        # sample new & memory batches
        idxs = np.random.permutation(len(task["train"]))
        new_idx = idxs[:mer_batch_size]
        new_batch = task["train"][new_idx]
        if len(memory_buffer)>=mer_batch_size:
            mem_batch = np.array(random.sample(memory_buffer, mer_batch_size))
        else:
            mem_batch = np.array(memory_buffer) if memory_buffer else np.empty((0,3))

        # 1) grad on new batch
        opt.zero_grad()
        h = model(g, g.ndata["feat"].to(device))
        if len(new_batch)>0:
            ub, vb, yb = new_batch.T
            logits = (h[ub.astype(int)] * h[vb.astype(int)]).sum(dim=1)
            loss_new = criterion(logits, torch.tensor(yb, device=device))
        else:
            loss_new = 0*sum(p.sum() for p in model.parameters())
        grads_new = torch.autograd.grad(loss_new, model.parameters(), create_graph=True)

        # 2) inner update
        backup = [p.data.clone() for p in model.parameters()]
        for p, gnew in zip(model.parameters(), grads_new):
            p.data.sub_(inner_lr * gnew)

        # 3) grad on mem batch
        if len(mem_batch)>0:
            ub, vb, yb = mem_batch.T
            logits_mem = (h[ub.astype(int)] * h[vb.astype(int)]).sum(dim=1)
            loss_mem   = criterion(logits_mem, torch.tensor(yb, device=device))
            grads_meta = torch.autograd.grad(loss_mem, model.parameters())
        else:
            grads_meta = grads_new

        # restore
        for p, b in zip(model.parameters(), backup):
            p.data.copy_(b)

        # 4) meta update
        for p, gmeta in zip(model.parameters(), grads_meta):
            p.grad = gmeta
        opt.step()

        if epoch in (1,10,50):
            print(f"  Epoch {epoch:02d}, loss_new: {loss_new.item():.4f}")

    # update memory
    buf = task["train"].tolist()
    random.shuffle(buf)
    memory_buffer.extend(buf[:memory_per_task])
    memory_snaps.append(list(memory_buffer))

    # snapshot params
    param_snaps.append({n: p.clone().detach().cpu()
                        for n,p in model.state_dict().items()})

    # evaluation
    model.eval()
    with torch.no_grad():
        h = model(g, g.ndata["feat"].to(device))
    res = {}
    for k in range(stage):
        test = tasks[k]["test"]
        ub, vb, yb = test.T
        logits = (h[ub.astype(int)] * h[vb.astype(int)]).sum(dim=1)
        pred = (torch.sigmoid(logits)>0.5).cpu().numpy().astype(int)
        acc = accuracy_score(yb, pred)
        res[f"task_{k+1}"] = acc
        print(f"  → Acc task {k+1} after stage {stage}: {acc*100:.2f}%")
        cm = confusion_matrix(yb, pred)
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion: stage{stage}/task{k+1}")
        plt.show()
    results[stage] = res

# ---------------------------------------
# 6. Summary of accuracies
# ---------------------------------------
df_res = pd.DataFrame(results).T
df_res.index.name="Stage"
print("\n=== Link‐Pred Accuracies ===")
print(df_res.applymap(lambda x: f"{x*100:.2f}%"))

# ---------------------------------------
# 7. Interpretability Dashboard
# ---------------------------------------
records = []
for t in range(2, len(param_snaps)+1):
    θ_t = param_snaps[t-1]
    for k in range(1, t):
        θ_k = param_snaps[k-1]
        # meta‐drift: sum_i (θ_t - θ_k)^2
        sq = sum((θ_t[n]-θ_k[n]).pow(2).sum().item() for n in θ_k)
        meta_drift = sq
        # forgetting ΔAcc
        da = results[k][f"task_{k}"] - results[t][f"task_{k}"]
        records.append({
            "from_task": k, "to_task": t,
            "meta_drift": meta_drift, "delta_acc": da
        })

import pandas as _pd
dash = _pd.DataFrame(records)
if not dash.empty:
    dash["from_to"] = (
        dash["from_task"].astype(int).astype(str)
        + "→" +
        dash["to_task"].astype(int).astype(str)
    )
    print("\n=== MER Dashboard: Meta-Drift vs Forgetting ===")
    print(dash[["from_to","meta_drift","delta_acc"]])
    print(f"Corr(meta_drift vs ΔAcc): {dash.meta_drift.corr(dash.delta_acc):.4f}")
else:
    print("Not enough data for MER interpretability.")
