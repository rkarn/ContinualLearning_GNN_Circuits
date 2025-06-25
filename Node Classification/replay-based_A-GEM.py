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
# 2. Build DGL graph
# ------------------------------------------------
edges = []
src_cands = df[df["fan_out"]>0]["node"].tolist()
for _, row in df.iterrows():
    tgt = node2idx[row["node"]]
    for s in src_cands[:int(row["fan_in"])]:
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
    ["and","or"],
    ["nand","nor"],
    ["xor","xnor"],
    ["buf","not"]
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
# 5. Helper: flatten & set grads
# ------------------------------------------------
def get_grad_vector(params):
    return torch.cat([p.grad.data.view(-1) for p in params])

def set_grad_vector(params, vec):
    pointer = 0
    for p in params:
        numel = p.numel()
        p.grad.data.copy_(vec[pointer:pointer+numel].view_as(p))
        pointer += numel

# ------------------------------------------------
# 6. Training loop w/ A-GEM + snapshots
# ------------------------------------------------
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model         = GCN(len(feat_cols), 64, len(le.classes_)).to(device)
opt           = optim.Adam(model.parameters(), lr=1e-2)
criterion     = nn.CrossEntropyLoss()
graph         = graph.to(device)

memory_buffer = []      # all exemplars seen so far
memory_snap   = []      # copy of memory_buffer after each stage
param_snapshots = []    # parameter state_dict per stage
results       = {}

memory_per_task = 200
mem_batch_size  = 100

for stage, (tr_mask, te_mask, grp) in enumerate(
        zip(train_masks, test_masks, valid_groups), start=1):
    print(f"\n=== Stage {stage}: learning {grp} ===")
    tr_mask = tr_mask.to(device)
    te_mask = te_mask.to(device)
    new_idxs = tr_mask.nonzero().squeeze().tolist()
    params = [p for p in model.parameters() if p.requires_grad]

    model.train()
    for epoch in range(1, 51):
        # 1) reference gradient on memory
        if memory_buffer:
            mem_idxs = random.sample(memory_buffer,
                                     min(mem_batch_size, len(memory_buffer)))
            opt.zero_grad()
            logits = model(graph, graph.ndata["feat"])
            loss_mem = criterion(logits[mem_idxs], graph.ndata["label"][mem_idxs])
            loss_mem.backward()
            g_ref = get_grad_vector(params).clone()

        # 2) gradient on new task
        opt.zero_grad()
        logits = model(graph, graph.ndata["feat"])
        loss_new = criterion(logits[new_idxs], graph.ndata["label"][new_idxs])
        loss_new.backward()
        g_new = get_grad_vector(params).clone()

        # 3) A-GEM projection
        if memory_buffer:
            dot = torch.dot(g_new, g_ref)
            if dot < 0:
                g_proj = g_new - dot / (g_ref.dot(g_ref) + 1e-12) * g_ref
                set_grad_vector(params, g_proj)

        opt.step()
        if epoch in (1, 10, 50):
            print(f"  Epoch {epoch:02d}, loss_new: {loss_new.item():.4f}")

    # update memory_buffer & snapshot it
    sampled = random.sample(new_idxs, min(memory_per_task, len(new_idxs)))
    memory_buffer.extend(sampled)
    memory_snap.append(memory_buffer.copy())

    # evaluation
    model.eval()
    with torch.no_grad():
        logits_full = model(graph, graph.ndata["feat"]).cpu().numpy()
        preds       = logits_full.argmax(axis=1)
        results[stage] = {}
        for i in range(stage):
            idxs = test_masks[i].nonzero().squeeze().cpu().numpy()
            acc = accuracy_score(graph.ndata["label"][idxs].cpu().numpy(), preds[idxs])
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
        plt.xlabel("Pred"); plt.ylabel("True")
        plt.show()

    # snapshot parameters after this stage
    param_snapshots.append({
        k: v.clone().detach().cpu()
        for k, v in model.state_dict().items()
    })

# ------------------------------------------------
# 7. Summarize results
# ------------------------------------------------
df_res = pd.DataFrame(results).T
df_res.index.name = "Stage"
print("\n=== Test accuracies per stage & group ===")
print(df_res.applymap(lambda v: f"{v*100:5.2f}%"))

# ------------------------------------------------
# 8. Interpretability Dashboard for A-GEM
# ------------------------------------------------
# (a) Cosine similarity vs forgetting
records = []
num_tasks = len(memory_snap)
for t in range(2, num_tasks + 1):
    # new-task gradient at t
    model.zero_grad()
    tr_idxs_t = train_masks[t-1].nonzero().squeeze().tolist()
    logits_t  = model(graph, graph.ndata["feat"])
    loss_new_t = criterion(logits_t[tr_idxs_t], graph.ndata["label"][tr_idxs_t])
    loss_new_t.backward()
    g_new_t = get_grad_vector([p for p in model.parameters() if p.requires_grad]).cpu()

    for k in range(1, t):
        # reference gradient for k
        model.zero_grad()
        mem_k = memory_snap[k-1]
        logits_k = model(graph, graph.ndata["feat"])
        loss_mem_k = criterion(logits_k[mem_k], graph.ndata["label"][mem_k])
        loss_mem_k.backward()
        g_ref_k = get_grad_vector([p for p in model.parameters() if p.requires_grad]).cpu()

        cos_sim = torch.dot(g_new_t, g_ref_k) / (
            g_new_t.norm() * g_ref_k.norm() + 1e-12
        )
        delta_acc = results[k][f"group_{k}"] - results[t][f"group_{k}"]
        records.append({
            "from_task": k, "to_task": t,
            "cos_sim": cos_sim.item(),
            "delta_acc": delta_acc
        })

agem_df = pd.DataFrame(records)
agem_df["from_to"] = agem_df.apply(lambda r: f"{int(r['from_task'])}→{int(r['to_task'])}", axis=1)
print("\n=== A-GEM Interpretability: Cosine Sim vs Forgetting ===")
print(agem_df[["from_to","cos_sim","delta_acc"]])
print(f"\nCorrelation (cosine similarity vs ΔAcc): {agem_df['cos_sim'].corr(agem_df['delta_acc']):.4f}")

# (b) Parameter drift vs forgetting
drift_records = []
for t in range(2, len(param_snapshots)+1):
    state_t = param_snapshots[t-1]
    for k in range(1, t):
        state_k = param_snapshots[k-1]
        sqnorm = 0.0
        for name in state_k:
            diff = state_t[name] - state_k[name]
            sqnorm += diff.pow(2).sum().item()
        param_drift = np.sqrt(sqnorm)
        delta_acc = results[k][f"group_{k}"] - results[t][f"group_{k}"]
        drift_records.append({
            "from_task": k, "to_task": t,
            "param_drift": param_drift,
            "delta_acc": delta_acc
        })

drift_df = pd.DataFrame(drift_records)
drift_df["from_to"] = drift_df.apply(lambda r: f"{int(r['from_task'])}→{int(r['to_task'])}", axis=1)
print("\n=== A-GEM Interpretability: Param Drift vs Forgetting ===")
print(drift_df[["from_to","param_drift","delta_acc"]])
print(f"\nCorrelation (param drift vs ΔAcc): {drift_df['param_drift'].corr(drift_df['delta_acc']):.4f}")
