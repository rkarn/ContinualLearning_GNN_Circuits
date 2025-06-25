import random
import copy
import numpy as np
import pandas as pd
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

# -----------------------------
# 1. Load & preprocess data
# -----------------------------
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

nodes   = df["node"].tolist()
node2idx = {n:i for i,n in enumerate(nodes)}
N = len(nodes)

# -----------------------------
# 2. Build DGL graph
# -----------------------------
edges = set()
src_cands = df[df["fan_out"]>0]["node"].tolist()
for _, r in df.iterrows():
    tgt = node2idx[r["node"]]
    k   = int(r["fan_in"])
    for s in src_cands[:k]:
        edges.add((node2idx[s], tgt))
src, dst = zip(*edges) if edges else ([],[])
g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=N)
g = dgl.add_self_loop(g)
g.ndata["feat"]  = torch.tensor(df[feat_cols].values, dtype=torch.float32)
g.ndata["label"] = torch.tensor(df["gate_label"].values, dtype=torch.long)

# -----------------------------
# 3. Define 4 classification tasks
# -----------------------------
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
    tm = torch.zeros(N, dtype=torch.bool); tm[tr] = True
    vm = torch.zeros(N, dtype=torch.bool); vm[te] = True
    train_masks.append(tm); test_masks.append(vm)

# -----------------------------
# 4. GCN for node classification
# -----------------------------
class GCN(nn.Module):
    def __init__(self, in_feats, hid, num_cls):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid,   num_cls, allow_zero_in_degree=True)
    def forward(self, graph, x):
        h = F.relu(self.conv1(graph, x))
        return self.conv2(graph, h)

# -----------------------------
# 5. A-GEM helpers
# -----------------------------
def get_grad_vector(params):
    return torch.cat([p.grad.view(-1) for p in params])

def set_grad_vector(params, vec):
    pointer = 0
    for p in params:
        numel = p.numel()
        p.grad.copy_(vec[pointer:pointer+numel].view_as(p))
        pointer += numel

# -----------------------------
# 6. Train w/ A-GEM + snapshot
# -----------------------------
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model          = GCN(len(feat_cols), 64, len(le.classes_)).to(device)
opt            = optim.Adam(model.parameters(), lr=1e-2)
criterion      = nn.CrossEntropyLoss()
graph          = g.to(device)

memory_buffer  = []         # list of node indices
memory_snap    = []
param_snapshots= []
results        = {}

memory_per_task= 200
mem_batch_size = 100

for stage, (tr_mask, te_mask, grp) in enumerate(
        zip(train_masks, test_masks, valid_groups), start=1):
    print(f"\n=== Stage {stage}: learning {grp} ===")
    tr = tr_mask.to(device); te = te_mask.to(device)
    new_idxs = tr.nonzero().squeeze().tolist()
    params = [p for p in model.parameters() if p.requires_grad]

    model.train()
    for epoch in range(1, 51):
        # 1) reference grad on memory
        if memory_buffer:
            mem = random.sample(memory_buffer, min(mem_batch_size, len(memory_buffer)))
            opt.zero_grad()
            logits = model(graph, graph.ndata["feat"])
            loss_mem = criterion(logits[mem], graph.ndata["label"][mem])
            loss_mem.backward()
            g_ref = get_grad_vector(params).clone()

        # 2) compute new-task grad
        opt.zero_grad()
        logits = model(graph, graph.ndata["feat"])
        loss_new = criterion(logits[new_idxs], graph.ndata["label"][new_idxs])
        loss_new.backward()
        g_new = get_grad_vector(params).clone()

        # 3) project if conflict
        if memory_buffer:
            dot = torch.dot(g_new, g_ref)
            if dot < 0:
                g_proj = g_new - dot / (g_ref.dot(g_ref)+1e-12) * g_ref
                set_grad_vector(params, g_proj)

        opt.step()
        if epoch in (1,10,50):
            print(f"  Epoch {epoch:02d}, loss_new: {loss_new.item():.4f}")

    # update memory & snapshots
    sampled = random.sample(new_idxs, min(memory_per_task, len(new_idxs)))
    memory_buffer.extend(sampled)
    memory_snap.append(list(memory_buffer))
    param_snapshots.append({n: p.clone().detach().cpu()
                             for n,p in model.state_dict().items()})

    # evaluate
    model.eval()
    with torch.no_grad():
        logits_full = model(graph, graph.ndata["feat"]).cpu().numpy()
        preds = logits_full.argmax(1)
    results[stage] = {}
    for k in range(stage):
        idxs = test_masks[k].nonzero().squeeze().cpu().numpy()
        y_true = g.ndata["label"][idxs].cpu().numpy()
        y_pred = preds[idxs]
        acc = accuracy_score(y_true, y_pred)
        results[stage][f"group_{k+1}"] = acc
        print(f"  → Acc group {k+1} {valid_groups[k]}: {acc*100:.2f}%")

    cm_idxs = te.nonzero().squeeze().cpu().numpy()
    cm = confusion_matrix(
        g.ndata["label"][cm_idxs].cpu().numpy(), preds[cm_idxs]
    )
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=grp, yticklabels=grp)
    plt.title(f"Confusion — {grp}")
    plt.show()

# -----------------------------
# 7. Summarize results
# -----------------------------
df_res = pd.DataFrame(results).T
df_res.index.name = "Stage"
print("\n=== Accuracies per stage & group ===")
print(df_res.applymap(lambda v: f"{v*100:.2f}%"))

# -----------------------------
# 8. Interpretability Dashboard
# -----------------------------
records = []
num_tasks = len(param_snapshots)

for t in range(2, num_tasks+1):
    # compute new-task gradient g_new_t
    model.zero_grad()
    tr_idxs = train_masks[t-1].nonzero().squeeze().tolist()
    logits = model(graph, graph.ndata["feat"])
    loss_t = criterion(logits[tr_idxs], graph.ndata["label"][tr_idxs])
    loss_t.backward()
    g_new_t = get_grad_vector(params).cpu()

    for k in range(1, t):
        # compute ref gradient g_ref_k on memory_snap[k-1]
        model.zero_grad()
        mem_k = memory_snap[k-1]
        logits_k = model(graph, graph.ndata["feat"])
        loss_k = criterion(logits_k[mem_k], graph.ndata["label"][mem_k])
        loss_k.backward()
        g_ref_k = get_grad_vector(params).cpu()

        # cosine similarity
        cos_sim = torch.dot(g_new_t, g_ref_k)/(g_new_t.norm()*g_ref_k.norm()+1e-12)

        # parameter drift
        θ_t = param_snapshots[t-1]; θ_k = param_snapshots[k-1]
        sq = sum((θ_t[n]-θ_k[n]).pow(2).sum().item() for n in θ_k)
        pd = np.sqrt(sq)

        # forgetting ΔAcc_k^(t)
        da = results[k][f"group_{k}"] - results[t][f"group_{k}"]

        records.append({
            "from_task": k, "to_task": t,
            "cos_sim": cos_sim.item(),
            "param_drift": pd,
            "delta_acc": da
        })

dash = pd.DataFrame(records)
dash["from_to"] = dash["from_task"].astype(str)+"→"+dash["to_task"].astype(str)

print("\n=== A-GEM Interpretability ===")
print(dash[["from_to","cos_sim","param_drift","delta_acc"]])
print(f"\nCorr(cos_sim vs ΔAcc):    {dash.cos_sim.corr(dash.delta_acc):.4f}")
print(f"Corr(param_drift vs ΔAcc): {dash.param_drift.corr(dash.delta_acc):.4f}")


# -----------------------------
# 8. Interpretability Dashboard
# -----------------------------
import pandas as _pd    # re-import pandas under a fresh alias
import numpy as _np

# `records` should be a list of dicts:
#   [ {"from_task": k, "to_task": t, "cos_sim": ..., "param_drift": ..., "delta_acc": ...}, ... ]

if not isinstance(records, list):
    raise RuntimeError("`records` must be a list of dicts, got: %r" % type(records))

dashboard = _pd.DataFrame.from_records(records)

if dashboard.empty:
    print("Not enough tasks to build A-GEM dashboard.")
else:
    # Create the “k→t” label
    dashboard["from_to"] = (
        dashboard["from_task"].astype(int).astype(str)
        + "→"
        + dashboard["to_task"].astype(int).astype(str)
    )

    # Print the table
    print("\n=== A-GEM Interpretability: CosineSim / ParamDrift vs Forgetting ===")
    print(dashboard[["from_to","cos_sim","param_drift","delta_acc"]])

    # Compute correlations
    corr_cs = dashboard["cos_sim"].corr(     dashboard["delta_acc"] )
    corr_pd = dashboard["param_drift"].corr(dashboard["delta_acc"])
    print(f"\nCorr(cosine-sim vs ΔAcc):    {corr_cs:.4f}")
    print(f"Corr(param-drift vs ΔAcc):   {corr_pd:.4f}")
