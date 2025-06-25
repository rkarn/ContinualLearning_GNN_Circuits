import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import dgl.nn as dglnn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# 1. Load & preprocess CSV
# -------------------------
df = pd.read_csv("all_circuits_features_graphclass.csv")
feat_cols = [
    "fan_in","fan_out","dist_to_output","is_primary_input",
    "is_primary_output","is_internal","is_key_gate",
    "degree_centrality","betweenness_centrality",
    "closeness_centrality","clustering_coefficient",
    "avg_fan_in_neighbors","avg_fan_out_neighbors"
]
df[feat_cols] = StandardScaler().fit_transform(df[feat_cols])

# -------------------------
# 2. Build one graph per circuit
# -------------------------
graphs, sizes = [], []
for cid, sub in df.groupby("circuit_id"):
    nodes = sub["node"].tolist()
    idx   = {n:i for i,n in enumerate(nodes)}
    edges = set()
    for _,r in sub.iterrows():
        u = idx[r["node"]]
        k = int(r["fan_in"])
        for p in nodes[:k]:
            edges.add((idx[p], u))
    if not edges: continue
    src, dst = zip(*edges)
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=len(nodes))
    g = dgl.add_self_loop(g)
    g.ndata["feat"] = torch.tensor(sub[feat_cols].values, dtype=torch.float32)
    graphs.append(g)
    sizes.append(len(nodes))

graphs = np.array(graphs)
sizes  = np.array(sizes)

# -------------------------
# 3. Define 4 quartile tasks
# -------------------------
qs = np.percentile(sizes, [25,50,75])
quarts = np.digitize(sizes, qs)  # labels 0–3

tasks = []
for q in range(4):
    pos = np.where(quarts==q)[0]
    neg = np.where(quarts!=q)[0]
    p_tr, p_te = train_test_split(pos, test_size=0.2, random_state=42)
    n_tr, n_te = train_test_split(neg, test_size=0.2, random_state=42)
    tasks.append({
        "q": q,
        "train_pos": p_tr.tolist(), "train_neg": n_tr.tolist(),
        "test_pos":  p_te.tolist(), "test_neg":  n_te.tolist()
    })

# -------------------------
# 4. GCN + readout
# -------------------------
class GCNEnc(nn.Module):
    def __init__(self, in_feats, hid):
        super().__init__()
        self.c1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.c2 = dglnn.GraphConv(hid,   hid,   allow_zero_in_degree=True)
        self.read = nn.Linear(hid, 2)
    def forward(self, g, x):
        h = torch.relu(self.c1(g, x))
        h = torch.relu(self.c2(g, h))
        hg= h.mean(dim=0, keepdim=True)
        return self.read(hg)

# -------------------------
# 5. A-GEM helpers
# -------------------------
def get_grad_vector(params):
    return torch.cat([p.grad.view(-1) for p in params])

def set_grad_vector(params, vec):
    pointer = 0
    for p in params:
        numel = p.numel()
        p.grad.copy_(vec[pointer:pointer+numel].view_as(p))
        pointer += numel

# -------------------------
# 6. Train w/ A-GEM
# -------------------------
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model         = GCNEnc(len(feat_cols), hid=64).to(device)
opt           = optim.Adam(model.parameters(), lr=1e-3)
criterion     = nn.CrossEntropyLoss()

memory_buffer = []    # stored graph indices
memory_snaps  = []
param_snaps   = []
results       = {}

mem_batch_size  = 64
memory_per_task = 200

for stage, t in enumerate(tasks, start=1):
    q      = t["q"]
    p_tr   = t["train_pos"]; n_tr = t["train_neg"]
    p_te   = t["test_pos"];  n_te = t["test_neg"]
    new_idxs = p_tr + n_tr
    params   = [p for p in model.parameters() if p.requires_grad]

    print(f"\n=== Stage {stage}: quartile {q} vs rest ===")
    model.train()
    for epoch in (1,10,50):
        # 1) ref grad on memory
        if memory_buffer:
            mem_idxs = random.sample(memory_buffer,
                                     min(mem_batch_size, len(memory_buffer)))
            opt.zero_grad()
            outs_mem = []
            for i in mem_idxs:
                g = graphs[i].to(device)
                outs_mem.append(model(g, g.ndata["feat"].to(device))[0])
            outs_mem = torch.stack(outs_mem)
            # labels for mem examples
            y_mem = []
            for i in mem_idxs:
                for j in range(stage-1):
                    if i in tasks[j]["train_pos"]:
                        y_mem.append(1); break
                    if i in tasks[j]["train_neg"]:
                        y_mem.append(0); break
            y_mem = torch.tensor(y_mem, dtype=torch.long, device=device)
            loss_mem = criterion(outs_mem, y_mem)
            loss_mem.backward()
            g_ref = get_grad_vector(params).clone()

        # 2) grad on new task
        opt.zero_grad()
        outs_new = []
        for i in new_idxs:
            g = graphs[i].to(device)
            outs_new.append(model(g, g.ndata["feat"].to(device))[0])
        outs_new = torch.stack(outs_new)
        y_new = torch.tensor([1]*len(p_tr)+[0]*len(n_tr),
                              dtype=torch.long, device=device)
        loss_new = criterion(outs_new, y_new)
        loss_new.backward()
        g_new = get_grad_vector(params).clone()

        # 3) A-GEM projection
        if memory_buffer:
            dot = torch.dot(g_new, g_ref)
            if dot < 0:
                g_proj = g_new - dot / (g_ref.dot(g_ref)+1e-12) * g_ref
                set_grad_vector(params, g_proj)

        opt.step()
        print(f"  Epoch {epoch:02d}, loss_new: {loss_new.item():.4f}")

    # update memory_buffer
    sampled = random.sample(new_idxs, min(memory_per_task, len(new_idxs)))
    memory_buffer.extend(sampled)
    memory_snaps.append(list(memory_buffer))

    # snapshot params
    param_snaps.append({n:p.clone().detach().cpu()
                        for n,p in model.state_dict().items()})

    # evaluation on all seen tasks
    model.eval()
    with torch.no_grad():
        all_logits = []
        for g in graphs:
            g = g.to(device)
            all_logits.append(model(g, g.ndata["feat"].to(device))[0].cpu().numpy())
        all_preds = np.vstack(all_logits).argmax(axis=1)

    results[stage] = {}
    for k in range(stage):
        tp = tasks[k]["test_pos"]; tn = tasks[k]["test_neg"]
        idxs = tp + tn
        y_true = np.array([1]*len(tp)+[0]*len(tn))
        y_pred = all_preds[idxs]
        acc = accuracy_score(y_true, y_pred)
        results[stage][f"task_{k+1}"] = acc
        print(f"  → Acc task{k+1}: {acc*100:.2f}%")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["rest",f"q{tasks[k]['q']}"],
                    yticklabels=["rest",f"q{tasks[k]['q']}"])
        plt.title(f"Confusion stage{stage}/task{k+1}")
        plt.show()

# -------------------------
# 7. Summary of accuracies
# -------------------------
df_res = pd.DataFrame(results).T
df_res.index.name = "Stage"
print("\n=== Accuracies per stage & task ===")
print(df_res.applymap(lambda x: f"{x*100:.2f}%"))

# -------------------------
# 8. Interpretability Dashboard
# -------------------------
records = []
for t in range(2, len(param_snaps)+1):
    θ_t = param_snaps[t-1]
    for k in range(1, t):
        θ_k = param_snaps[k-1]
        # param drift
        pd2 = sum((θ_t[n]-θ_k[n]).pow(2).sum().item() for n in θ_k)
        pd  = np.sqrt(pd2)
        # cosine sim of gradients
        # compute g_new_t
        model.zero_grad()
        outs = []
        for i in tasks[t-1]["train_pos"]+tasks[t-1]["train_neg"]:
            g = graphs[i].to(device)
            outs.append(model(g, g.ndata["feat"].to(device))[0])
        outs = torch.stack(outs)
        lbl = torch.tensor([1]*len(tasks[t-1]["train_pos"])+[0]*len(tasks[t-1]["train_neg"]),
                           device=device)
        loss_t = criterion(outs, lbl); loss_t.backward()
        g_new_t = get_grad_vector([p for p in model.parameters() if p.requires_grad]).cpu()
        # reference grad for task k
        model.zero_grad()
        mem_k = memory_snaps[k-1]
        outs_k = []
        for i in mem_k:
            g = graphs[i].to(device)
            outs_k.append(model(g, g.ndata["feat"].to(device))[0])
        if outs_k:
            outs_k = torch.stack(outs_k)
            lbl_k  = []
            for i in mem_k:
                if i in tasks[k-1]["train_pos"]: lbl_k.append(1)
                else: lbl_k.append(0)
            lbl_k = torch.tensor(lbl_k, device=device)
            loss_k = criterion(outs_k, lbl_k); loss_k.backward()
            g_ref_k = get_grad_vector([p for p in model.parameters() if p.requires_grad]).cpu()
            cos_sim = torch.dot(g_new_t, g_ref_k)/(g_new_t.norm()*g_ref_k.norm()+1e-12)
            cos_sim = cos_sim.item()
        else:
            cos_sim = np.nan

        # forgetting
        da = results[k][f"task_{k}"] - results[t][f"task_{k}"]

        records.append({
            "from_task":   k,
            "to_task":     t,
            "param_drift": pd,
            "cos_sim":     cos_sim,
            "delta_acc":   da
        })

dash = pd.DataFrame(records)
dash["from_to"] = dash["from_task"].astype(str)+"→"+dash["to_task"].astype(str)
print("\n=== A-GEM Dashboard: Drift vs Forgetting ===")
print(dash[["from_to","param_drift","cos_sim","delta_acc"]])
print(f"\nCorr(param_drift vs ΔAcc): {dash.param_drift.corr(dash.delta_acc):.4f}")
print(f"Corr(cosine_sim vs ΔAcc):  {dash.cos_sim.corr(dash.delta_acc):.4f}")

# -----------------------------
# Interpretability Dashboard (fixed)
# -----------------------------
import pandas as _pd
import numpy as _np

# Sanity check
if not isinstance(records, list):
    raise RuntimeError(f"`records` must be a list, got {type(records)}")

# Build DataFrame
dash = _pd.DataFrame.from_records(records)

if dash.empty:
    print("Not enough data for A-GEM dashboard.")
else:
    # Create the "from→to" label
    dash["from_to"] = (
        dash["from_task"].astype(int).astype(str)
        + "→"
        + dash["to_task"].astype(int).astype(str)
    )

    # Print the dashboard
    print("\n=== A-GEM Dashboard: Drift vs Forgetting ===")
    print(dash[["from_to","param_drift","cos_sim","delta_acc"]])

    # Correlations
    corr_pd  = dash["param_drift"].corr(dash["delta_acc"])
    corr_cs  = dash["cos_sim"].corr(dash["delta_acc"])
    print(f"\nCorr(param_drift vs ΔAcc): {corr_pd:.4f}")
    print(f"Corr(cos_sim vs ΔAcc):       {corr_cs:.4f}")
