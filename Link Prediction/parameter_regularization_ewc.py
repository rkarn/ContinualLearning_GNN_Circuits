import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import dgl.nn as dglnn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# 1. Load & preprocess
# ------------------------------------------------
df = pd.read_csv("all_circuits_features.csv")
feat_cols = [
    "fan_in","fan_out","dist_to_output",
    "is_primary_input","is_primary_output","is_internal","is_key_gate",
    "degree_centrality","betweenness_centrality",
    "closeness_centrality","clustering_coefficient",
    "avg_fan_in_neighbors","avg_fan_out_neighbors"
]
df = df.dropna(subset=feat_cols)
df[feat_cols] = StandardScaler().fit_transform(df[feat_cols].astype(float))

# build DGL graph
nodes   = df["node"].tolist()
nid2idx = {n:i for i,n in enumerate(nodes)}
N       = len(nodes)

edge_set = set()
src_cands = df[df["fan_out"]>0]["node"].tolist()
for _, r in df.iterrows():
    u = nid2idx[r["node"]]
    k = int(r["fan_in"])
    for vname in src_cands[:k]:
        v = nid2idx[vname]
        edge_set.add((u,v))
src_arr = np.array([u for u,_ in edge_set])
dst_arr = np.array([v for _,v in edge_set])

g = dgl.graph((torch.tensor(src_arr), torch.tensor(dst_arr)), num_nodes=N)
g = dgl.add_self_loop(g)
g.ndata["feat"] = torch.tensor(df[feat_cols].values, dtype=torch.float32)

# ------------------------------------------------
# 2. Make 4 *random* link‐prediction tasks
# ------------------------------------------------
all_pos = np.column_stack([src_arr, dst_arr])
np.random.shuffle(all_pos)
chunks = np.array_split(all_pos, 4)  # exactly 4 tasks

tasks = []
for i, chunk in enumerate(chunks, start=1):
    n = len(chunk)
    if n < 10:
        # too few edges, skip or combine—but here we just warn
        print(f"Warning: Task {i} has only {n} edges.")
    # 80/20 split
    perm  = np.random.permutation(n)
    split = int(0.8*n)
    pos_tr, pos_te = chunk[perm[:split]], chunk[perm[split:]]
    # sample negatives of equal size
    neg = []
    pos_set = set(map(tuple, chunk.tolist()))
    while len(neg) < n:
        u = np.random.randint(0, N)
        v = np.random.randint(0, N)
        if u!=v and (u,v) not in pos_set:
            neg.append((u,v))
    neg = np.array(neg)
    neg_tr, neg_te = neg[perm[:split]], neg[perm[split:]]
    tasks.append({
        "name":       f"Task{i}",
        "train_pos":  pos_tr,  "train_neg": neg_tr,
        "test_pos":   pos_te,  "test_neg":  neg_te
    })

# ------------------------------------------------
# 3. GCN encoder + EWC helper
# ------------------------------------------------
class GCNEnc(nn.Module):
    def __init__(self, in_feats, hid):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid, hid,   allow_zero_in_degree=True)
    def forward(self, g, x):
        h = torch.relu(self.conv1(g, x))
        return self.conv2(g, h)

class EWC:
    def __init__(self, model, g, task, device):
        self.device = device
        # snapshot
        self.params = {n: p.clone().detach().to(device)
                       for n,p in model.named_parameters()}
        # fisher
        self.fisher = {n: torch.zeros_like(p,device=device)
                       for n,p in model.named_parameters()}
        self._compute_fisher(model, g, task)

    def _compute_fisher(self, model, g, task):
        model.eval()
        P = torch.tensor(task["train_pos"], dtype=torch.long, device=self.device)
        N = torch.tensor(task["train_neg"], dtype=torch.long, device=self.device)
        pairs  = torch.cat([P,N], dim=0)
        labels = torch.cat([
            torch.ones(len(P), device=self.device),
            torch.zeros(len(N),device=self.device)
        ])
        h      = model(g, g.ndata["feat"].to(self.device))
        logits = (h[pairs[:,0]] * h[pairs[:,1]]).sum(dim=1)
        prob   = torch.sigmoid(logits)
        loss   = F.binary_cross_entropy(prob, labels)
        model.zero_grad(); loss.backward()
        for n,p in model.named_parameters():
            if p.grad is not None:
                self.fisher[n] = p.grad.detach()**2

    def penalty(self, model):
        loss = 0
        for n,p in model.named_parameters():
            loss += (self.fisher[n]*(p-self.params[n])**2).sum()
        return loss

# ------------------------------------------------
# 4. Train + evaluate under EWC
# ------------------------------------------------
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc         = GCNEnc(len(feat_cols), 64).to(device)
opt         = optim.Adam(enc.parameters(), lr=1e-2)
ewc_list    = []
λ           = 1000.0
results     = {}
param_snaps = []
fishers     = []

for t, task in enumerate(tasks, start=1):
    print(f"\n=== Stage {t}: {task['name']} ===")

    # build full‐batch training set
    Ptr = torch.tensor(task["train_pos"], dtype=torch.long, device=device)
    Ntr = torch.tensor(task["train_neg"], dtype=torch.long, device=device)
    pairs  = torch.cat([Ptr, Ntr], dim=0)
    labels = torch.cat([
        torch.ones(len(Ptr)), torch.zeros(len(Ntr))
    ], dim=0).to(device)

    # train 50 epochs (we report at 1/25/50)
    for epoch in (1, 25, 50):
        enc.train()
        perm = torch.randperm(len(labels), device=device)
        ups  = pairs[perm,0]; vps = pairs[perm,1]
        h    = enc(g, g.ndata["feat"].to(device))
        logit = (h[ups] * h[vps]).sum(dim=1)
        loss  = F.binary_cross_entropy_with_logits(logit, labels[perm])
        if ewc_list:
            penalty = sum(e.penalty(enc) for e in ewc_list)
            loss = loss + (λ/2)*penalty
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"  Epoch {epoch:2d}, loss: {loss.item():.4f}")

    # snapshot for EWC
    fishers.append(EWC(enc, g, task, device))
    param_snaps.append({n: p.clone().detach().cpu()
                        for n,p in enc.named_parameters()})

    # evaluate on all seen tasks
    enc.eval()
    with torch.no_grad():
        h = enc(g, g.ndata["feat"].to(device))
    accs = {}
    for k, prev in enumerate(tasks[:t], start=1):
        Pte = prev["test_pos"]; Nte = prev["test_neg"]
        Pte = torch.tensor(Pte, dtype=torch.long, device=device)
        Nte = torch.tensor(Nte, dtype=torch.long, device=device)
        ups1 = Pte[:,0]; vps1 = Pte[:,1]
        ups2 = Nte[:,0]; vps2 = Nte[:,1]
        prob_pos = torch.sigmoid((h[ups1]*h[vps1]).sum(dim=1)).cpu().numpy()
        prob_neg = torch.sigmoid((h[ups2]*h[vps2]).sum(dim=1)).cpu().numpy()
        preds    = np.concatenate([prob_pos>0.5, prob_neg>0.5]).astype(int)
        labs     = np.concatenate([np.ones(len(ups1)), np.zeros(len(ups2))])
        accs[f"task_{k}"] = accuracy_score(labs, preds)
        print(f"  → Acc task {k} after stage {t}: {accs[f'task_{k}']*100:.2f}%")
    results[t] = accs

    # confusion matrix for this stage
    labs_te = np.concatenate([np.ones(len(Nte))*0,
                              np.ones(len(ups1))])  # careful ordering
    labs_te = np.concatenate([np.ones(len(ups1)), np.zeros(len(ups2))])  
    probs   = np.concatenate([prob_pos, prob_neg])
    cm      = confusion_matrix(labs_te, (probs>0.5).astype(int))
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                xticklabels=["no‐edge","edge"],
                yticklabels=["no‐edge","edge"])
    plt.title(f"Confusion — Stage {t}")
    plt.show()

# ------------------------------------------------
# 5. Summary of link‐pred accuracies
# ------------------------------------------------
df_res = pd.DataFrame(results).T
df_res.index.name = "Stage"
print("\n=== Link‐Prediction Acc. per Stage ===")
print(df_res.applymap(lambda x: f"{x*100:.2f}%"))

# ------------------------------------------------
# 6. Interpretability Dashboard
# ------------------------------------------------
drifts = []
for t in range(2, len(param_snaps)+1):
    θ_t = param_snaps[t-1]
    for k in range(1, t):
        θ_k   = param_snaps[k-1]
        F_k   = fishers[k-1].fisher
        # param‐drift
        pd2 = sum((θ_t[n]-θ_k[n]).pow(2).sum().item() for n in θ_k)
        pd  = np.sqrt(pd2)
        # EWC‐drift
        ed  = sum((F_k[n].cpu()*(θ_t[n]-θ_k[n])**2).sum().item() for n in θ_k)
        # forgetting ΔAcc
        a_kk = results[k][f"task_{k}"]
        a_kt = results[t][f"task_{k}"]
        da   = a_kk - a_kt
        drifts.append({
            "from_task":   k,
            "to_task":     t,
            "param_drift": pd,
            "ewc_drift":   ed,
            "delta_acc":   da
        })

dash = pd.DataFrame(drifts)
dash["from_to"] = dash["from_task"].astype(str)+"→"+dash["to_task"].astype(str)
print("\n=== Dashboard: Drift vs Forgetting ===")
print(dash[["from_to","param_drift","ewc_drift","delta_acc"]])
print(f"\nCorr param drift vs ΔAcc: {dash.param_drift.corr(dash.delta_acc):.4f}")
print(f"Corr EWC drift vs ΔAcc:   {dash.ewc_drift.corr(dash.delta_acc):.4f}")


# ------------------------------------------------
# Interpretability Dashboard (corrected)
# ------------------------------------------------
import numpy as np
import pandas as _pd   # use a fresh alias to avoid any pd shadowing

# Build drift records
records = []
for t in range(2, len(param_snaps) + 1):
    state_t = param_snaps[t-1]
    for k in range(1, t):
        state_k = param_snaps[k-1]
        Fk = fishers[k-1].fisher

        # 1) Parameter drift ‖θ_t − θ_k‖₂
        sq = sum((state_t[n] - state_k[n]).pow(2).sum().item() for n in state_k)
        param_drift = np.sqrt(sq)

        # 2) EWC‐weighted drift
        ewc_drift = sum((Fk[n].cpu() * (state_t[n] - state_k[n])**2).sum().item()
                        for n in state_k)

        # 3) Forgetting ΔAcc = Acc_k(k) − Acc_k(t)
        acc_kk = results[k].get(f"task_{k}", 0.0)
        acc_kt = results[t].get(f"task_{k}", 0.0)
        delta_acc = acc_kk - acc_kt

        records.append({
            "from_task":   k,
            "to_task":     t,
            "param_drift": param_drift,
            "ewc_drift":   ewc_drift,
            "delta_acc":   delta_acc
        })

# Create DataFrame
dashboard = _pd.DataFrame(records)

if not dashboard.empty:
    # Vectorized label
    dashboard["from_to"] = (
        dashboard["from_task"].astype(int).astype(str)
        + "→" +
        dashboard["to_task"].astype(int).astype(str)
    )

    # Print table
    print("\n=== Dashboard: Drift vs Forgetting ===")
    print(dashboard[["from_to","param_drift","ewc_drift","delta_acc"]])

    # Compute correlations
    corr_pd  = dashboard["param_drift"].corr(dashboard["delta_acc"])
    corr_ewc = dashboard["ewc_drift"].corr(dashboard["delta_acc"])
    print(f"\nCorrelation (param drift vs ΔAcc): {corr_pd:.4f}")
    print(f"Correlation (EWC drift vs ΔAcc):   {corr_ewc:.4f}")
else:
    print("Not enough task pairs for Interpretability Dashboard.")
