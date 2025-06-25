import random
import numpy as np
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
# 1. Load & preprocess data
# ------------------------------------------------
import pandas as pd
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
        edge_set.add((u, v))

# arrays of all positive edges
pos_all = np.array(list(edge_set))

# ------------------------------------------------
# 2. Split into 4 tasks (randomly)
# ------------------------------------------------
np.random.shuffle(pos_all)
chunks = np.array_split(pos_all, 4)  # 4 tasks

tasks = []
for i, pos in enumerate(chunks, start=1):
    name = f"Task{i}"
    n    = len(pos)
    if n == 0:
        raise ValueError(f"No edges in {name}")
    perm  = np.random.permutation(n)
    split = int(0.8 * n)
    pos_tr = pos[perm[:split]]
    pos_te = pos[perm[split:]]
    # sample negatives equal to n
    pos_set = set(map(tuple, pos.tolist()))
    neg = []
    while len(neg) < n:
        u = np.random.randint(0, N)
        v = np.random.randint(0, N)
        if u != v and (u,v) not in pos_set:
            neg.append((u,v))
    neg = np.array(neg)
    neg_tr = neg[perm[:split]]
    neg_te = neg[perm[split:]]
    tasks.append({
        "name":      name,
        "train_pos": pos_tr, "train_neg": neg_tr,
        "test_pos":  pos_te, "test_neg":  neg_te
    })

# build DGL once
src_arr = pos_all[:,0]
dst_arr = pos_all[:,1]
g = dgl.graph((torch.tensor(src_arr), torch.tensor(dst_arr)), num_nodes=N)
g = dgl.add_self_loop(g)
g.ndata["feat"] = torch.tensor(df[feat_cols].values, dtype=torch.float32)

# ------------------------------------------------
# 3. GCN encoder + dot‐product decoder
# ------------------------------------------------
class GCNEnc(nn.Module):
    def __init__(self, in_feats, hid):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid, hid,   allow_zero_in_degree=True)
    def forward(self, graph, x):
        h = torch.relu(self.conv1(graph, x))
        return self.conv2(graph, h)

# ------------------------------------------------
# 4. Synaptic Intelligence
# ------------------------------------------------
class SynapticIntelligence:
    def __init__(self, model, xi=0.1):
        self.xi = xi
        # total importance
        self.omega = {n: torch.zeros_like(p.data)
                      for n,p in model.named_parameters() if p.requires_grad}
        # snapshot after last task
        self.theta_old = {n: p.data.clone().detach()
                          for n,p in model.named_parameters() if p.requires_grad}

    def begin_task(self, model):
        self.prev_theta = {n: p.data.clone().detach()
                           for n,p in model.named_parameters() if p.requires_grad}
        self.path_omega = {n: torch.zeros_like(p.data)
                           for n,p in model.named_parameters() if p.requires_grad}

    def accumulate(self, model):
        for n,p in model.named_parameters():
            if not p.requires_grad: continue
            grad  = p.grad.data
            delta = p.data - self.prev_theta[n]
            self.path_omega[n] += - grad * delta
            self.prev_theta[n] = p.data.clone().detach()

    def end_task(self, model):
        for n,p in model.named_parameters():
            if not p.requires_grad: continue
            delta = p.data - self.theta_old[n]
            denom = delta.pow(2) + self.xi
            omega_task = self.path_omega[n] / denom
            self.omega[n] += omega_task
            self.theta_old[n] = p.data.clone().detach()

    def penalty(self, model):
        loss = 0
        for n,p in model.named_parameters():
            if not p.requires_grad: continue
            loss += (self.omega[n] * (p - self.theta_old[n]).pow(2)).sum()
        return loss

# ------------------------------------------------
# 5. Training loop w/ SI
# ------------------------------------------------
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc       = GCNEnc(len(feat_cols), hid=64).to(device)
opt       = optim.Adam(enc.parameters(), lr=1e-2)
si        = SynapticIntelligence(enc, xi=0.1)
lambda_si = 1000.0
results   = {}
# snapshots
theta_snaps = []
omega_snaps = []

for stage, task in enumerate(tasks, start=1):
    print(f"\n=== Stage {stage}: {task['name']} ===")
    # prepare data
    P = torch.tensor(task["train_pos"], dtype=torch.long, device=device)
    Nn= torch.tensor(task["train_neg"], dtype=torch.long, device=device)
    pairs  = torch.cat([P, Nn], dim=0)
    labels = torch.cat([
        torch.ones(len(P)), torch.zeros(len(Nn))
    ], dim=0).to(device)

    # begin SI for this task
    si.begin_task(enc)
    enc.train()
    for epoch in (1,10,50):
        opt.zero_grad()
        h    = enc(g, g.ndata["feat"].to(device))
        logits = (h[pairs[:,0]] * h[pairs[:,1]]).sum(dim=1)
        loss   = F.binary_cross_entropy_with_logits(logits, labels)
        if stage > 1:
            loss = loss + lambda_si * si.penalty(enc)
        loss.backward()
        si.accumulate(enc)
        opt.step()
        print(f"  Epoch {epoch:02d}, loss: {loss.item():.4f}")

    # end SI for this task
    si.end_task(enc)
    # snapshot
    theta_snaps.append({n: p.clone().detach().cpu()
                        for n,p in enc.named_parameters()})
    omega_snaps.append({n: v.clone().cpu() for n,v in si.omega.items()})

    # evaluate on all seen tasks
    enc.eval()
    with torch.no_grad():
        h = enc(g, g.ndata["feat"].to(device))
    accs = {}
    for k, prev in enumerate(tasks[:stage], start=1):
        Pte = torch.tensor(prev["test_pos"], dtype=torch.long, device=device)
        Nte = torch.tensor(prev["test_neg"], dtype=torch.long, device=device)
        ups1,vps1 = Pte[:,0], Pte[:,1]
        ups2,vps2 = Nte[:,0], Nte[:,1]
        prob_pos = torch.sigmoid((h[ups1]*h[vps1]).sum(dim=1)).cpu().numpy()
        prob_neg = torch.sigmoid((h[ups2]*h[vps2]).sum(dim=1)).cpu().numpy()
        preds    = np.concatenate([prob_pos>0.5, prob_neg>0.5]).astype(int)
        labs     = np.concatenate([np.ones(len(ups1)), np.zeros(len(ups2))])
        accs[f"task_{k}"] = accuracy_score(labs, preds)
        print(f"  → Acc task {k} after stage {stage}: {accs[f'task_{k}']*100:.2f}%")
    results[stage] = accs

    # confusion matrix for this task
    labs_te = np.concatenate([np.ones(len(prev["test_pos"])),
                              np.zeros(len(prev["test_neg"]))])
    pairs_te = np.vstack([prev["test_pos"], prev["test_neg"]])
    ups_te = torch.tensor(pairs_te[:,0], device=device)
    vps_te = torch.tensor(pairs_te[:,1], device=device)
    with torch.no_grad():
        probs = torch.sigmoid((h[ups_te]*h[vps_te]).sum(dim=1)).cpu().numpy()
    preds = (probs>0.5).astype(int)
    cm    = confusion_matrix(labs_te, preds)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["no‐edge","edge"],
                yticklabels=["no‐edge","edge"])
    plt.title(f"Confusion — Stage {stage}")
    plt.show()

# ------------------------------------------------
# 6. Summary of link‐prediction accuracies
# ------------------------------------------------
df_res = pd.DataFrame(results).T
df_res.index.name = "Stage"
print("\n=== Link‐Pred Accuracies per Stage ===")
print(df_res.applymap(lambda x: f"{x*100:.2f}%"))

# ------------------------------------------------
# 7. Interpretability Dashboard for SI
# ------------------------------------------------
import pandas as pd2

records = []
num_tasks = len(theta_snaps)
for t in range(2, num_tasks+1):
    θ_t = theta_snaps[t-1]
    for k in range(1, t):
        θ_k = theta_snaps[k-1]
        Ω_k = omega_snaps[k-1]
        # param drift
        sq = sum((θ_t[n]-θ_k[n]).pow(2).sum().item() for n in θ_k)
        pd_ = np.sqrt(sq)
        # SI‐weighted drift
        sid = sum((Ω_k[n] * (θ_t[n]-θ_k[n])**2).sum().item() for n in θ_k)
        # forgetting
        a_kk = results[k][f"task_{k}"]
        a_kt = results[t][f"task_{k}"]
        da   = a_kk - a_kt
        records.append({
            "from_task":   k,
            "to_task":     t,
            "param_drift": pd_,
            "si_drift":    sid,
            "delta_acc":   da
        })

dash = pd2.DataFrame(records)
if not dash.empty:
    dash["from_to"] = (
        dash["from_task"].astype(int).astype(str)
        + "→" +
        dash["to_task"].astype(int).astype(str)
    )
    print("\n=== SI Dashboard: Drift vs Forgetting ===")
    print(dash[["from_to","param_drift","si_drift","delta_acc"]])
    print(f"\nCorr(param drift vs ΔAcc): {dash.param_drift.corr(dash.delta_acc):.4f}")
    print(f"Corr(si drift vs ΔAcc):    {dash.si_drift.corr(dash.delta_acc):.4f}")
else:
    print("Not enough task pairs to build SI dashboard.")
