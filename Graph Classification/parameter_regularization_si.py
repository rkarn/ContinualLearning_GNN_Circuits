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

# 1) Load & preprocess CSV
df = pd.read_csv("all_circuits_features_graphclass.csv")
feat_cols = [
    "fan_in","fan_out","dist_to_output","is_primary_input",
    "is_primary_output","is_internal","is_key_gate",
    "degree_centrality","betweenness_centrality",
    "closeness_centrality","clustering_coefficient",
    "avg_fan_in_neighbors","avg_fan_out_neighbors"
]
df[feat_cols] = StandardScaler().fit_transform(df[feat_cols])

# 2) Build one graph per circuit_id and record sizes
graphs, sizes = [], []
for cid, sub in df.groupby("circuit_id"):
    nodes = sub["node"].tolist()
    idx   = {n:i for i,n in enumerate(nodes)}
    edges = set()
    for _,r in sub.iterrows():
        u = idx[r["node"]]
        k = int(r["fan_in"])
        # take first k nodes as pseudo‐inputs
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

# 3) Quartile labels 0–3
qs = np.percentile(sizes, [25,50,75])
labels = np.digitize(sizes, qs)  # each graph’s quartile

# 4) Build 4 binary tasks: quartile q vs rest, with 80/20 splits
tasks = []
for q in range(4):
    pos = np.where(labels==q)[0]
    neg = np.where(labels!=q)[0]
    if len(pos)<5:
        continue
    p_tr, p_te = train_test_split(pos, test_size=0.2, random_state=42)
    n_tr, n_te = train_test_split(neg, test_size=0.2, random_state=42)
    tasks.append({
        "q": q,
        "train_pos": p_tr, "train_neg": n_tr,
        "test_pos":  p_te, "test_neg":  n_te
    })

# 5) GCN + readout
class GCNClassifier(nn.Module):
    def __init__(self, in_feats, hid):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid,   hid,   allow_zero_in_degree=True)
        self.read  = nn.Linear(hid, 2)
    def forward(self, g, x):
        h = torch.relu(self.conv1(g, x))
        h = torch.relu(self.conv2(g, h))
        hg= h.mean(dim=0, keepdim=True)
        return self.read(hg)

# 6) Synaptic Intelligence (SI)
class SI:
    def __init__(self, model, xi=0.1):
        self.xi = xi
        self.omega = {n: torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
        self.theta_old = {n: p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}
    def begin_task(self, model):
        self.prev_theta = {n: p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}
        self.path_omega = {n: torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
    def accumulate(self, model):
        for n,p in model.named_parameters():
            if not p.requires_grad: continue
            grad  = p.grad.data
            delta = p.data - self.prev_theta[n]
            self.path_omega[n] += -grad * delta
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

# 7) Train w/ SI
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model       = GCNClassifier(len(feat_cols), hid=64).to(device)
opt         = optim.Adam(model.parameters(), lr=1e-3)
si          = SI(model, xi=0.1)
lambda_si   = 1000.0
results     = {}
theta_snaps = []
omega_snaps = []

for stage, t in enumerate(tasks, start=1):
    q = t["q"]
    print(f"\n=== Stage {stage}: quartile {q} vs rest ===")
    # prepare train sets
    P = t["train_pos"]; Nn = t["train_neg"]
    train_idxs = np.concatenate([P, Nn])
    y_tr = torch.tensor([1]*len(P)+[0]*len(Nn), dtype=torch.long, device=device)

    si.begin_task(model)
    model.train()
    for epoch in (1,10,50):
        opt.zero_grad()
        outs=[]
        for i in train_idxs:
            g = graphs[i].to(device)
            outs.append(model(g, g.ndata["feat"].to(device))[0])
        outs = torch.stack(outs)
        loss = nn.CrossEntropyLoss()(outs, y_tr)
        if stage>1:
            loss = loss + lambda_si * si.penalty(model)
        loss.backward()
        si.accumulate(model)
        opt.step()
        print(f"  Epoch {epoch:02d}, loss: {loss.item():.4f}")

    si.end_task(model)
    # snapshots
    theta_snaps.append({n:p.clone().cpu() for n,p in model.named_parameters() if p.requires_grad})
    omega_snaps.append({n:v.clone().cpu() for n,v in si.omega.items()})

    # evaluate on all seen tasks
    model.eval()
    with torch.no_grad():
        logits = []
        for g in graphs:
            g = g.to(device)
            logits.append(model(g, g.ndata["feat"].to(device))[0].cpu().numpy())
        logits = np.vstack(logits)
        preds  = logits.argmax(axis=1)

    results[stage] = {}
    for k in range(stage):
        pos_k = tasks[k]["test_pos"]
        neg_k = tasks[k]["test_neg"]
        idxs = np.concatenate([pos_k, neg_k])
        y_true = np.concatenate([np.ones(len(pos_k)), np.zeros(len(neg_k))])
        y_pred = preds[idxs]
        acc = accuracy_score(y_true, y_pred)
        results[stage][f"task_{k+1}"] = acc
        print(f"  → Acc task{k+1} after stage{stage}: {acc*100:.2f}%")
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion — stage{stage}/task{k+1}")
        plt.show()

# 8) Summary
df_res = pd.DataFrame(results).T
df_res.index.name="Stage"
print("\n=== Accuracies per stage & task ===")
print(df_res.applymap(lambda x: f"{x*100:.2f}%"))

# 9) Interpretability Dashboard
records=[]
for t in range(2, len(theta_snaps)+1):
    θ_t = theta_snaps[t-1]
    for k in range(1, t):
        θ_k = theta_snaps[k-1]
        Ω_k = omega_snaps[k-1]
        # param drift
        pd2 = sum((θ_t[n]-θ_k[n]).pow(2).sum().item() for n in θ_k)
        pd_  = np.sqrt(pd2)
        # SI‐drift
        sid = sum((Ω_k[n] * (θ_t[n]-θ_k[n])**2).sum().item() for n in θ_k)
        # forgetting
        da  = results[k][f"task_{k}"] - results[t][f"task_{k}"]
        records.append({
            "from": k, "to": t,
            "param_drift": pd_,
            "si_drift":    sid,
            "delta_acc":   da
        })

dash = pd.DataFrame(records)
dash["from_to"] = dash["from"].astype(str)+"→"+dash["to"].astype(str)
print("\n=== SI Dashboard: Drift vs Forgetting ===")
print(dash[["from_to","param_drift","si_drift","delta_acc"]])
print(f"Corr(param drift vs ΔAcc): {dash.param_drift.corr(dash.delta_acc):.4f}")
print(f"Corr(SI drift vs ΔAcc):    {dash.si_drift.corr(dash.delta_acc):.4f}")
