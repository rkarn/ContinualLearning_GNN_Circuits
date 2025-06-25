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
# 1. Load & preprocess node‐feature CSV
# ---------------------------------------
df = pd.read_csv("all_circuits_features_graphclass.csv")
feat_cols = [
    "fan_in","fan_out","dist_to_output","is_primary_input",
    "is_primary_output","is_internal","is_key_gate",
    "degree_centrality","betweenness_centrality",
    "closeness_centrality","clustering_coefficient",
    "avg_fan_in_neighbors","avg_fan_out_neighbors"
]
df[feat_cols] = StandardScaler().fit_transform(df[feat_cols])

# ---------------------------------------
# 2. Build one DGLGraph per circuit_id
# ---------------------------------------
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
    if not edges:
        continue
    src, dst = zip(*edges)
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=len(nodes))
    g = dgl.add_self_loop(g)
    g.ndata["feat"] = torch.tensor(sub[feat_cols].values, dtype=torch.float32)
    graphs.append(g)
    sizes.append(len(nodes))

graphs = np.array(graphs)
sizes  = np.array(sizes)

# ---------------------------------------
# 3. Define 4 binary tasks by quartile
# ---------------------------------------
qs    = np.percentile(sizes, [25,50,75])
quarts= np.digitize(sizes, qs)   # 0..3

tasks = []
for q in range(4):
    pos = np.where(quarts==q)[0].tolist()
    neg = np.where(quarts!=q)[0].tolist()
    if len(pos)<5:
        continue
    p_tr, p_te = train_test_split(pos, test_size=0.2, random_state=42)
    n_tr, n_te = train_test_split(neg, test_size=0.2, random_state=42)
    tasks.append({
        "q": q,
        "train_pos": p_tr, "train_neg": n_tr,
        "test_pos":  p_te, "test_neg":  n_te
    })

# ---------------------------------------
# 4. CoPEModel: GCN + pretext head + classifier
# ---------------------------------------
class CoPEModel(nn.Module):
    def __init__(self, in_feats, hid, drop=0.5):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.bn1   = nn.BatchNorm1d(hid)
        self.conv2 = dglnn.GraphConv(hid, hid, allow_zero_in_degree=True)
        self.bn2   = nn.BatchNorm1d(hid)
        self.drop  = nn.Dropout(drop)
        self.pre   = nn.Linear(hid, in_feats)
        self.cls   = nn.Linear(hid, 2)

    def encode(self, g, x):
        h = F.relu(self.bn1(self.conv1(g, x)))
        h = self.drop(h)
        h = F.relu(self.bn2(self.conv2(g, h)))
        return self.drop(h)

    def reconstruct(self, g, x):
        h = self.encode(g, x)
        return self.pre(h)

    def forward(self, g, x):
        h  = self.encode(g, x)
        hg = h.mean(dim=0, keepdim=True)
        return self.cls(hg)

# ---------------------------------------
# 5. Hyperparameters & optimizers
# ---------------------------------------
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model      = CoPEModel(len(feat_cols), hid=64, drop=0.4).to(device)
opt_pre    = optim.Adam(
    list(model.conv1.parameters())+list(model.bn1.parameters())+
    list(model.conv2.parameters())+list(model.bn2.parameters())+
    list(model.pre.parameters()),
    lr=5e-4, weight_decay=1e-5
)
opt_cls    = optim.Adam(
    list(model.conv1.parameters())+list(model.bn1.parameters())+
    list(model.conv2.parameters())+list(model.bn2.parameters())+
    list(model.cls.parameters()),
    lr=5e-3, weight_decay=1e-5
)
mask_ratio = 0.2
pre_epochs = 30
cls_epochs = 40

# ---------------------------------------
# 6. Global self‐supervised warm‐up
# ---------------------------------------
print("Global self‐supervised pretraining...")
for ep in range(1, pre_epochs+1):
    total_loss = 0
    for g in graphs:
        g = g.to(device)
        x = g.ndata["feat"]
        dims = np.random.choice(x.shape[1], int(mask_ratio*x.shape[1]), replace=False)
        xm   = x.clone(); xm[:,dims] = 0
        recon= model.reconstruct(g, xm)
        loss = F.mse_loss(recon[:,dims], x[:,dims])
        opt_pre.zero_grad(); loss.backward(); opt_pre.step()
        total_loss += loss.item()
    if ep in (1, pre_epochs//2, pre_epochs):
        print(f"  Ep {ep}/{pre_epochs}, avg pre‐loss: {total_loss/len(graphs):.4f}")

# ---------------------------------------
# 7. Stage‐wise CoPE training & eval
# ---------------------------------------
results = {}
param_snaps   = []
pretext_snaps = []

for stage, t in enumerate(tasks, start=1):
    q      = t["q"]
    p_tr, n_tr = t["train_pos"], t["train_neg"]
    p_te, n_te = t["test_pos"],  t["test_neg"]
    tr_idxs    = p_tr + n_tr
    te_idxs    = p_te + n_te

    print(f"\n=== Stage {stage}: CoPE quartile {q} vs rest ===")

    # (a) Task‐specific self‐supervised pretrain
    for ep in range(1, pre_epochs+1):
        total_loss = 0
        for i in tr_idxs:
            g = graphs[i].to(device)
            x = g.ndata["feat"]
            dims = np.random.choice(x.shape[1], int(mask_ratio*x.shape[1]), replace=False)
            xm   = x.clone(); xm[:,dims]=0
            recon= model.reconstruct(g, xm)
            loss = F.mse_loss(recon[:,dims], x[:,dims])
            opt_pre.zero_grad(); loss.backward(); opt_pre.step()
            total_loss += loss.item()
        if ep in (1, pre_epochs//2, pre_epochs):
            print(f"  [Pre{stage}] Ep {ep}/{pre_epochs}, avg pre‐loss: {total_loss/len(tr_idxs):.4f}")

    # (b) Supervised fine‐tune
    y_tr = torch.tensor([1]*len(p_tr)+[0]*len(n_tr),
                        dtype=torch.long, device=device)
    for ep in range(1, cls_epochs+1):
        outs=[]
        for i in tr_idxs:
            g = graphs[i].to(device)
            outs.append(model(g, g.ndata["feat"].to(device))[0])
        outs = torch.stack(outs)
        loss = nn.CrossEntropyLoss()(outs, y_tr)
        opt_cls.zero_grad(); loss.backward(); opt_cls.step()
        if ep in (1, cls_epochs//2, cls_epochs):
            print(f"  [Cls{stage}] Ep {ep}/{cls_epochs}, cls‐loss: {loss.item():.4f}")

    # snapshots
    param_snaps.append({n:p.clone().cpu()
                        for n,p in model.state_dict().items()})
    pretext_snaps.append({n:p.clone().cpu()
                          for n,p in model.pre.state_dict().items()})

    # evaluate on all seen tasks
    model.eval()
    preds = []
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            preds.append(model(g, g.ndata["feat"].to(device))[0].cpu().numpy())
    preds = np.stack(preds).argmax(axis=1)

    results[stage] = {}
    for k in range(stage):
        tp = tasks[k]["test_pos"]; tn = tasks[k]["test_neg"]
        idxs = tp + tn
        y_true = np.array([1]*len(tp)+[0]*len(tn))
        y_pred = preds[idxs]
        acc    = accuracy_score(y_true, y_pred)
        results[stage][f"task_{k+1}"] = acc
        print(f"  → Acc task{k+1}: {acc*100:.2f}%")
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Stage{stage}/task{k+1} Confusion"); plt.show()

# ---------------------------------------
# 8. Results summary
# ---------------------------------------
df_res = pd.DataFrame(results).T; df_res.index.name="Stage"
print("\n=== Test accuracies per stage & task ===")
print(df_res.applymap(lambda x: f"{x*100:.2f}%"))

# ---------------------------------------
# 9. Interpretability Dashboard
# ---------------------------------------
records = []
for t in range(2, len(param_snaps)+1):
    θ_t = param_snaps[t-1]
    for k in range(1, t):
        θ_k = param_snaps[k-1]
        # 1) param drift
        pd2 = sum((θ_t[n]-θ_k[n]).pow(2).sum().item() for n in θ_k)
        pd  = np.sqrt(pd2)
        # 2) CoPE‐shift: pretext‐head drift
        ph_t = pretext_snaps[t-1]; ph_k = pretext_snaps[k-1]
        ss   = sum((ph_t[n]-ph_k[n]).pow(2).sum().item() for n in ph_k)
        shift = np.sqrt(ss)
        # 3) forgetting
        da = results[k][f"task_{k}"] - results[t][f"task_{k}"]
        records.append({
            "from_task": k, "to_task": t,
            "param_drift": pd,
            "cope_shift":  shift,
            "delta_acc":   da
        })

dash = pd.DataFrame(records)
dash["from_to"] = dash["from_task"].astype(str)+"→"+dash["to_task"].astype(str)
print("\n=== CoPE Interpretability: Drift vs Forgetting ===")
print(dash[["from_to","param_drift","cope_shift","delta_acc"]])
print(f"\nCorr(param_drift vs ΔAcc): {dash.param_drift.corr(dash.delta_acc):.4f}")
print(f"Corr(CoPE_shift vs ΔAcc): {dash.cope_shift.corr(dash.delta_acc):.4f}")
