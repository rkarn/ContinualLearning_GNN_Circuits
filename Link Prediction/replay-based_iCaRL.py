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

# 1) Load & preprocess
df = pd.read_csv("all_circuits_features.csv")
feat_cols = ["fan_in","fan_out","dist_to_output","is_primary_input",
             "is_primary_output","is_internal","is_key_gate",
             "degree_centrality","betweenness_centrality",
             "closeness_centrality","clustering_coefficient",
             "avg_fan_in_neighbors","avg_fan_out_neighbors"]
df = df.dropna(subset=feat_cols)
df[feat_cols] = StandardScaler().fit_transform(df[feat_cols].astype(float))

# Build base graph
nodes   = df["node"].tolist()
nid2idx = {n:i for i,n in enumerate(nodes)}
N       = len(nodes)
edges = set()
src_cands = df[df["fan_out"]>0]["node"]
for _, r in df.iterrows():
    u = nid2idx[r["node"]]; k = int(r["fan_in"])
    for vname in src_cands[:k]:
        v = nid2idx[vname]; edges.add((u,v))
pos_all = np.array(list(edges))

g = dgl.graph((torch.tensor(pos_all[:,0]), torch.tensor(pos_all[:,1])),
              num_nodes=N)
g = dgl.add_self_loop(g)
g.ndata["feat"] = torch.tensor(df[feat_cols].values, dtype=torch.float32)

# 2) Define 4 tasks by gate‐type groups
group_defs = [["and","or"],["nand","nor"],["xor","xnor"],["buf","not"]]
tasks = []
for grp in group_defs:
    # positive edges within grp
    mask = df["gate_type"].isin(grp).to_numpy()
    idxs = np.where(mask)[0]
    mpos = np.isin(pos_all[:,0], idxs) & np.isin(pos_all[:,1], idxs)
    pos = pos_all[mpos]
    # force at least a few edges
    if len(pos) < 5:
        pos = pos_all[np.random.choice(len(pos_all), 10, replace=False)]
    n = len(pos)
    perm = np.random.permutation(n)
    split = int(0.8*n)
    tasks.append({
        "name": f"{grp}",
        "train_pos": pos[perm[:split]],
        "test_pos":  pos[perm[split:]]
    })

# add negatives equal to positives
for t in tasks:
    pos = t["train_pos"]
    n   = len(pos)
    neg = set()
    while len(neg)<n:
        u = random.randrange(N); v = random.randrange(N)
        if u!=v and (u,v) not in edges:
            neg.add((u,v))
    t["train_neg"] = np.array(list(neg))
    # same for test
    pos_te = t["test_pos"]; m = len(pos_te)
    neg_te = set()
    while len(neg_te)<m:
        u,v = random.randrange(N), random.randrange(N)
        if u!=v and (u,v) not in edges:
            neg_te.add((u,v))
    t["test_neg"] = np.array(list(neg_te))

# 3) GCN encoder
class GCNEnc(nn.Module):
    def __init__(self, in_feats, hid):
        super().__init__()
        self.c1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.c2 = dglnn.GraphConv(hid, hid,   allow_zero_in_degree=True)
    def forward(self, g, x):
        h = F.relu(self.c1(g, x)); return self.c2(g, h)

# 4) iCaRL replay + distill
memory, results = [], {}
param_snaps, embed_snaps, mem_snaps = [], [], []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc    = GCNEnc(len(feat_cols), 64).to(device)
opt    = optim.Adam(enc.parameters(), lr=1e-2)

old_enc = None
for stage, t in enumerate(tasks, start=1):
    print(f"\n=== Stage{stage}: {t['name']} ===")
    # build train set
    P = [(u,v,1) for u,v in t["train_pos"]]
    Nn= [(u,v,0) for u,v in t["train_neg"]]
    data = P+Nn+memory

    us = torch.tensor([u for u,_,_ in data], dtype=torch.long, device=device)
    vs = torch.tensor([v for _,v,_ in data], dtype=torch.long, device=device)
    lb = torch.tensor([l for _,_,l in data], dtype=torch.float32, device=device)

    # distill on memory if available
    if old_enc and memory:
        with torch.no_grad():
            h_old = old_enc(g, g.ndata["feat"].to(device))
            mus = torch.tensor([u for u,_,_ in memory], device=device)
            mvs = torch.tensor([v for _,v,_ in memory], device=device)
            old_score = (h_old[mus]*h_old[mvs]).sum(dim=1)

    enc.train()
    for ep in (1,10,50):
        opt.zero_grad()
        h = enc(g, g.ndata["feat"].to(device))
        sc = (h[us]*h[vs]).sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(sc, lb)
        if old_enc and memory:
            new_sc = (h[mus]*h[mvs]).sum(dim=1)
            loss += F.mse_loss(new_sc, old_score)
        loss.backward(); opt.step()
        print(f" Ep{ep}, loss {loss.item():.4f}")

    # update memory w/ new data
    memory += random.sample(P+Nn, min(100,len(P+Nn)))
    mem_snaps.append(list(memory))

    # snapshots
    param_snaps.append({n:p.cpu().clone() for n,p in enc.named_parameters()})
    with torch.no_grad():
        embed_snaps.append(enc(g,g.ndata["feat"].to(device)).cpu())

    # eval
    enc.eval()
    with torch.no_grad():
        h = enc(g, g.ndata["feat"].to(device))
    res = {}
    for k in range(stage):
        Pte = tasks[k]["test_pos"]; Nte = tasks[k]["test_neg"]
        us2 = torch.tensor(Pte[:,0], device=device); vs2 = torch.tensor(Pte[:,1], device=device)
        us3 = torch.tensor(Nte[:,0], device=device); vs3 = torch.tensor(Nte[:,1], device=device)
        scp = torch.sigmoid((h[us2]*h[vs2]).sum(1)).cpu().numpy()
        scn = torch.sigmoid((h[us3]*h[vs3]).sum(1)).cpu().numpy()
        pred = np.concatenate([scp>0.5, scn>0.5]).astype(int)
        lab  = np.concatenate([np.ones(len(scp)), np.zeros(len(scn))])
        acc = accuracy_score(lab, pred); res[f"task{k+1}"] = acc
        print(f"  Acc task{k+1}: {acc*100:.2f}%")
        cm = confusion_matrix(lab, pred)
        plt.figure(figsize=(3,3)); sns.heatmap(cm,annot=True,fmt="d"); plt.show()
    results[stage]=res
    old_enc = copy.deepcopy(enc).eval().to(device)

# summarize
df_res = pd.DataFrame(results).T; df_res.index.name="Stage"
print(df_res.applymap(lambda x: f"{x*100:.2f}%"))

# dashboard
records=[]
for t in range(2, len(param_snaps)+1):
    pt = param_snaps[t-1]; et=embed_snaps[t-1]
    for k in range(1,t):
        pk= param_snaps[k-1]; ek=embed_snaps[k-1]; mk=mem_snaps[k-1]
        # param drift
        sq=sum((pt[n]-pk[n]).pow(2).sum().item() for n in pk)
        pd_=np.sqrt(sq)
        # distill loss on mk
        if mk:
            us4=np.array([u for u,_,_ in mk]); vs4=np.array([v for _,v,_ in mk])
            dt = F.mse_loss((et[us4]*et[vs4]).sum(1),(ek[us4]*ek[vs4]).sum(1)).item()
        else:
            dt=np.nan
        da=results[k][f"task{k}"]-results[t][f"task{k}"]
        records.append({"from":k,"to":t,"pd":pd_,"dl":dt,"da":da})
dash=pd.DataFrame(records)
dash["task"]=dash["from"].astype(str)+"→"+dash["to"].astype(str)
print(dash[["task","pd","dl","da"]])
