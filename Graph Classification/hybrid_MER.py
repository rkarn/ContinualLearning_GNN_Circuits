import random
import copy
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

# ------------------------------------------------
# 1. Load & preprocess CSV
# ------------------------------------------------
df = pd.read_csv("all_circuits_features_graphclass.csv")
feat_cols = [
    "fan_in","fan_out","dist_to_output","is_primary_input",
    "is_primary_output","is_internal","is_key_gate",
    "degree_centrality","betweenness_centrality",
    "closeness_centrality","clustering_coefficient",
    "avg_fan_in_neighbors","avg_fan_out_neighbors"
]
df[feat_cols] = StandardScaler().fit_transform(df[feat_cols])

# ------------------------------------------------
# 2. Build one DGLGraph per circuit_id
# ------------------------------------------------
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
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)),
                  num_nodes=len(nodes))
    g = dgl.add_self_loop(g)
    g.ndata["feat"] = torch.tensor(sub[feat_cols].values,
                                   dtype=torch.float32)
    graphs.append(g)
    sizes.append(len(nodes))

graphs = np.array(graphs)
sizes  = np.array(sizes)

# ------------------------------------------------
# 3. Define 4 binary quartile tasks
# ------------------------------------------------
qs = np.percentile(sizes, [25,50,75])
quarts = np.digitize(sizes, qs)   # 0..3

tasks = []
for q in range(4):
    pos = np.where(quarts==q)[0]
    neg = np.where(quarts!=q)[0]
    if len(pos)<5:
        continue
    p_tr, p_te = train_test_split(pos, test_size=0.2, random_state=42)
    n_tr, n_te = train_test_split(neg, test_size=0.2, random_state=42)
    tasks.append({
        "q": q,
        "train_pos": p_tr.tolist(), "train_neg": n_tr.tolist(),
        "test_pos":  p_te.tolist(), "test_neg":  n_te.tolist()
    })

# ------------------------------------------------
# 4. GCN encoder + readout
# ------------------------------------------------
class GCNEnc(nn.Module):
    def __init__(self, in_feats, hid):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid,   hid,   allow_zero_in_degree=True)
        self.read  = nn.Linear(hid, 2)
    def forward(self, g, x):
        h = torch.relu(self.conv1(g, x))
        h = torch.relu(self.conv2(g, h))
        hg = h.mean(dim=0, keepdim=True)
        return self.read(hg)

# ------------------------------------------------
# 5. MER hyperparams
# ------------------------------------------------
inner_lr        = 0.005
meta_lr         = 0.01
memory_per_task = 200
mer_batch_size  = 64

# ------------------------------------------------
# 6. Training loop w/ MER
# ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GCNEnc(len(feat_cols), 64).to(device)
opt    = optim.SGD(model.parameters(), lr=meta_lr)
criterion = nn.CrossEntropyLoss()

memory_buffer   = []   # stored graph indices
memory_snapshots= []
param_snapshots = []
results         = {}
logits_snapshots= []

for stage, t in enumerate(tasks, start=1):
    q       = t["q"]
    p_tr, n_tr = t["train_pos"], t["train_neg"]
    p_te, n_te = t["test_pos"],  t["test_neg"]
    new_idxs   = p_tr + n_tr

    print(f"\n=== Stage {stage}: quartile {q} vs rest ===")
    model.train()
    for epoch in (1,10,50):
        # --- sample batches ---
        new_batch = random.sample(new_idxs, 
                                  min(mer_batch_size, len(new_idxs)))
        if len(memory_buffer)>=mer_batch_size:
            mem_batch = random.sample(memory_buffer, mer_batch_size)
        else:
            mem_batch = memory_buffer.copy()

        # 1) grad on new batch
        logits_new=[]
        for i in new_batch:
            g = graphs[i].to(device)
            logits_new.append(model(g, g.ndata["feat"].to(device))[0])
        logits_new = torch.stack(logits_new)
        y_new = torch.tensor(
            [1]*len(p_tr if q in range(4) else []) + [0]*len(n_tr),
            dtype=torch.long, device=device
        )
        loss_new = criterion(logits_new, y_new)
        grads_new = torch.autograd.grad(loss_new, model.parameters(),
                                        create_graph=True)

        # 2) inner update
        backup = [p.data.clone() for p in model.parameters()]
        for p, gnew in zip(model.parameters(), grads_new):
            p.data.sub_(inner_lr * gnew.data)

        # 3) grad on mem batch
        if mem_batch:
            logits_mem=[]
            for i in mem_batch:
                g = graphs[i].to(device)
                logits_mem.append(model(g, g.ndata["feat"].to(device))[0])
            logits_mem = torch.stack(logits_mem)
            # labels: we know exemplar origin was p_tr or n_tr
            y_mem = []
            for i in mem_batch:
                y_mem.append(1 if i in p_tr else 0)
            y_mem = torch.tensor(y_mem, dtype=torch.long, device=device)
            loss_mem = criterion(logits_mem, y_mem)
            grads_meta = torch.autograd.grad(loss_mem, model.parameters())
        else:
            grads_meta = grads_new

        # 4) restore & meta-step
        for p, b in zip(model.parameters(), backup):
            p.data.copy_(b)
        opt.zero_grad()
        for p, gm in zip(model.parameters(), grads_meta):
            p.grad = gm.data.clone()
        opt.step()

        print(f"  Epoch {epoch:02d}, loss_new: {loss_new.item():.4f}")

    # update memory
    sampled = random.sample(new_idxs, 
                            min(memory_per_task, len(new_idxs)))
    memory_buffer.extend(sampled)
    memory_snapshots.append(list(memory_buffer))

    # snapshot params & logits
    param_snapshots.append({
        n: v.clone().detach().cpu()
        for n,v in model.state_dict().items()
    })
    with torch.no_grad():
        all_logits=[]
        for g in graphs:
            g = g.to(device)
            all_logits.append(model(g, g.ndata["feat"].to(device))[0].cpu().numpy())
    logits_snapshots.append(np.vstack(all_logits))

    # evaluate on all seen tasks
    model.eval()
    preds = logits_snapshots[-1].argmax(axis=1)
    results[stage] = {}
    for k in range(stage):
        pp = tasks[k]["test_pos"]; nnk = tasks[k]["test_neg"]
        idxs = pp + nnk
        y_true = np.array([1]*len(pp)+[0]*len(nnk))
        y_pred = preds[idxs]
        acc = accuracy_score(y_true, y_pred)
        results[stage][f"task_{k+1}"] = acc
        print(f"  → Acc task{k+1} after stage{stage}: {acc*100:.2f}%")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["rest",f"quart{tasks[k]['q']}"],
                    yticklabels=["rest",f"quart{tasks[k]['q']}"])
        plt.title(f"Stage{stage}/task{k+1}")
        plt.show()

# ------------------------------------------------
# 7. Summary table
# ------------------------------------------------
df_res = pd.DataFrame(results).T
df_res.index.name = "Stage"
print("\n=== Accuracies per stage & task ===")
print(df_res.applymap(lambda x: f"{x*100:.2f}%"))

# ------------------------------------------------
# 8. Interpretability Dashboard
# ------------------------------------------------
import pandas as _pd

records = []
for t in range(2, len(param_snapshots)+1):
    θ_t = param_snapshots[t-1]
    for k in range(1, t):
        θ_k = param_snapshots[k-1]
        # param drift
        sq = sum((θ_t[n]-θ_k[n]).pow(2).sum().item() for n in θ_k)
        pd_ = np.sqrt(sq)
        # cosine sim of meta-grad? skip – use param drift only
        da  = results[k][f"task_{k}"] - results[t][f"task_{k}"]
        records.append({
            "from_task": k, "to_task": t,
            "param_drift": pd_, "delta_acc": da
        })

dash = _pd.DataFrame(records)
dash["from_to"] = dash["from_task"].astype(str)+"→"+dash["to_task"].astype(str)
print("\n=== A-GEM Dashboard: Drift vs Forgetting ===")
print(dash[["from_to","param_drift","delta_acc"]])
print(f"\nCorr(param_drift vs ΔAcc): {dash.param_drift.corr(dash.delta_acc):.4f}")
