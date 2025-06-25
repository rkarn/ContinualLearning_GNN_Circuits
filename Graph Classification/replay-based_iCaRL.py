import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import dgl.nn as dglnn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load & preprocess CSV
# -------------------------------
df = pd.read_csv("all_circuits_features_graphclass.csv")
feat_cols = [
    "fan_in","fan_out","dist_to_output","is_primary_input",
    "is_primary_output","is_internal","is_key_gate",
    "degree_centrality","betweenness_centrality",
    "closeness_centrality","clustering_coefficient",
    "avg_fan_in_neighbors","avg_fan_out_neighbors"
]
df[feat_cols] = StandardScaler().fit_transform(df[feat_cols])

# -------------------------------
# 2. Build one DGLGraph per circuit
# -------------------------------
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

# -------------------------------
# 3. Four binary tasks: quartile q vs rest
# -------------------------------
qs = np.percentile(sizes, [25,50,75])
quarts = np.digitize(sizes, qs)  # 0..3
tasks = []
for q in range(4):
    pos = np.where(quarts==q)[0]
    neg = np.where(quarts!=q)[0]
    # train/test split
    p_tr, p_te = train_test_split(pos, test_size=0.2, random_state=42)
    n_tr, n_te = train_test_split(neg, test_size=0.2, random_state=42)
    tasks.append({
        "q": q,
        "train_pos": p_tr, "train_neg": n_tr,
        "test_pos":  p_te, "test_neg":  n_te
    })

# -------------------------------
# 4. GCN + FC readout
# -------------------------------
class GCNEnc(nn.Module):
    def __init__(self, in_feats, hid):
        super().__init__()
        self.c1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.c2 = dglnn.GraphConv(hid,   hid,   allow_zero_in_degree=True)
        self.read = nn.Linear(hid, 2)
    def forward(self, g, x):
        h = torch.relu(self.c1(g, x))
        h = torch.relu(self.c2(g, h))
        hg = h.mean(dim=0, keepdim=True)
        return self.read(hg)

# -------------------------------
# 5. iCaRL‐style memory + distillation
# -------------------------------
memory_per_class = 20
temperature      = 2.0
lambda_distill   = 1.0
kl_loss          = nn.KLDivLoss(reduction='batchmean')
exemplar_sets    = {0: [], 1: []}

def update_exemplars(label, new_idxs):
    """Keep at most memory_per_class exemplars per label."""
    np.random.shuffle(new_idxs)
    exemplar_sets[label] = new_idxs[:memory_per_class]

def get_all_exemplars():
    return exemplar_sets[0] + exemplar_sets[1]  # list of graph indices

# -------------------------------
# 6. Train w/ iCaRL
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GCNEnc(len(feat_cols), 64).to(device)
opt    = optim.Adam(model.parameters(), lr=1e-3)
results = {}
param_snaps, logits_snaps, memory_snaps = [], [], []
old_model = None

for stage, t in enumerate(tasks, start=1):
    q      = t["q"]
    p_tr   = t["train_pos"].tolist()
    n_tr   = t["train_neg"].tolist()
    p_te   = t["test_pos"]
    n_te   = t["test_neg"]

    print(f"\n=== Stage {stage}: quartile {q} vs rest ===")

    # freeze old model for distillation
    if old_model:
        old_model.eval()

    # training set = new + exemplars
    new_idxs = p_tr + n_tr
    exem_idxs= get_all_exemplars()
    train_idxs = new_idxs + exem_idxs
    train_labels = torch.tensor(
        [1]*len(p_tr) + [0]*len(n_tr) + 
        [1 if i in exemplar_sets[1] else 0 for i in exem_idxs],
        dtype=torch.long, device=device
    )

    # precompute old soft targets
    if old_model and exem_idxs:
        with torch.no_grad():
            snaps = []
            for i in exem_idxs:
                g = graphs[i].to(device)
                snaps.append(old_model(g, g.ndata["feat"].to(device))[0])
            old_logits = torch.stack(snaps, dim=0)
            old_soft   = F.softmax(old_logits/temperature, dim=1)

    # train 30 epochs
    model.train()
    for epoch in range(1, 31):
        opt.zero_grad()
        outs = []
        for i in train_idxs:
            g = graphs[i].to(device)
            outs.append(model(g, g.ndata["feat"].to(device))[0])
        outs = torch.stack(outs, dim=0)

        ce = nn.CrossEntropyLoss()(outs, train_labels)
        if old_model and exem_idxs:
            new_logp = F.log_softmax(outs[-len(exem_idxs):]/temperature, dim=1)
            dist = kl_loss(new_logp, old_soft) * (temperature**2)
            loss = ce + lambda_distill * dist
        else:
            loss = ce

        loss.backward(); opt.step()
        if epoch in (1,15,30):
            print(f"  Epoch {epoch:02d}, loss: {loss.item():.4f}")

    # update exemplars for this binary task
    update_exemplars(1, p_tr)
    update_exemplars(0, n_tr)
    memory_snaps.append(get_all_exemplars())

    # snapshots
    param_snaps.append({n:p.clone().detach().cpu() for n,p in model.named_parameters()})
    with torch.no_grad():
        all_logits = []
        for g in graphs:
            g = g.to(device)
            all_logits.append(model(g, g.ndata["feat"].to(device))[0].cpu().numpy())
    logits_snaps.append(np.vstack(all_logits))

    # evaluate on all seen tasks
    model.eval()
    with torch.no_grad():
        all_preds = logits_snaps[-1].argmax(axis=1)

    results[stage] = {}
    for k in range(stage):
        tp = tasks[k]["test_pos"]
        tn = tasks[k]["test_neg"]
        idxs = np.concatenate([tp, tn])
        y_true = np.concatenate([np.ones(len(tp)), np.zeros(len(tn))])
        y_pred = all_preds[idxs]
        acc = accuracy_score(y_true, y_pred)
        results[stage][f"task_{k+1}"] = acc
        print(f"  → Acc task{k+1} after stage{stage}: {acc*100:.2f}%")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["rest","quart"+str(tasks[k]["q"])],
                    yticklabels=["rest","quart"+str(tasks[k]["q"])])
        plt.title(f"Confusion stage{stage}/task{k+1}")
        plt.show()

    old_model = copy.deepcopy(model).eval().to(device)

# -------------------------------
# 7. Summary of accuracies
# -------------------------------
df_res = pd.DataFrame(results).T
df_res.index.name = "Stage"
print("\n=== Accuracies per stage & task ===")
print(df_res.applymap(lambda x: f"{x*100:.2f}%"))

# -------------------------------
# 8. Interpretability Dashboard
# -------------------------------
records = []
num_t = len(param_snaps)
for t in range(2, num_t+1):
    θ_t = param_snaps[t-1]
    logits_t = logits_snaps[t-1]
    M_k = memory_snaps[t-2]  # memory after stage t-1
    for k in range(1, t):
        θ_k = param_snaps[k-1]
        logits_k = logits_snaps[k-1]

        # 1) param drift
        sq = sum((θ_t[n]-θ_k[n]).pow(2).sum().item() for n in θ_k)
        pd_ = np.sqrt(sq)
        # 2) distill loss on M_k
        if M_k:
            diffs = logits_t[M_k] - logits_k[M_k]
            dl = np.mean(np.sum(diffs**2, axis=1))
        else:
            dl = np.nan
        # 3) forgetting ΔAcc
        da = results[k][f"task_{k}"] - results[t][f"task_{k}"]

        records.append({
            "from_task":   k,
            "to_task":     t,
            "param_drift": pd_,
            "distill_loss": dl,
            "delta_acc":   da
        })

dash = pd.DataFrame(records)
dash["from_to"] = dash["from_task"].astype(str)+"→"+dash["to_task"].astype(str)
print("\n=== iCaRL Dashboard: Drift vs Forgetting ===")
print(dash[["from_to","param_drift","distill_loss","delta_acc"]])
print(f"\nCorr(param drift vs ΔAcc): {dash.param_drift.corr(dash.delta_acc):.4f}")
print(f"Corr(distill loss vs ΔAcc): {dash.distill_loss.corr(dash.delta_acc):.4f}")
