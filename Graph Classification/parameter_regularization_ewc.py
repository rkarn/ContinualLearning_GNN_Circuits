import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import dgl.nn as dglnn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load & preprocess
df = pd.read_csv("all_circuits_features_graphclass.csv")
feat_cols = [
    "fan_in","fan_out","dist_to_output","is_primary_input",
    "is_primary_output","is_internal","is_key_gate",
    "degree_centrality","betweenness_centrality",
    "closeness_centrality","clustering_coefficient",
    "avg_fan_in_neighbors","avg_fan_out_neighbors"
]
df[feat_cols] = StandardScaler().fit_transform(df[feat_cols])

# 2) Build one graph per circuit_id
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

# quartile labels 0–3
qs = np.percentile(sizes, [25,50,75])
quarts = np.digitize(sizes, qs)  # each graph’s quartile

# 3) Four binary tasks: quartile q vs rest
tasks = []
for q in range(4):
    pos = np.where(quarts==q)[0]
    neg = np.where(quarts!=q)[0]
    tasks.append((pos, neg))

# 4) GCN + EWC helper
class GCNEnc(nn.Module):
    def __init__(self, in_feats, hid):
        super().__init__()
        self.c1 = dglnn.GraphConv(in_feats, hid, allow_zero_in_degree=True)
        self.c2 = dglnn.GraphConv(hid, hid,   allow_zero_in_degree=True)
        self.read = nn.Linear(hid, 2)

    def forward(self, g, x):
        h = torch.relu(self.c1(g, x))
        h = torch.relu(self.c2(g, h))
        hg= h.mean(dim=0, keepdim=True)
        return self.read(hg)

class EWC:
    def __init__(self, model, graphs, task_pos, device):
        self.device = device
        # save params
        self.params = {n:p.clone().detach().to(device)
                       for n,p in model.named_parameters()}
        # init fisher
        self.fisher = {n:torch.zeros_like(p,device=device)
                       for n,p in model.named_parameters()}
        model.eval()
        # compute on train positives
        for i in task_pos:
            g = graphs[i].to(device)
            out = model(g, g.ndata["feat"].to(device))
            logp= nn.functional.log_softmax(out, dim=1)[0,1]
            model.zero_grad(); logp.backward()
            for n,p in model.named_parameters():
                self.fisher[n] += p.grad.detach()**2
        # normalize
        for n in self.fisher:
            self.fisher[n] /= len(task_pos)

    def penalty(self, model):
        loss=0
        for n,p in model.named_parameters():
            loss += (self.fisher[n] * (p-self.params[n])**2).sum()
        return loss

# 5) Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GCNEnc(len(feat_cols), hid=64).to(device)
opt    = optim.Adam(model.parameters(), lr=1e-3)
lambda_ewc=500.0
ewc_list=[]
results   = {}
param_snaps=[]

for stage,(pos,neg) in enumerate(tasks, start=1):
    print(f"\n=== Stage {stage}: quartile {stage-1} vs rest ===")
    # split train/test
    p_tr, p_te = train_test_split(pos, test_size=0.2, random_state=42)
    n_tr, n_te = train_test_split(neg, test_size=0.2, random_state=42)
    train_idx = np.concatenate([p_tr, n_tr])
    train_lbl = torch.tensor(
        [1]*len(p_tr)+[0]*len(n_tr), dtype=torch.long, device=device
    )

    # train
    model.train()
    for epoch in (1,10,50):
        opt.zero_grad()
        outs=[]
        for i in train_idx:
            g=graphs[i].to(device)
            outs.append(model(g, g.ndata["feat"].to(device))[0])
        outs = torch.stack(outs)
        loss = nn.CrossEntropyLoss()(outs, train_lbl)
        if ewc_list:
            loss += (lambda_ewc/2)*sum(e.penalty(model) for e in ewc_list)
        loss.backward(); opt.step()
        print(f"  Epoch {epoch:02d}, loss: {loss.item():.4f}")

    # snapshot & fisher
    ewc_list.append(EWC(model, graphs, p_tr, device))
    param_snaps.append({n:p.clone().detach().cpu()
                        for n,p in model.state_dict().items()})

    # evaluate all seen tasks
    model.eval()
    with torch.no_grad():
        logits = [model(g.to(device), g.ndata["feat"].to(device))[0].cpu().numpy()
                  for g in graphs]
    logits = np.vstack(logits)
    preds  = logits.argmax(axis=1)

    results[stage]={}
    for k in range(stage):
        pos_k, neg_k = tasks[k]
        _,pte_k = train_test_split(pos_k, test_size=0.2, random_state=42)
        _,nte_k = train_test_split(neg_k, test_size=0.2, random_state=42)
        idxs = np.concatenate([pte_k, nte_k])
        lbls = np.concatenate([np.ones(len(pte_k)), np.zeros(len(nte_k))])
        prd  = preds[idxs]
        acc  = accuracy_score(lbls, prd)
        results[stage][f"task_{k+1}"] = acc
        print(f"  → Acc task{k+1}: {acc*100:.2f}%")
        cm=confusion_matrix(lbls, prd)
        sns.heatmap(cm,annot=True,fmt="d"); plt.show()

# summary
df_res=pd.DataFrame(results).T
df_res.index.name="Stage"
print(df_res.applymap(lambda x:f"{x*100:.2f}%"))

# dashboard
records=[]
for t in range(2,len(param_snaps)+1):
    θ_t=param_snaps[t-1]
    for k in range(1,t):
        θ_k=param_snaps[k-1]
        # param drift
        pd2=sum((θ_t[n]-θ_k[n]).pow(2).sum().item() for n in θ_k)
        pd_=np.sqrt(pd2)
        # ewc drift
        Fk=ewc_list[k-1].fisher
        ed=sum((Fk[n].cpu()*(θ_t[n]-θ_k[n])**2).sum().item() for n in θ_k)
        # forgetting
        da=results[k][f"task_{k}"]-results[t][f"task_{k}"]
        records.append({"from":k,"to":t,"pd":pd_,"ed":ed,"da":da})

dash=pd.DataFrame(records)
dash["from_to"]=dash["from"].astype(str)+"→"+dash["to"].astype(str)
print(dash[["from_to","pd","ed","da"]])
print("Corr(pd vs da):",dash.pd.corr(dash.da))
print("Corr(ed vs da):",dash.ed.corr(dash.da))
