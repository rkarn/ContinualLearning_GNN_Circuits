import os
import re
import networkx as nx
import pandas as pd
from networkx.exception import NetworkXNoPath

def parse_verilog_netlist(verilog_file):
    G = nx.DiGraph()
    gate_types = ['and','or','nand','nor','xor','xnor','buf','not']
    gates = {}
    all_signals = set()
    output_signals = set()
    input_signals = set()

    with open(verilog_file,'r') as f:
        lines = f.readlines()

    gate_pattern = re.compile(rf'\s*({"|".join(gate_types)})\s+(\w+)\s*\((.*)\);')
    for i,line in enumerate(lines):
        if i%1000==0:
            print(f"Parsing line {i}/{len(lines)}")
        m = gate_pattern.match(line.strip())
        if not m: 
            continue
        gate_type, gate_name, conn = m.groups()
        sigs = [s.strip() for s in conn.split(',')]
        out = sigs[0]
        ins = sigs[1:]
        input_signals.update(ins)
        output_signals.add(out)
        all_signals.update(sigs)
        gates[out] = (gate_name, gate_type, ins)
        G.add_node(gate_name, type=gate_type)

    # identify PIs/POs
    primary_inputs  = input_signals - output_signals
    primary_outputs = output_signals - input_signals

    # wire up edges
    for out,(gname,_,ins) in gates.items():
        for inp in ins:
            if inp in gates:
                G.add_edge(gates[inp][0], gname)
            elif inp in primary_inputs:
                # treat PI as a node
                G.add_node(inp, type='input')
                G.add_edge(inp, gname)

    # ensure all POs get a node
    for po in primary_outputs:
        G.add_node(po, type='output')
        if po in gates:
            G.add_edge(gates[po][0], po)

    return G, gates, primary_inputs, primary_outputs

def extract_features(G, gates, primary_inputs, primary_outputs, circuit_id):
    """
    circuit_id: base filename without .v
    """
    import networkx as nx
    features = []
    deg_cent  = nx.degree_centrality(G)
    btw_cent  = nx.betweenness_centrality(G, k=min(100,len(G)))
    clo_cent  = nx.closeness_centrality(G)
    clust_coef= nx.clustering(G)

    for node in G.nodes:
        node_type = G.nodes[node]['type']
        is_pi      = int(node in primary_inputs)
        is_po      = int(node in primary_outputs)
        is_int     = int(not (is_pi or is_po))
        fan_in     = G.in_degree(node)
        fan_out    = G.out_degree(node)
        nbrs       = list(G.neighbors(node))
        avg_fi_nbr = sum(G.in_degree(n) for n in nbrs)/len(nbrs) if nbrs else 0
        avg_fo_nbr = sum(G.out_degree(n) for n in nbrs)/len(nbrs) if nbrs else 0

        # depth (distance from any PI)
        try:
            if is_pi:
                depth = 0
            else:
                depths = [nx.shortest_path_length(G, source=pi, target=node)
                          for pi in primary_inputs if nx.has_path(G, pi, node)]
                depth = min(depths) if depths else -1
        except (NetworkXNoPath, KeyError):
            depth = -1

        # dist to any PO
        try:
            if is_po:
                dist_out = 0
            else:
                ds = [nx.shortest_path_length(G, source=node, target=po)
                      for po in primary_outputs if nx.has_path(G, node, po)]
                dist_out = min(ds) if ds else -1
        except (NetworkXNoPath, ValueError):
            dist_out = -1

        is_key = int(node_type in ['xor','xnor'])
        features.append({
            'circuit_id': circuit_id,          # <-- new column
            'node': node,
            'gate_type': node_type,
            'fan_in': fan_in,
            'fan_out': fan_out,
            'depth': depth,
            'dist_to_output': dist_out,
            'is_primary_input': is_pi,
            'is_primary_output': is_po,
            'is_internal': is_int,
            'is_key_gate': is_key,
            'degree_centrality': deg_cent.get(node,0),
            'betweenness_centrality': btw_cent.get(node,0),
            'closeness_centrality': clo_cent.get(node,0),
            'clustering_coefficient': clust_coef.get(node,0),
            'avg_fan_in_neighbors': avg_fi_nbr,
            'avg_fan_out_neighbors': avg_fo_nbr
        })
    return pd.DataFrame(features)


def process_all_netlists(folder_path):
    all_feats = []
    for fname in os.listdir(folder_path):
        if not fname.endswith('.v'):
            continue
        path = os.path.join(folder_path, fname)
        print(f"→ processing {fname}")
        circuit_id = os.path.splitext(fname)[0]     # strip “.v”
        try:
            G, gates, pis, pos = parse_verilog_netlist(path)
            feats = extract_features(G, gates, pis, pos, circuit_id)
            all_feats.append(feats)
        except Exception as e:
            print(f"⚠️ {fname} skipped: {e}")

    if all_feats:
        df_all = pd.concat(all_feats, ignore_index=True)
        df_all.to_csv("all_circuits_features_graphclass.csv", index=False)
        print("✅ saved all_circuits_features.csv")
    else:
        print("❌ no features extracted.")

if __name__=="__main__":
    process_all_netlists("verilog_benchmark_circuits-master")


import pandas as pd

# Load the extracted features CSV file
df = pd.read_csv("all_circuits_features_graphclass.csv")

# Display the first few rows
print("\n🔹 First 5 rows of the DataFrame:")
print(df.head())

# Summary statistics
print("\n🔹 DataFrame Summary:")
print(df.info())

print("\n🔹 Statistical Summary:")
print(df.describe())

# Check for missing values
print("\n🔹 Missing Values in Each Column:")
print(df.isnull().sum())

# Count the number of circuits processed
num_circuits = df['circuit_id'].nunique()
num_nodes = df.shape[0]
print(f"\n✅ Number of circuits processed: {num_circuits}")
print(f"✅ Total number of nodes (gates + inputs/outputs): {num_nodes}")

# Count of different gate types
print("\n🔹 Gate Type Distribution:")
print(df['gate_type'].value_counts())

# Check the number of key gates detected
num_key_gates = df['is_key_gate'].sum()
print(f"\n🔑 Number of detected key gates: {num_key_gates}")
