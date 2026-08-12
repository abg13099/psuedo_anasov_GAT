import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import softmax
from community import community_louvain as co_louvain
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import ZINC 
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import SpectralClustering
from networkx.algorithms import community
from scipy.linalg import eigh
from itertools import product

def extract_largest_connected_component(data):
    G_nx = to_networkx(data, to_undirected=True)

    if nx.is_connected(G_nx):
        print("Graph is connected. Proceeding with the full graph.")
        return data

    print(f"Graph with {data.num_nodes} nodes is disconnected. Extracting the largest connected component.")
    connected_components = list(nx.connected_components(G_nx))
    largest_component_nodes = max(connected_components, key=len)

    if len(largest_component_nodes) == data.num_nodes:
        print("Largest connected component contains all nodes. No change needed.")
        return data

    # Create a mapping from original node IDs to new contiguous node IDs (0 to N-1 for LCC)
    old_node_ids_in_lcc = sorted(list(largest_component_nodes))
    old_to_new_node_map = {old_id: new_id for new_id, old_id in enumerate(old_node_ids_in_lcc)}

    # Filter `data` attributes to keep only nodes in the largest connected component
    new_x = data.x[old_node_ids_in_lcc]
    new_y = data.y[old_node_ids_in_lcc]

    # Filter masks: Only keep mask entries for nodes in LCC, and re-index them
    new_train_mask = data.train_mask[old_node_ids_in_lcc]
    new_val_mask = data.val_mask[old_node_ids_in_lcc]
    new_test_mask = data.test_mask[old_node_ids_in_lcc]

    # Filter and re-index edge_index
    # Only keep edges where both source and target are in the LCC
    edge_index_list = data.edge_index.t().tolist() # Convert to list of [u, v] pairs
    new_edge_index_list = []
    for u_orig, v_orig in edge_index_list:
        if u_orig in largest_component_nodes and v_orig in largest_component_nodes:
            new_edge_index_list.append([old_to_new_node_map[u_orig], old_to_new_node_map[v_orig]])
    new_edge_index = torch.tensor(new_edge_index_list, dtype=torch.long).t().contiguous()

    # Create a new Data object for the largest connected component
    new_data = torch_geometric.data.Data(x=new_x, edge_index=new_edge_index, y=new_y)
    new_data.train_mask = new_train_mask
    new_data.val_mask = new_val_mask
    new_data.test_mask = new_test_mask
    new_data.num_nodes = len(old_node_ids_in_lcc) # Explicitly set num_nodes

    print(f"Reduced graph to largest connected component with {new_data.num_nodes} nodes.")
    return new_data

# ------------------- Build block operator D -------------------
def build_block_operator_D(G_nx, communities, alpha=0.2, base_anisotropy_c=1, beta=0.01):
    """
    Build block operator D with anisotropic intra-community diffusion
    AND Option 1 modulation for inter-community diffusion.
    """

    n = G_nx.number_of_nodes()
    nodes_list = list(G_nx.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes_list)}

    D = np.zeros((n, n), dtype=float)

    # Storage for Option 1
    comm_us = {}          # cid -> unstable eigenvector (block-local indexing)
    comm_u_maxabs = {}    # cid -> max |u|
    comm_lambda_u = {}    # cid -> lambda_unstable_rescaled
    node_pos_in_block = {}   # node -> (cid, local_idx)

    # ----------- Build each community block ------------
    for cid, block in enumerate(communities):
        nodes = list(block)
        idx = np.array([node_to_idx[v] for v in nodes], dtype=int)

        sub = G_nx.subgraph(nodes)
        L_block = nx.laplacian_matrix(sub, nodelist=nodes).toarray().astype(float)

        nb = L_block.shape[0]
        if nb == 0:
            continue

        if nb == 1:
            # 1-node block: no eigenvectors
            D_block = np.eye(1)
            u = np.array([[1.0]])     # dummy
            lam_unstable_rescaled = 1.0
        else:
            # ---- Compute spectrum ----
            vals, vecs = eigh(L_block)
            vals = np.real(vals)
            vecs = np.real(vecs)

            sort_idx = np.argsort(vals)
            vals = vals[sort_idx]
            vecs = vecs[:, sort_idx]

            # stable = second smallest eigenvector
            # unstable = largest eigenvector
            s = vecs[:, 1].reshape(-1, 1)
            u = vecs[:, -1].reshape(-1, 1)

            # (You can replace these by rescaling again if desired)
            lam_stable_rescaled = vals[1]
            lam_unstable_rescaled = vals[-1]

            # ---- Adaptive anisotropic scale ----
            #    Smaller in dense blocks, larger in sparse blocks
            if lam_unstable_rescaled < 1e-9:
                anisotropic_scale = 0.0
            else:
                anisotropic_scale = base_anisotropy_c / lam_unstable_rescaled


            # ---- Build anisotropic intra-community diffusion ----
            P = anisotropic_scale * (lam_unstable_rescaled * (u @ u.T) -
                                     lam_stable_rescaled * (s @ s.T))
            D_block = np.eye(nb) - alpha * L_block + P

        # ---- Store unstable direction info for Option 1 ----
        comm_us[cid] = u.flatten()
        comm_u_maxabs[cid] = np.max(np.abs(comm_us[cid])) if np.max(np.abs(comm_us[cid])) > 0 else 1.0
        comm_lambda_u[cid] = lam_unstable_rescaled

        for local_idx, node in enumerate(nodes):
            node_pos_in_block[node] = (cid, local_idx)

        # ---- Insert block into D ----
        for ii, vi in enumerate(idx):
            for jj, vj in enumerate(idx):
                D[vi, vj] = D_block[ii, jj]

    # ----------------- Inter-community edges (Option 1) -----------------

    # First compute Z (global max alignment factor)
    Z = 1.0
    max_num = 0.0
    for u_node, v_node in G_nx.edges():
        cid_u, pos_u = node_pos_in_block[u_node]
        cid_v, pos_v = node_pos_in_block[v_node]
        if cid_u == cid_v:
            continue

        # normalized |u|
        val_u = abs(comm_us[cid_u][pos_u]) / comm_u_maxabs[cid_u]
        val_v = abs(comm_us[cid_v][pos_v]) / comm_u_maxabs[cid_v]

        numerator = val_u * val_v * 0.5 * (comm_lambda_u[cid_u] + comm_lambda_u[cid_v])
        max_num = max(max_num, numerator)

    if max_num > 0:
        Z = max_num  # ensures w <= beta approximately

    # Now add weighted cross-community diffusion
    for u_node, v_node in G_nx.edges():
        cid_u, pos_u = node_pos_in_block[u_node]
        cid_v, pos_v = node_pos_in_block[v_node]

        if cid_u == cid_v:
            continue

        val_u = abs(comm_us[cid_u][pos_u]) / comm_u_maxabs[cid_u]
        val_v = abs(comm_us[cid_v][pos_v]) / comm_u_maxabs[cid_v]

        numerator = val_u * val_v * 0.5 * (comm_lambda_u[cid_u] + comm_lambda_u[cid_v])
        w = beta * (numerator / Z)

        iu = node_to_idx[u_node]
        iv = node_to_idx[v_node]
        D[iu, iv] += w
        D[iv, iu] += w   # symmetric

    return D, node_to_idx, nodes_list


# ------------------- Attach weights from D -------------------
def attach_weights_from_D(data, D, node_to_idx, nodes_list, clip_negative=True, add_self_loop=True):
    W = 0.5 * (D + D.T)
    if clip_negative:
        W[W < 0] = 0.0

    edge_index = data.edge_index.cpu().numpy()
    num_edges = edge_index.shape[1]
    edge_weights = np.zeros(num_edges, dtype=float)

    for eidx in range(num_edges):
        u = int(edge_index[0, eidx])
        v = int(edge_index[1, eidx])
        iu, iv = node_to_idx[u], node_to_idx[v]
        edge_weights[eidx] = W[iu, iv]

    data.edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    if add_self_loop:
        u_list, v_list = data.edge_index[0].tolist(), data.edge_index[1].tolist()
        w_list = data.edge_weight.tolist()
        diag = np.diag(W)
        for node in range(data.num_nodes):
            exists = False
            for j, (uu, vv) in enumerate(zip(u_list, v_list)):
                if uu == node and vv == node:
                    w_list[j] = diag[node]
                    exists = True
                    break
            if not exists:
                u_list.append(node)
                v_list.append(node)
                w_list.append(float(diag[node]))
        data.edge_index = torch.tensor([u_list, v_list], dtype=torch.long)
        data.edge_weight = torch.tensor(w_list, dtype=torch.float)

    return data

def compute_diffused_laplacian_weights(data,
                                       alpha=0.1,
                                       base_anisotropy_c=1,
                                       beta=0.01,
                                       soft_positive=True):
    """
    Precompute anisotropic diffusion-derived edge weights for the entire graph data.
    Returns modified data and number of communities.
    """
    G_nx = to_networkx(data, to_undirected=True)
    partition = co_louvain.best_partition(G_nx, random_state=42)
    # Choose number of communities
    #k = 10

    # Returns an iterator of sets
    #comp = community.asyn_fluidc(G_nx, k=k)

    # Convert to partition dict
    #partition = {}
    #for comm_id, node_set in enumerate(comp):
    #    for node in node_set:
    #        partition[node] = comm_id

    # convert partition dict -> list of node sets
    communities_dict = {}
    for node, comm_id in partition.items():
        communities_dict.setdefault(comm_id, set()).add(node)
    communities_sets = list(communities_dict.values())
    num_communities = len(communities_sets)

    D, node_to_idx, nodes_list = build_block_operator_D(G_nx, communities_sets,
                                                       alpha=alpha,
                                                       base_anisotropy_c=base_anisotropy_c,
                                                       beta=beta)
    data = attach_weights_from_D(data, D, node_to_idx, nodes_list)
    return data, num_communities


def compute_structural_features(data):
    G = to_networkx(data, to_undirected = True)
    nodes = list(G.nodes())
    degrees = [G.degree(n) for n in nodes]
    clustering = list(nx.clustering(G).values())
    features = torch.tensor(
        [[d, c] for d, c in zip(degrees, clustering)],
        dtype=torch.float
    )
    data.x = features
    return data

def prepare_tu_dataset(name, root='data/TU', diffusion_params=None):

    dataset = TUDataset(root=root, name=name, use_node_attr=True)

    num_classes = dataset.num_classes
    num_features = dataset.num_features

    #print(f"First graph x: {dataset[0].x}")
    #print(f"First graph y: {dataset[0].y}")

    #Sometimes data is empty
    dataset = [data for data in dataset if data.x is not None and data.y is not None]

    if diffusion_params:
        processed=[]
        for data in dataset:
            try:
                data,_ = compute_diffused_laplacian_weights(data, **diffusion_params)
            except Exception as e:
                print(f"Error when computing diffused laplacian: {e}")
            processed.append(data)
        dataset = processed
    return dataset, num_classes, num_features 

def prepare_zinc_dataset(name, root='data/ZINC', subset=True,diffusion_params=None):
    train_dataset = ZINC(root=root, subset=subset, split='train')
    val_dataset = ZINC(root=root, subset=subset, split='val')
    test_dataset = ZINC(root=root, subset=subset, split='test')

    num_classes = 1 
    num_features = train_dataset.num_features

    full_dataset = list(train_dataset) + list(val_dataset) + list(test_dataset)

    train_end = len(train_dataset)
    val_end = train_end + len(val_dataset)

    train_indices = torch.arange(0, train_end)
    val_indices = torch.arange(train_end, val_end)
    test_indices = torch.arange(val_end, len(full_dataset))

    processed = []
    for data in full_dataset:
        if data.x is not None and data.y is not None:
            if diffusion_params:
                try:
                    data, _ = compute_diffused_laplacian_weights(data, **diffusion_params)
                except Exception as e:
                    print(f"Error when computing diffused laplacian: {e}")
            processed.append(data)
    
    splits = ([train_indices], [val_indices], [test_indices])

    return processed, num_classes, num_features, splits

def prepare_ogb_dataset(name, root='data/OGB', diffusion_params=None):
    dataset = PygGraphPropPredDataset(name=name, root=root)

    split_idx = dataset.get_idx_split()
    train_indices = split_idx["train"]
    val_indices = split_idx["val"]
    test_indices = test_idx["test"]

    num_classes = dataset.num_classes
    num_features = dataset.num_features
    processed = []

    for data in dataset:
        if hasattr(data, 'x') == False or data.x is None:
            data = compute_structural_features(data)
            num_features = data.x.shape[1]

        if data.y is not None:
            if data.y.dim() > 1:
                data.y = data.y.squeeze()

            if diffusion_params: 
                try:
                    data, _ = compute_diffused_laplacian_weights(data, **diffusion_params)
                except Exception as e:
                    print(f"Error when computing diffused laplacian: {e}")
            processed.append(data)

    splits = ([train_indices], [val_indices], [test_indices])

    return processed, num_classes, num_features, splits
