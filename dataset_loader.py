import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import softmax
from community import community_louvain as co_louvain
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import ZINC
from torch_geometric.datasets import Planetoid, WebKB, Coauthor, Actor, HeterophilousGraphDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import SpectralClustering
from networkx.algorithms import community
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
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

    # D is built as sparse triplets rather than a dense n x n array: for
    # large single-graph node classification datasets (tens of thousands of
    # nodes) a dense matrix would need several GB, while the actual nonzero
    # entries are confined to per-community blocks plus inter-community edges.
    D_rows, D_cols, D_vals = [], [], []

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
                D_rows.append(vi)
                D_cols.append(vj)
                D_vals.append(D_block[ii, jj])

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
        D_rows.append(iu); D_cols.append(iv); D_vals.append(w)
        D_rows.append(iv); D_cols.append(iu); D_vals.append(w)   # symmetric

    D = coo_matrix((D_vals, (D_rows, D_cols)), shape=(n, n)).tocsr()
    return D, node_to_idx, nodes_list


# ------------------- Attach weights from D -------------------
def attach_weights_from_D(data, D, node_to_idx, nodes_list, clip_negative=True, add_self_loop=True):
    W = (D + D.T).tocsr()
    W.data *= 0.5
    if clip_negative:
        W.data[W.data < 0] = 0.0
        W.eliminate_zeros()

    edge_index = data.edge_index.cpu().numpy()
    # node_to_idx maps original node ids -> position in the (community-detection)
    # adjacency; idx_map lets us go from a node id straight to that position.
    idx_map = np.array([node_to_idx[node] for node in range(data.num_nodes)])
    iu = idx_map[edge_index[0]]
    iv = idx_map[edge_index[1]]
    edge_weights = np.asarray(W[iu, iv]).flatten()

    data.edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    if add_self_loop:
        diag = np.asarray(W.diagonal()).flatten()[idx_map]

        u_arr, v_arr = edge_index[0], edge_index[1]
        is_self_loop = (u_arr == v_arr)
        # position of each node's existing self-loop edge (if any), else -1
        self_loop_pos = -np.ones(data.num_nodes, dtype=np.int64)
        self_loop_pos[u_arr[is_self_loop]] = np.nonzero(is_self_loop)[0]

        u_list, v_list, w_list = u_arr.tolist(), v_arr.tolist(), edge_weights.tolist()
        for node in range(data.num_nodes):
            pos = self_loop_pos[node]
            if pos >= 0:
                w_list[pos] = float(diag[node])
            else:
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

def prepare_graph_dataset(name, diffusion_params=None):
    """Routes a graph classification/regression dataset name to its loader.

    `root` only controls where the raw/processed graph structure is cached on
    disk (independent of diffusion_params, which are applied in-memory after
    loading), so unweighted and diffusion-weighted calls intentionally share
    the same weighted/unweighted root regardless of caller (a fixed grid run
    or a diffusion-parameter search trial).

    Returns (dataset, num_classes, num_features, splits, generate_splits, regression).
    `splits` is None when generate_splits is True (caller should build it via k_fold).
    """
    weighted = diffusion_params is not None
    if name == 'ZINC':
        root = f'data/ZINC/{"weighted" if weighted else "unweighted"}'
        dataset, num_classes, num_features, splits = prepare_zinc_dataset(root=root, name=name, diffusion_params=diffusion_params)
        return dataset, num_classes, num_features, splits, False, True
    elif "ogbg" in name:
        root = f'data/OBG/{"weighted" if weighted else "unweighted"}'
        dataset, num_classes, num_features, splits = prepare_ogb_dataset(root=root, name=name, diffusion_params=diffusion_params)
        return dataset, num_classes, num_features, splits, False, False
    else:
        root = f'data/TU/{"weighted" if weighted else "unweighted"}'
        dataset, num_classes, num_features = prepare_tu_dataset(root=root, name=name, diffusion_params=diffusion_params)
        return dataset, num_classes, num_features, None, True, False

NODE_DATASET_FAMILY = {
    "Cora": "Planetoid", "CiteSeer": "Planetoid", "PubMed": "Planetoid",
    "Cornell": "WebKB", "Texas": "WebKB", "Wisconsin": "WebKB",
    "CS": "Coauthor", "Physics": "Coauthor",
    "Actor": "Actor",
    "Roman-empire": "Heterophilous", "Amazon-ratings": "Heterophilous",
    "Minesweeper": "Heterophilous", "Tolokers": "Heterophilous", "Questions": "Heterophilous",
}

def random_node_split(num_nodes, seed=42, train_ratio=0.6, val_ratio=0.2):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)
    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True
    return train_mask, val_mask, test_mask

def prepare_node_dataset(name, root='data/Node', diffusion_params=None, split_idx=0):
    family = NODE_DATASET_FAMILY.get(name)
    if family == "Planetoid":
        dataset = Planetoid(root=root, name=name)
    elif family == "WebKB":
        dataset = WebKB(root=root, name=name)
    elif family == "Coauthor":
        dataset = Coauthor(root=root, name=name)
    elif family == "Actor":
        dataset = Actor(root=root)
    elif family == "Heterophilous":
        dataset = HeterophilousGraphDataset(root=root, name=name)
    else:
        raise ValueError(f"Unknown node classification dataset {name}")

    data = dataset[0]
    num_classes = dataset.num_classes
    num_features = dataset.num_features

    if not hasattr(data, 'train_mask') or data.train_mask is None:
        # Coauthor ships no split; carve out a random 60/20/20 split.
        data.train_mask, data.val_mask, data.test_mask = random_node_split(data.num_nodes)
    elif data.train_mask.dim() == 2:
        # WebKB / Actor ship 10 geom-gcn splits as columns; pick one.
        data.train_mask = data.train_mask[:, split_idx]
        data.val_mask = data.val_mask[:, split_idx]
        data.test_mask = data.test_mask[:, split_idx]

    if diffusion_params:
        try:
            data, _ = compute_diffused_laplacian_weights(data, **diffusion_params)
        except Exception as e:
            print(f"Error when computing diffused laplacian: {e}")

    return data, num_classes, num_features

def prepare_ogb_dataset(name, root='data/OGB', diffusion_params=None):
    dataset = PygGraphPropPredDataset(name=name, root=root)

    split_idx = dataset.get_idx_split()
    train_indices = split_idx["train"]
    val_indices = split_idx["val"]
    test_indices = split_idx["test"]

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
