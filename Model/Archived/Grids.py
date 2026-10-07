import os
import json
import torch

def build_Vi_grids(v_max_list, device):
    #TO BE UPDATED
    """
    Create discretized value grids with step size 1, from 0 to v̄i (included).
    v_max_list: list of v̄i, used to determine # of discretize.
    Returns list of m tensors size d1*d2*...dm]
    """
    m = len(v_max_list)
    value_vectors = []
    for i in range(m):
        vmax_int = int(float(v_max_list[i]))
        vec = torch.arange(0, vmax_int + 1/m, step = 1/m, device=device, dtype=torch.float32)
        #discretizeing size, di, is not defined, so I use default step size = 1
        #discretized values from 0 to vmax_int, including vmax_int
        value_vectors.append(vec)
    #value vectors is list of length m, with each element being Vi
    V = torch.meshgrid(value_vectors, indexing='ij')
    #meshing Vi_grids is product of all Vi, gives V, aka all possible value profiles
    return V

def save_grid_dataset_offline(path, Vi_grids, buyer_pmf, meta=None):
    """
    Saves the discretized value grids and PMF to disk.
    """
    os.makedirs(path, exist_ok=True)
    torch.save({
        'Vi_grids': [vg.cpu() for vg in Vi_grids],
        'buyer_pmf': buyer_pmf.cpu()
    }, os.path.join(path, 'dataset.pt'))
    if meta is not None:
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

def load_grid_dataset_offline(path, device):
    """
    Loads Vi_grids and buyer_pmf from disk.
    """
    data_path = os.path.join(path, 'dataset.pt')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No offline dataset found at {data_path}. Run the prepare step first.")
    
    data = torch.load(data_path, map_location=device)
    Vi_grids = [vg.to(device).float() for vg in data['Vi_grids']]
    buyer_pmf = data['buyer_pmf'].to(device).float()
    meta = None

    meta_path = os.path.join(path, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    return Vi_grids, buyer_pmf, meta



#Maps buyer's strategy(distribution over all menu items)for each value profile v
def Buyer_Strategy(V):
    buyer_pmf = torch.ones_like(V[0])
    
    #sets starting point as uniform distribution
    for i in range(len(V)):
        vmax = V[i].amax()
        #finds largest value in each grid
        buyer_pmf = buyer_pmf * (V[i] / vmax.clamp_min(1e-12))
        #multipies the current buyer_pmf by the normalized grid values
    
    #for loop finds the probability of each possible value profile
    buyer_pmf = buyer_pmf / buyer_pmf.sum()
    #normalizes the entire pmf to sum to 1
    return buyer_pmf

def ensure_offline_grid_dataset(path, v_max_list, device):
    """
    Ensures dataset exists at 'path'. If not, builds and saves it.
    """
    if not os.path.exists(os.path.join(path, 'dataset.pt')):
        # if no dataset, build data based on given parameters
        Vi_grids = build_Vi_grids(v_max_list, device)
        buyer_pmf = Buyer_Strategy(Vi_grids)
        meta = {'v_max_list': v_max_list}
        save_grid_dataset_offline(path, Vi_grids, buyer_pmf, meta=meta)
    return load_grid_dataset_offline(path, device)


