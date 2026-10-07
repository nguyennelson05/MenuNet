import os
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from . import Grids as BM
except ImportError:
    import Grids as BM



# Model definitions
class MechNet(nn.Module):
    """
    MechNet inputs 1 and outputs :
        - Allocation matrix X, alloc are in [0,1] and size m * k
        - Payment vector p, size k 
    The last column of both are 0 (exit option).
    """
    def __init__(self, m, k):
        super().__init__()
        self.m = m
        self.k = k
        # parameters for the first k-1 menu items; last is exit (zeros).
        self.X_params = nn.Parameter(torch.randn(m, k - 1) * 0.1)  # allocations
        self.p_params = nn.Parameter(torch.randn(k - 1) * 0.1)     # payments

    def forward(self):
        # Sigmoid to keep allocations in [0,1]
        X_learn = torch.sigmoid(self.X_params)
        p_learn = self.p_params                 
        # Add zeros back for exit option
        device = X_learn.device
        X = torch.cat([X_learn, torch.zeros(self.m, 1, device=device)], dim=1) 
        p = torch.cat([p_learn, torch.zeros(1, device=device)], dim=0)       
        return X, p

class BuyerNet(nn.Module):
    """
    Buyer network 
        - inputs: (X, p) from MechNet
        - Output: choice distribution S of shape [d1,...,dm,k]
        - Utility for menu item j = sum(vi * X[i, j]) - p[j]
    """

    def __init__(self, Vi_grids, temperature=0.1):
        super().__init__()
        #print(Vi_grids)
        self.script_V = torch.stack(Vi_grids, dim=-1)
        # stacks grid to create list of tensors representing all possible valuation profiles
        self.register_buffer('V', self.script_V)          
        # makes V a non-parameter (not learned) b/c utility known (additive)
        self.temperature = float(temperature)

    def forward(self, X, p):
        """
        Compute choice distribution S given mechanism (X, p).
        X: [m, k]
        p: [k]
        Returns S: [d1*...dm*k]
        """
        #print("\n SCRIPT V",self.script_V)
        Sum_ScriptXi = torch.einsum('...m,mk->...k', self.script_V, X)  
        #print("\n SUM SCRIPT XI",Sum_ScriptXi)
        # represents valuations for all menu items
        U = Sum_ScriptXi - p
        # subtracts payment vector from valuation to get util tensor
        S = F.softmax(U / self.temperature, dim=-1).clamp_min(1e-12)
        # represents softmax on k for util tensor, aggregation of buyer strategy
        # normalizes utilities per value profile (high util = highest prob)
        return S


class MenuNet(nn.Module):
    """
    Parent class that holds both MechNet and BuyerNet, returns
      - expected_revenue: returns expected payment * buyer pmf (calculated in grids)
      - loss: negative revenue (to maximize)
    """
    def __init__(self, m=0, k=0, Vi_grids=None, buyer_pmf=None, temperature=0.05):
        super().__init__()
        self.mech = MechNet(m, k)
        self.buyer = BuyerNet(Vi_grids, temperature)
        # Store buyer PMF on the parent MenuNet (optional)
        if buyer_pmf is not None:
            self.register_buffer('buyer_pmf', buyer_pmf)
        else:
            self.buyer_pmf = None

    def forward(self, v=None):
        X, p = self.mech()
        if v is None:
            S = self.buyer(X, p)
            return X, p, S
        else:            
            U = v @ X - p
            probs = F.softmax(U / self.buyer.temperature, dim=-1).clamp_min(1e-12)
            j_star = probs.argmax(dim=-1)
            alloc = X[:, j_star]
            price = p[j_star]
            return alloc, price

    def expected_revenue(self, S, p, buyer_pmf=None):

        # Determine buyer PMF: prefer provided arg, then buyer buffer on the buyer
        if buyer_pmf is None:
            buyer_pmf = getattr(self, 'buyer_pmf', None)
            if buyer_pmf is None:
                raise ValueError("buyer_pmf not provided and not set in MenuNet.")

        expected_payment = (S * p).sum(dim=-1)
        revenue = (expected_payment * buyer_pmf).sum()
        return revenue

    def loss(self, S=None, p=None, buyer_pmf=None):
        rev = self.expected_revenue(S=S, p=p, buyer_pmf=buyer_pmf)
        return -rev



# Saving/loading utilities
def save_full_model_state(path, menu):
    """
    Save the full model state_dict (parame & buffers).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(menu.state_dict(), path)

def load_full_model_state(path, menu, device):
    """
    Load the full model state_dict into MenuNet instance.
    """
    state = torch.load(path, map_location=device)
    menu.load_state_dict(state)



# Example usage / training
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #store dataset and models inside MenuNet
    base_dir = os.path.dirname(__file__)

    #put model files into a models subfolder
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    curr_best_path = os.path.join(model_dir, 'menu_curr_best.pt')
    final_path = os.path.join(model_dir, 'menu_final.pt')

    menu, cfg = setup(base_dir, device)
    opt = torch.optim.Adam(menu.mech.parameters(), lr=1e-2)

    # If a pre-existing model exists, load it (resume from checkpoint)
    if os.path.isfile(curr_best_path):
        print(f"Found existing model at {curr_best_path}. Loading and resuming training.")
        load_full_model_state(curr_best_path, menu, device)
        menu.eval()
        with torch.no_grad():
            X0, p0, S0 = menu()
            best_rev = menu.expected_revenue(S=S0, p=p0).item()
        print(f"[resume] Starting best_rev from loaded model: {best_rev:.6f}")
    else:
        print("No existing model found. Training from scratch.")
        best_rev = float('-inf')

    # Training loop (save best and final full model state_dict)
    num_steps = 500
    for step in range(num_steps+1):
        menu.train()
        X, p, S = menu()
        rev = menu.expected_revenue(S=S, p=p)
        loss = -rev

        opt.zero_grad()
        loss.backward()
        opt.step()

        if rev.item() > best_rev:
            best_rev = rev.item()
            save_full_model_state(curr_best_path, menu,)

        if step % 100 == 0:
            print(f"Step {step:02d}: Revenue={rev.item():.5f} (best={best_rev:.5f})")

    # Save final model at the end
    save_full_model_state(final_path, menu)
    return cfg, curr_best_path, device




# Load and evaluate example
def evaluate_loaded_model(train):
    # Recreate model (with the same dataset) and load best full model
    cfg, curr_best_path, device = train
    m, k, temperature, Vi_grids, buyer_pmf = cfg
    menu_loaded = MenuNet(m=m, k=k, Vi_grids=Vi_grids, buyer_pmf=buyer_pmf, temperature=temperature).to(device)
    load_full_model_state(curr_best_path, menu_loaded, device)

    menu_loaded.eval()
    with torch.no_grad():
        Xl, pl, Sl = menu_loaded()
        rev_loaded = menu_loaded.expected_revenue(S=Sl, p=pl)
    print(f"[eval] Loaded full model revenue: {rev_loaded.item():.6f}")


def load_trained_model(model_path = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #store dataset and models inside MenuNet
    base_dir = os.path.dirname(__file__)
    menu, cfg = setup(base_dir, device)
    load_full_model_state(model_path, menu, device)
    menu.eval()
    return menu


def setup(base_dir, device='cpu'):
    #store dataset and models inside MenuNet
    dataset_path = os.path.join(base_dir, 'offline_fullgrid')
    v_max_list = [1, 1]

    #Ensure offline dataset
    Vi_grids, buyer_pmf, meta = BM.ensure_offline_grid_dataset(dataset_path, v_max_list, device)
    if meta is not None:
        if meta.get('v_max_list') != v_max_list:
            print("Loaded dataset v_max_list does not match current config.")

    # Model settings
    k = 3
    m = 2
    temperature = 0.05
    menu = MenuNet(m=m, k=k, Vi_grids=Vi_grids, buyer_pmf=buyer_pmf, temperature=temperature).to(device)
    cfg = (m, k, temperature, Vi_grids, buyer_pmf)
    return menu, cfg




def main():
    evaluate_loaded_model(train())

if __name__ == "__main__":
    main()