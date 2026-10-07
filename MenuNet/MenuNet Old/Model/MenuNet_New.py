import torch
import torch.nn as nn
import torch.nn.functional as F



class MechNet(nn.Module):
    """
    Takes no input, outputs allocation matrix X and payment vector p.
    last columns are both 0 (exit)
    m = number of items, k = number of menu options (including exit)
    X has shape [m, k], p has shape [k]
    """
    def __init__(self, m, k):
        super().__init__()
        self.m = m
        self.k = k
        #learn on params, k-1 to exclude exit
        self.X_params = nn.Parameter(torch.randn(m, k - 1) * 0.1)  # allocations
        self.p_params = nn.Parameter(torch.randn(k - 1) * 0.1)     # payments

    def forward(self):
        X_learn = torch.sigmoid(self.X_params)
        p_learn = F.softplus(self.p_params)
        # Add zeros back for exit option
        device = X_learn.device
        X = torch.cat([X_learn, torch.zeros(self.m, 1, device=device)], dim=1) 
        p = torch.cat([p_learn, torch.zeros(1, device=device)], dim=0)       
        return X, p
    


class BuyerNet(nn.Module):
    """
    Takes allocation matrix X and payment vector p as input,
    outputs choice distribution S over all menu items for each value profile v.
    shape of output S: [d1,...,dm,k]
    Utility for menu item j = sum(vi * X[i, j]) - p[j]
    vi = value profile for item i, X[i, j] = allocation of item i for menu item j, p[j] = payment for menu item j
    """
    def __init__(self):
        super().__init__()

    """
    Stacks Vis to convert tensors to list of all possible value profiles,
    (ex:2 3x3 tensors -> 9 sets of value profiles (v1,v2) for all possible valuations of each item)
    Muliplies each value profile with allocation matrix X to get valuation for each menu item.
    Sum Script Xis to get total valuation for each menu item, subtract p to get utility.
    Softmax Utility on last dim to get choice distribution S.
    Sum_ScriptXi, U, and S have shape [d1,...,dm,k] (in this case 3x3x3)
    S_det is choice dist using argmax U(v, j): deterministic, rational choice
    """
    def forward(self, X, p, V):
        Script_Vi = torch.stack(V, dim=-1)
        #print("Script_Vi:", Script_Vi)
        Sum_Script_Xi = torch.einsum('...m,mk->...k', Script_Vi, X)
        #print(Sum_ScriptXi)
        U = Sum_Script_Xi - p
        S = F.softmax(U/.05, dim=-1)
        j_star = U.argmax(dim=-1)
        S_det = F.one_hot(j_star, num_classes=p.shape[0]).to(torch.float32)
        #print(S.shape)
        return S, S_det

    """
    inputs list of all vi_bar to discretized di times; Vi is set of all vi (set from 0 to vi_bar with di steps)
    note: we set di to be the same for all i for simplicity
    use these functions to find all possible value profiles V by gettign product of all Vi
    for this case, di = m, so step size is 1/m; value vectors is list of Vi with length m
    returns V, a tensor of length m containg all script_Vis;
    each script_Vi is tensor of all possible value profiles for item i, size d1*d2*...dm (in this case di^m since di = dj for all i,j)
    """
    def build_V(self, vi_bar_list, device):
        d = 100 #number of discretizations per item
        value_vectors = []
        for vi_bar in vi_bar_list:
            Vi = torch.linspace(0, vi_bar, steps=d, device=device)
            value_vectors.append(Vi)
        V = torch.meshgrid(value_vectors, indexing='ij')
        return V #scriptV

    """
    inputs V, list of all possible value profiles
    returns Pr[v], the probability distribution over all value profiles
    for uniform distribution, its just 1/(d1*d2*...dm) for all v
    """
    def Probability_Distribution(self, V):
        uniform_dist = torch.ones_like(V[0])
        uniform_dist = uniform_dist / uniform_dist.sum()
        return uniform_dist



class MenuNet(nn.Module):
    """
    Overarching class, holds both MechNet and BuyerNet with 2 main functions:
     - forward:  given a valuation profile v from a bidder, outputs allocation and price
     - loss: Calculate expected revenue and loss (negative expected revenue)
    expected revenue is summation of buyer_pmf * payment vector * choice distribution S
    """
    def __init__(self, m, k, vi_bar_list, device):
        super().__init__()
        self.mech = MechNet(m, k)
        self.buyer = BuyerNet()
        self.vi_bar_list = vi_bar_list
        self.device = device

    def forward(self, v):
        X, p = self.mech()
        U = v @ X - p
        j_star = U.argmax(dim=-1)
        alloc = X[:, j_star].tolist()
        price = float(p[j_star])
        return alloc, price
    
    def revenue(self):
        X, p = self.mech()
        V = self.buyer.build_V(self.vi_bar_list, self.device)
        uniform_dist = self.buyer.Probability_Distribution(V)

        S, S_det = self.buyer(X, p, V)
        expected_payment = torch.einsum('...k,k->...', S, p)
        expected_revenue = (expected_payment * uniform_dist).sum()
        expected_payment = torch.einsum('...k,k->...', S_det, p)
        expected_revenue_rational = (expected_payment * uniform_dist).sum()

        return expected_revenue, expected_revenue_rational
    
