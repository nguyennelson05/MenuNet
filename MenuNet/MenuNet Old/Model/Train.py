import os
import torch
try:
    from . import MenuNet_New as MN
except ImportError:
    import MenuNet_New as MN


params = [[1,1],    #vi_bar_list,
        2,          #m (items),
        3           #k (menu options)
        ]
model_name = f"[0,{params[0][0]}] x [0,{params[0][1]}]"

#train model
def train_model(device):
    vi_bar_list, m, k = params
    num_steps = 5000
    menu = MN.MenuNet(m=m, k=k, vi_bar_list=vi_bar_list, device=device).to(device)
    opt = torch.optim.Adam(menu.parameters(), lr=0.01)

    model = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model, exist_ok=True)
    final_path = os.path.join(model, f'{model_name}.pt')

    if os.path.exists(final_path):
        print(f"Model already exists at {final_path}, loading existing model.")
        load_model_state(final_path, menu, device)
        menu.eval()
        with torch.no_grad():
            best_rev, best_rev_rational = (menu.revenue())
            print(f"Loaded model revenue: {best_rev.item()}")
    else:
        print("No existing model found, starting training from scratch.")
        best_rev = float('-inf')

    for step in range(num_steps+1):
        menu.train()
        rev, rev_rational = menu.revenue()
        loss = -rev

        opt.zero_grad()
        loss.backward()
        opt.step()

        if rev.item() > best_rev:
            best_rev = rev.item()
            save_model_state(final_path, menu)
        
        if step % (num_steps/10) == 0:
            print(f"Step {step}: Revenue = {rev.item():.6f}")
        
    save_model_state(final_path, menu)
    with torch.no_grad():
        rev, rev_rational = menu.revenue()
        print(f"Loaded model revenue: {rev.item():.5f}")
        print(f"Rational rev: {rev_rational.item():.5f}")
    return menu, final_path



#save & loading models
def load_trained_model(parameters, path, device):
    vi_bar_list, m, k = parameters
    menu = MN.MenuNet(m=m, k=k, vi_bar_list=vi_bar_list, device=device).to(device)
    load_model_state(path, menu, device)
    menu.eval()
    with torch.no_grad():
        rev, rev_rational = menu.revenue()
        print(f"  Loaded model revenue: {rev.item():.5f}")
        print(f"  Rational rev: {rev_rational.item():.5f}")
    return menu

def save_model_state(path, menu):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(menu.state_dict(), path)

def load_model_state(path, menu, device):
    state = torch.load(path, map_location=device)
    menu.load_state_dict(state)



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(device)

