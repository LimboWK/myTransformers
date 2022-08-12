import torch 

def set_torch_device():
    print('mps for Mac backend built:', torch.backends.mps.is_built())
    print('mps available:', torch.backends.mps.is_available())
    print('cuda available:', torch.cuda.is_available())
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        # apple device
        device = torch.device("mps")
    elif  torch.cuda.is_available():
        device = torch.device("cuda:0") 
    else:
        device = 'cpu'
    print("*** set torch device to:", device, "***")
    return device
