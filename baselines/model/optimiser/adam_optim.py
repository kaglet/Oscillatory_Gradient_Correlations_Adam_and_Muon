import torch 


optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)