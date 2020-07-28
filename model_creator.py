import os
from network import ReCoNet
import torch

ALL_MODEL_NAMES = ["style25_train_20200722213134.pt"]
def get_all_models(device):
    print("Getting all models...")
    result = []
    for model_name in ALL_MODEL_NAMES:
        model = ReCoNet()
        model.load_state_dict(
            torch.load(os.path.join("trained_models", model_name)) #, map_location="cpu")
        )
        model=model.to(device)
        result.append(model)
    return result