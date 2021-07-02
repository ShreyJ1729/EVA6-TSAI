import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def tensor_to_image(x_tensor):
    image = x_tensor.squeeze().permute(1, 2, 0).numpy()
    image = (image - np.min(image)) / (image.max() - image.min())
    image = (image * 255).astype("uint8")

    plt.imshow(image)
