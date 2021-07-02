import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activation_maps = None
        self.target_layer = target_layer

        self.target_layer.register_forward_hook(self.store_activation_maps)
        self.target_layer.register_full_backward_hook(self.store_gradients)

    def store_gradients(self, module, inp, out):
        self.gradients = out

    def store_activation_maps(self, module, inp, out):
        self.activation_maps = out

    def get_activation_map_weights(self, gradients):
        return torch.mean(gradients[0], dim=[2, 3]).unsqueeze(2).unsqueeze(3)

    def generate_heatmap(self, x):
        self.model.eval()
        pred = self.model(x)
        pred_idx = pred.argmax(dim=1).item()
        pred[:, pred_idx].backward()
        activation_weights = self.get_activation_map_weights(self.gradients)
        weighted_activation = F.relu(activation_weights * self.activation_maps)
        heatmap = (
            torch.mean(weighted_activation, dim=1).detach().cpu().squeeze(0).numpy()
        )

        return heatmap

    def resize_heatmap(self, image, new_height, new_width):
        return cv2.resize(image, dsize=(new_width, new_height))

    def prepare_heatmap_for_overlay(self, heatmap):
        heatmap = (heatmap - np.min(heatmap)) / (heatmap.max() - heatmap.min())
        heatmap = (heatmap * 255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        return heatmap

    def prepare_image_for_overlay(self, image):
        image = image.cpu().squeeze().permute(1, 2, 0).numpy()
        image = (image - np.min(image)) / (image.max() - image.min())
        image = (image * 255).astype("uint8")

        return image

    def overlay_image_and_heatmap(self, image, heatmap, alpha=0.5):
        return cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0, dtype=cv2.CV_8U)

    def __call__(self, x):
        heatmap = self.generate_heatmap(x.unsqueeze(0))
        heatmap = self.resize_heatmap(
            heatmap, new_height=x.shape[1], new_width=x.shape[2]
        )
        heatmap = self.prepare_heatmap_for_overlay(heatmap)
        image = self.prepare_image_for_overlay(x)
        output = self.overlay_image_and_heatmap(image, heatmap)

        return output
