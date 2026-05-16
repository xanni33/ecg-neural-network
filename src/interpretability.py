#do saliency map -> jak kazda probka wplynela na predykcje klasy
import torch
import numpy as np

def compute_saliency_map(model, signal, target_class):

    model.eval()

    signal = signal.clone().detach()

    signal.requires_grad = True

    output = model(signal)

    score = output[0, target_class]

    model.zero_grad()

    score.backward()

    saliency = signal.grad.data.abs().cpu().numpy()

    saliency = saliency[0, 0]

    return saliency