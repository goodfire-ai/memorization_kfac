#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
perplexity.py

Perplexity computation function copied directly from neuron_utils.py
"""

import torch


def perplexity(dataloader, model):
    """
    Compute perplexity over a dataloader.
    
    Copied directly from:
    replicate_bsn/memorization/src/localize/neuron/neuron_utils.py
    
    Args:
        dataloader: PyTorch DataLoader containing token sequences
        model: Language model with .device attribute and ability to compute loss
    
    Returns:
        float: Average perplexity over the dataloader
    """
    avg_metric = 0
    for batch in dataloader:
        batch = batch.to(model.device)
        with torch.no_grad():
            model_output = model(batch, labels=batch)
        loss = model_output.loss
        avg_metric += torch.exp(loss)
    return avg_metric.cpu().item() / len(dataloader)