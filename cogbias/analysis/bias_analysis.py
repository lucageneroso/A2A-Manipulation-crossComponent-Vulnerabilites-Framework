import torch
from sklearn.decomposition import PCA
import numpy as np
from typing import List, Dict, Any, Tuple

class BiasExtractor:
    """
    Estrae vettori semantici confrontando distribuzioni di hidden states tra prompt neutrali e orientati al bias.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device

    def extract_mean_difference(self, neutral_states: List[torch.Tensor], biased_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Calcola il Mean Difference Vector: v_bias = mu_biased - mu_neutral
        """
        neutrals = torch.stack(neutral_states).mean(dim=0)
        biased = torch.stack(biased_states).mean(dim=0)
        return (biased - neutrals)
        
    def extract_contrastive_pca(self, neutral_states: List[torch.Tensor], biased_states: List[torch.Tensor], n_components: int = 3) -> torch.Tensor:
        """
        Calcola le componenti principali della varianza differenziale per isolare il segnale semantico pulito.
        """
        # PCA requires 2D arrays (samples, features)
        X_neutral = torch.stack(neutral_states).to(torch.float32).cpu().numpy()
        X_biased = torch.stack(biased_states).to(torch.float32).cpu().numpy()
        
        diffs = X_biased - X_neutral
        # reshape if it's (N, seq_len, hidden_size) or similar? 
        # For M6, we typically compare the final token embedding or mean pooling.
        # Assume inputs are already pooled (N, hidden_size)
        
        if diffs.ndim > 2:
            diffs = diffs.reshape(diffs.shape[0], -1)
            
        pca = PCA(n_components=n_components)
        pca.fit(diffs)
        
        # Returns the principal components as a torch tensor (components, features)
        return torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
