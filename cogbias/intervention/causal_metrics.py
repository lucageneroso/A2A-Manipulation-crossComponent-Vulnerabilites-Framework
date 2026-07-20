import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

class CausalMetrics:
    @staticmethod
    def kl_divergence(logits_baseline, logits_intervention):
        """
        Calculate KL divergence between baseline and intervention token distributions.
        logits: shape (batch_size, seq_len, vocab_size)
        """
        # We need logits on the same device
        if isinstance(logits_baseline, torch.Tensor):
            logits_baseline = logits_baseline.float().cpu()
        if isinstance(logits_intervention, torch.Tensor):
            logits_intervention = logits_intervention.float().cpu()
            
        p = F.softmax(logits_baseline, dim=-1)
        log_p = F.log_softmax(logits_baseline, dim=-1)
        log_q = F.log_softmax(logits_intervention, dim=-1)
        
        # KL(P || Q) = sum(P * (log_p - log_q))
        kl = torch.sum(p * (log_p - log_q), dim=-1)
        # Average over sequence length and batch
        return kl.mean().item()

    @staticmethod
    def representation_alignment(hidden_state, direction):
        """
        Calculate cosine similarity between a hidden state and a direction.
        hidden_state: shape (batch_size, seq_len, hidden_dim) or (hidden_dim,)
        direction: shape (hidden_dim,)
        """
        if isinstance(hidden_state, torch.Tensor):
            h = hidden_state.detach().float().cpu().numpy()
        else:
            h = hidden_state
            
        if isinstance(direction, torch.Tensor):
            d = direction.detach().float().cpu().numpy()
        else:
            d = direction
            
        h_flat = h.reshape(-1, h.shape[-1])
        h_mean = h_flat.mean(axis=0)
        
        # normalize direction just in case
        d_norm = d / (np.linalg.norm(d) + 1e-8)
        cos_sim = np.dot(h_mean, d_norm) / (np.linalg.norm(h_mean) + 1e-8)
        return float(cos_sim)

class AuthorityStyleClassifier:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False
        
    def fit(self, X_auth, X_neut, X_exp=None):
        """
        Train a binary classifier: 1 for Authority, 0 for Neutral/Expertise.
        X_auth, X_neut: shape (n_samples, hidden_dim)
        """
        X_list = [X_auth, X_neut]
        y_list = [np.ones(len(X_auth)), np.zeros(len(X_neut))]
        
        if X_exp is not None:
            X_list.append(X_exp)
            y_list.append(np.zeros(len(X_exp)))
            
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        self.clf.fit(X, y)
        self.is_trained = True
        
    def score(self, X):
        """
        Return the probability of the Authority class.
        X: shape (n_samples, seq_len, hidden_dim) or (n_samples, hidden_dim)
        """
        if not self.is_trained:
            raise ValueError("Classifier is not trained yet")
            
        if isinstance(X, torch.Tensor):
            X = X.detach().float().cpu().numpy()
            
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        elif len(X.shape) == 3:
            X = X.mean(axis=1) # average over seq_len
            
        probs = self.clf.predict_proba(X)
        return probs[:, 1].tolist()
