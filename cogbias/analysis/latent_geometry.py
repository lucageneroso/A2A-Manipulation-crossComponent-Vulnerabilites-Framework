import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import json

class LatentGeometryAnalyzer:
    def __init__(self, device="cpu"):
        self.device = device
        
    def perform_pca_analysis(self, embeddings_dict):
        """
        embeddings_dict: { class_name: [tensor, tensor, ...] }
        Returns PCA explained variance and effective dimensionality.
        """
        all_vecs = []
        labels = []
        for c_name, tensors in embeddings_dict.items():
            for t in tensors:
                all_vecs.append(t.to(torch.float32).cpu().numpy().flatten())
                labels.append(c_name)
                
        X = np.stack(all_vecs)
        
        pca = PCA()
        pca.fit(X)
        
        # Effective dimensionality (number of components needed for 95% variance)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        eff_dim = np.argmax(cum_var >= 0.95) + 1
        
        return {
            "explained_variance_ratio": pca.explained_variance_ratio_[:10].tolist(),
            "effective_dimensionality_95": int(eff_dim),
            "residual_variance": float(1.0 - cum_var[eff_dim-1]) if eff_dim > 0 else 1.0
        }

    def perform_cluster_analysis(self, embeddings_dict):
        """
        Computes intra/inter class distances and silhouette score.
        """
        all_vecs = []
        labels = []
        for c_idx, (c_name, tensors) in enumerate(embeddings_dict.items()):
            for t in tensors:
                all_vecs.append(t.to(torch.float32).cpu().numpy().flatten())
                labels.append(c_idx)
                
        X = np.stack(all_vecs)
        
        if len(set(labels)) > 1:
            sil_score = float(silhouette_score(X, labels, metric='cosine'))
        else:
            sil_score = 0.0
            
        return {
            "silhouette_score_cosine": sil_score
        }
        
    def perform_boundary_analysis(self, embeddings_dict, class_pairs):
        """
        Trains Logistic Regression probes for linear separability.
        class_pairs: list of tuples like ("neutral", "authority")
        """
        results = {}
        for c1, c2 in class_pairs:
            if c1 not in embeddings_dict or c2 not in embeddings_dict:
                continue
                
            X_list = []
            y_list = []
            
            for t in embeddings_dict[c1]:
                X_list.append(t.to(torch.float32).cpu().numpy().flatten())
                y_list.append(0)
            for t in embeddings_dict[c2]:
                X_list.append(t.to(torch.float32).cpu().numpy().flatten())
                y_list.append(1)
                
            X = np.stack(X_list)
            y = np.array(y_list)
            
            clf = LogisticRegression(max_iter=1000, random_state=42)
            # Use cross-validation (leave-one-out if small dataset, or 3-fold)
            cv_folds = min(3, len(X) // 2)
            if cv_folds >= 2:
                scores = cross_val_score(clf, X, y, cv=cv_folds)
                acc = float(np.mean(scores))
            else:
                clf.fit(X, y)
                acc = float(clf.score(X, y))
                
            results[f"{c1}_vs_{c2}"] = {"accuracy": acc}
            
        return results
