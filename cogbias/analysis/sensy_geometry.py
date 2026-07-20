import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Any
import torch.nn.functional as F

class SensyGeometryAnalyzer:
    def __init__(self, device="cpu"):
        self.device = device
        
    def pca_manifold_analysis(self, X: np.ndarray, labels: np.ndarray):
        """
        Measure explained variance, effective dimensionality.
        """
        pca = PCA()
        pca.fit(X)
        
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        eff_dim = np.argmax(cum_var >= 0.95) + 1
        
        return {
            "explained_variance_ratio": pca.explained_variance_ratio_[:10].tolist(),
            "effective_dimensionality_95": int(eff_dim),
            "residual_variance": float(1.0 - cum_var[eff_dim-1]) if eff_dim > 0 else 1.0
        }

    def cluster_analysis(self, X: np.ndarray, labels: np.ndarray, unique_classes: List[Any]):
        """
        Compare sensitive vs non-sensitive, and separately each thematic category.
        """
        if len(set(labels)) > 1:
            sil_score = float(silhouette_score(X, labels, metric='cosine'))
        else:
            sil_score = 0.0
            
        centroids = {}
        for cls in unique_classes:
            idx = (labels == cls)
            if idx.sum() > 0:
                centroids[cls] = X[idx].mean(axis=0)
                
        # Inter-cluster distances
        inter_dist = {}
        cls_list = list(centroids.keys())
        for i in range(len(cls_list)):
            for j in range(i+1, len(cls_list)):
                c1, c2 = cls_list[i], cls_list[j]
                # Cosine distance
                cos_sim = np.dot(centroids[c1], centroids[c2]) / (np.linalg.norm(centroids[c1]) * np.linalg.norm(centroids[c2]))
                inter_dist[f"{c1}_vs_{c2}"] = float(1.0 - cos_sim)
                
        return {
            "silhouette_score": sil_score,
            "inter_cluster_cosine_distance": inter_dist
        }

    def boundary_analysis(self, X: np.ndarray, y: np.ndarray):
        """
        Train probes for classification.
        """
        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv_folds = min(5, len(X) // 2)
        if cv_folds >= 2:
            scores = cross_val_score(clf, X, y, cv=cv_folds)
            acc = float(np.mean(scores))
        else:
            clf.fit(X, y)
            acc = float(clf.score(X, y))
            
        return {"accuracy": acc}

    def latent_field_mapping(self, mu_sens: torch.Tensor, mu_nonsens: torch.Tensor, v_authority: torch.Tensor):
        """
        Compare SENSY geometry with existing Authority vector.
        """
        v_sens = mu_sens - mu_nonsens
        
        # Normalize
        v_sens_norm = F.normalize(v_sens, p=2, dim=0)
        v_auth_norm = F.normalize(v_authority.to(v_sens.device), p=2, dim=0)
        
        cos_sim = F.cosine_similarity(v_sens_norm.unsqueeze(0), v_auth_norm.unsqueeze(0)).item()
        
        return {
            "sensitivity_vector_norm": float(torch.linalg.norm(v_sens).item()),
            "cosine_similarity_with_authority": cos_sim
        }

    def null_space_analysis(self, X: np.ndarray):
        """
        Estimate active dimensions and unused dimensions via eigenvalue spectrum.
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # Sort descending
        eigenvalues = eigenvalues[::-1]
        
        # Threshold for "unused" dimensions (e.g., eigenvalue < 1e-5 max_eigenvalue)
        threshold = eigenvalues[0] * 1e-5
        active_dims = int(np.sum(eigenvalues > threshold))
        unused_dims = len(eigenvalues) - active_dims
        
        return {
            "total_dimensions": len(eigenvalues),
            "active_dimensions": active_dims,
            "unused_dimensions": unused_dims
        }
