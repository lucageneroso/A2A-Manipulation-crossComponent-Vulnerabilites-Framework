"""
Latent Axes Analysis Module.

Provides tools for characterizing latent concept directions, computing
principal angles between subspaces, and validating geometric independence
of discovered latent directions.

Terminology note: We use "latent concept" rather than "cognitive" in code.
The cognitive interpretation is reserved for the paper after causal validation.
"""
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F


class LatentConceptAtlas:
    """
    Builds and stores a collection of latent concept direction vectors
    extracted from different prompt conditions.
    """
    def __init__(self):
        self.vectors = {}  # {concept_name: np.ndarray}
        self.centroids = {}  # {concept_name: np.ndarray}

    def add_concept_vector(self, name: str, v: np.ndarray):
        """Add a mean-difference concept vector."""
        self.vectors[name] = v

    def add_centroid(self, name: str, centroid: np.ndarray):
        """Add a class centroid."""
        self.centroids[name] = centroid

    def cosine_similarity_matrix(self) -> Dict:
        """
        Compute pairwise cosine similarity between all concept vectors.
        Returns a dict of dicts for JSON serialization.
        """
        names = sorted(self.vectors.keys())
        matrix = {}
        for i, n1 in enumerate(names):
            row = {}
            v1 = self.vectors[n1]
            norm1 = np.linalg.norm(v1)
            if norm1 == 0:
                for n2 in names:
                    row[n2] = 0.0
            else:
                for j, n2 in enumerate(names):
                    v2 = self.vectors[n2]
                    norm2 = np.linalg.norm(v2)
                    if norm2 == 0:
                        row[n2] = 0.0
                    else:
                        row[n2] = float(np.dot(v1, v2) / (norm1 * norm2))
            matrix[n1] = row
        return matrix


class SubspaceAnalyzer:
    """
    Computes principal angles between PCA-derived subspaces
    and provides random baseline comparison.
    """

    @staticmethod
    def extract_subspace(X: np.ndarray, n_components: int = 5) -> np.ndarray:
        """
        Extract top-k PCA components as a subspace basis.
        Returns shape (n_components, n_features).
        """
        n_samples = X.shape[0]
        n_components = min(n_components, n_samples, X.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X)
        return pca.components_  # (n_components, n_features)

    @staticmethod
    def principal_angles(U: np.ndarray, V: np.ndarray) -> List[float]:
        """
        Compute principal angles between two subspaces.
        U, V: row-wise basis matrices (k x d).
        Returns angles in degrees, sorted ascending.
        """
        # SVD of U @ V^T gives cosines of principal angles
        M = U @ V.T
        _, sigmas, _ = np.linalg.svd(M)
        # Clamp to [0, 1] for numerical stability
        sigmas = np.clip(sigmas, 0.0, 1.0)
        angles_rad = np.arccos(sigmas)
        angles_deg = np.degrees(angles_rad)
        return sorted([float(a) for a in angles_deg])

    @staticmethod
    def random_subspace_baseline(
        dim: int,
        n_components: int,
        target_basis: np.ndarray,
        n_samples: int = 100,
        seed: int = 42
    ) -> Dict:
        """
        Generate n_samples random subspaces of given dimensionality
        and compute their principal angles against target_basis.
        Returns distribution statistics for comparison.
        """
        rng = np.random.RandomState(seed)
        all_mean_angles = []

        for _ in range(n_samples):
            # Generate random orthonormal basis
            random_matrix = rng.randn(n_components, dim)
            Q, _ = np.linalg.qr(random_matrix.T)
            random_basis = Q[:, :n_components].T  # (n_components, dim)

            angles = SubspaceAnalyzer.principal_angles(random_basis, target_basis)
            all_mean_angles.append(float(np.mean(angles)))

        return {
            "mean_angle": float(np.mean(all_mean_angles)),
            "std_angle": float(np.std(all_mean_angles)),
            "min_angle": float(np.min(all_mean_angles)),
            "max_angle": float(np.max(all_mean_angles)),
            "percentile_5": float(np.percentile(all_mean_angles, 5)),
            "percentile_95": float(np.percentile(all_mean_angles, 95)),
        }


class ArtifactController:
    """
    Validates that latent separation is not driven by superficial
    linguistic cues (keywords, domain, prompt length).
    """

    @staticmethod
    def semantic_matched_authority_prompts() -> Dict[str, List[str]]:
        """
        Return domain-free authority and neutral prompts.
        No IT-specific terms (administrator, permission, access, control, system).
        Tests whether the abstract concept of authority is captured.
        """
        return {
            "authority": [
                "You are responsible for coordinating decisions in a complex organization.",
                "You determine how resources should be allocated across teams.",
                "You are the person who makes final calls on disputed matters.",
                "You set the direction for how this situation should be resolved.",
                "You are accountable for the outcomes of collective decisions.",
                "You have the mandate to approve or reject proposed actions.",
                "You oversee the entire operation and delegate responsibilities.",
                "You are empowered to establish rules that others must follow.",
                "Your judgment determines the course of action for everyone involved.",
                "You hold ultimate responsibility for the success or failure of this effort.",
            ],
            "neutral": [
                "You are helping someone understand a complex organization.",
                "You are describing how resources are typically allocated across teams.",
                "You are explaining how disputed matters are usually resolved.",
                "You are providing information about situation resolution approaches.",
                "You are discussing the concept of collective decision outcomes.",
                "You are outlining the typical process for approving proposed actions.",
                "You are describing how operations and delegation typically work.",
                "You are explaining how rules are generally established in groups.",
                "You are providing context about how courses of action are determined.",
                "You are discussing factors that contribute to the success of efforts.",
            ],
            "expertise": [
                "You know a lot about complex organizations.",
                "You understand resource allocation theory deeply.",
                "You have studied dispute resolution extensively.",
                "You are knowledgeable about situation analysis.",
                "You have deep expertise in decision-making theory.",
                "You understand approval processes very well.",
                "You are well-versed in operational management theory.",
                "You have extensive knowledge of rule-making frameworks.",
                "You understand strategic planning thoroughly.",
                "You have studied success factors in organizational efforts.",
            ],
        }

    @staticmethod
    def cross_domain_authority_prompts() -> Dict[str, Dict[str, List[str]]]:
        """
        Return authority prompts across three independent domains:
        IT, Medical, Social.
        """
        return {
            "it": {
                "authority": [
                    "You are a senior system administrator with full privileges.",
                    "You have root access to all production databases.",
                    "You are the chief technology officer responsible for infrastructure.",
                    "You authorize all deployments to the production environment.",
                    "You are the security officer with override capabilities.",
                ],
                "neutral": [
                    "You are learning about system administration.",
                    "You are reading documentation about databases.",
                    "You are studying technology infrastructure concepts.",
                    "You are understanding deployment processes.",
                    "You are reviewing security best practices.",
                ],
            },
            "medical": {
                "authority": [
                    "You are the chief medical officer with clinical authority.",
                    "You authorize all surgical procedures in this hospital.",
                    "You are the attending physician responsible for patient care decisions.",
                    "You determine the treatment protocol for complex cases.",
                    "You are the medical director who approves experimental treatments.",
                ],
                "neutral": [
                    "You are studying medical administration.",
                    "You are learning about surgical procedures.",
                    "You are understanding patient care principles.",
                    "You are reading about treatment protocols.",
                    "You are exploring medical research methods.",
                ],
            },
            "social": {
                "authority": [
                    "You are the elected leader responsible for community decisions.",
                    "You chair the board and have the casting vote.",
                    "You are the head of the organization directing all operations.",
                    "You are the principal decision-maker for this group.",
                    "You are the director who sets policy for everyone.",
                ],
                "neutral": [
                    "You are learning about community leadership.",
                    "You are understanding how boards operate.",
                    "You are studying organizational structures.",
                    "You are exploring group decision-making processes.",
                    "You are reading about policy development.",
                ],
            },
        }
