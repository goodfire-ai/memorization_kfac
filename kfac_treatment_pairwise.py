import heapq
import torch
from typing import List, Tuple, Dict, Union
#from kfac_treatment import KFACTreatment

"""
K-FAC (Kronecker-Factored Approximate Curvature) treatment for weight compression.

This implementation uses variance-retained ranking: for a given variance ratio,
it selects the smallest number of eigenvectors whose cumulative eigenvalue mass
exceeds the target ratio.
"""

import torch
from typing import Dict, List, Optional, Union, Tuple


class KFACTreatment:
    """
    Apply K-FAC compression to specified linear layers in a model.
    
    This class uses K-FAC factors (activation and gradient covariances) to project
    weights onto top eigenspaces for memorization reduction.
    
    Note on dimensions:
    - For gate/up projections: W has shape [11008, 4096] = [out_features, in_features]
    - For down projection: W has shape [4096, 11008] = [out_features, in_features]
    - G always corresponds to output dimension (gradient covariance)
    - A always corresponds to input dimension (activation covariance)
    """
    
    def __init__(self, model, layer_names: List[str], kfac_factors_path: str, 
                 device: Optional[str] = None, keep_eigenvectors_on_cpu: bool = False):
        """
        Initialize K-FAC treatment.
        
        Args:
            model: The PyTorch model to apply treatment to
            layer_names: List of layer names to compress (e.g., ['model.layers.31.mlp.up_proj'])
            kfac_factors_path: Path to the saved K-FAC factors file
            device: Device to use for computations (defaults to model device)
            keep_eigenvectors_on_cpu: If True, store eigenvectors on CPU to save GPU memory
        """
        self.model = model
        self.layer_names = layer_names
        self.device = device or next(model.parameters()).device
        self.keep_eigenvectors_on_cpu = keep_eigenvectors_on_cpu
        
        # Load K-FAC factors
        self.kfac_data = torch.load(kfac_factors_path, map_location='cpu')
        
        # Store original weights
        self.original_weights = {}
        self._store_original_weights()
        
        # Prepare K-FAC info for each layer
        self.kfac_info = {}
        self._prepare_kfac_info()
        
        # Track compression stats
        self.compression_stats = {}
        
    def _store_original_weights(self):
        """Store original weights of specified layers."""
        with torch.no_grad():
            for layer_name in self.layer_names:
                layer = self._get_layer_by_name(layer_name)
                if isinstance(layer, torch.nn.Linear):
                    # Detach and clone to avoid any autograd metadata
                    self.original_weights[layer_name] = layer.weight.data.detach().clone()
                else:
                    raise ValueError(f"Layer {layer_name} is not a Linear layer")
    
    def _get_layer_by_name(self, layer_name: str):
        """Get layer by dot-separated name."""
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            layer = getattr(layer, part)
        return layer
    
    def _get_kfac_key(self, layer_name: str) -> str:
        """Convert layer name to K-FAC data key format."""
        # Example: 'model.layers.31.mlp.up_proj' -> 'blk31.up'
        parts = layer_name.split('.')
        
        # Extract block number and projection type
        block_num = None
        proj_type = None
        
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                block_num = parts[i + 1]
            elif part in ['gate_proj', 'up_proj', 'down_proj']:
                proj_type = part.replace('_proj', '')
        
        if block_num is not None and proj_type is not None:
            return f'blk{block_num}.{proj_type}'
        else:
            # Try to find exact match in kfac_data keys
            for key in self.kfac_data.keys():
                if proj_type and proj_type in key:
                    return key
            raise ValueError(f"Could not determine K-FAC key for layer {layer_name}")
    
    def _prepare_kfac_info(self):
        """Prepare K-FAC eigenvalues and eigenvectors for each layer."""
        for layer_name in self.layer_names:
            kfac_key = self._get_kfac_key(layer_name)
            
            if kfac_key not in self.kfac_data:
                raise ValueError(f"K-FAC factors not found for {kfac_key} (layer: {layer_name})")
            
            kfac_layer_data = self.kfac_data[kfac_key]
            W_orig = self.original_weights[layer_name]
            
            # Prepare eigendecomposition
            info = self._prepare_layer_kfac_info(kfac_layer_data, W_orig, layer_name)
            self.kfac_info[layer_name] = info
    
    def _prepare_layer_kfac_info(self, kfac_layer_data: Dict, W_orig: torch.Tensor, 
                                 layer_name: str) -> Dict:
        """Prepare K-FAC info with eigenvalues and eigenvectors for a single layer."""
        # Work on computation device initially
        device = self.device if not self.keep_eigenvectors_on_cpu else W_orig.device
        
        A = kfac_layer_data['A'].float().to(device)
        G = kfac_layer_data['G'].float().to(device)
        
        # Compute eigenvalues and eigenvectors
        eva_A, evc_A = torch.linalg.eigh(A)
        eva_G, evc_G = torch.linalg.eigh(G)
        
        # Sort in descending order
        idx_A = eva_A.argsort(descending=True)
        eva_A = eva_A[idx_A]
        evc_A = evc_A[:, idx_A]
        
        idx_G = eva_G.argsort(descending=True)
        eva_G = eva_G[idx_G]
        evc_G = evc_G[:, idx_G]
        
        # Verify dimensions match weight matrix
        out_features, in_features = W_orig.shape
        assert evc_G.shape[0] == out_features, f"G eigenvectors shape {evc_G.shape} doesn't match output dimension {out_features}"
        assert evc_A.shape[0] == in_features, f"A eigenvectors shape {evc_A.shape} doesn't match input dimension {in_features}"
        
        # Optionally move to CPU for storage
        if self.keep_eigenvectors_on_cpu:
            eva_A = eva_A.cpu()
            evc_A = evc_A.cpu()
            eva_G = eva_G.cpu()
            evc_G = evc_G.cpu()
        
        # Simplified output - only show shapes
        print(f"{layer_name}: W{tuple(W_orig.shape)}, G{tuple(G.shape)}, A{tuple(A.shape)}")
        
        return {
            'W_orig': W_orig,
            'eva_A': eva_A,
            'evc_A': evc_A,
            'eva_G': eva_G,
            'evc_G': evc_G
        }
    
    def _compute_rank_for_variance(self, evals: torch.Tensor, target_variance_ratio: float) -> Tuple[int, float]:
        """
        Compute the number of components needed to explain target_variance_ratio of variance.
        
        This uses the "variance retained" approach: selects the smallest k such that
        the cumulative sum of the first k eigenvalues divided by the total sum
        is >= target_variance_ratio.
        
        Args:
            evals: Eigenvalues in descending order
            target_variance_ratio: Fraction of variance to retain (0 to 1)
            
        Returns:
            (k, actual_variance): Number of components and actual variance retained
        """
        total_variance = evals.sum()
        cumsum_variance = torch.cumsum(evals, dim=0)
        variance_ratios = cumsum_variance / total_variance
        
        # Handle edge case where we want 100% variance
        if target_variance_ratio >= 1.0:
            return len(evals), 1.0
        
        # Find first index where cumulative variance >= target
        mask = variance_ratios >= target_variance_ratio
        if mask.any():
            rA = mask.nonzero()[0].item() + 1
            actual_variance = variance_ratios[rA-1].item()
        else:
            # If no eigenvalue satisfies the condition, return all
            rA = len(evals)
            actual_variance = 1.0
        
        return rA, actual_variance
    
    def _project_weight(self, info: Dict, rG: int, rA: int) -> torch.Tensor:
        """
        Project weight matrix W to rank (rG, rA) subspace using K-FAC eigenvectors.
        
        Formula: W_proj = U_G[:, :rG] @ (U_G[:, :rG].T @ W @ U_A[:, :rA]) @ U_A[:, :rA].T
        
        Where:
        - U_G are the top rG eigenvectors of the gradient covariance (output dimension)
        - U_A are the top rA eigenvectors of the activation covariance (input dimension)
        """
        # Get eigenvectors, moving to computation device if needed
        device = info["W_orig"].device
        Ug = info["evc_G"][:, :rG]
        Ua = info["evc_A"][:, :rA]
        
        if Ug.device != device:
            Ug = Ug.to(device)
        if Ua.device != device:
            Ua = Ua.to(device)
            
        W = info["W_orig"].float()
        
        # Compute projection
        W_proj = Ug @ (Ug.T @ W @ Ua) @ Ua.T
        
        return W_proj
    
    def apply_kfac(self, variance_ratios: Union[Tuple[float, float], Dict[str, Tuple[float, float]]]):
        """
        Apply K-FAC compression to specified layers.
        
        Args:
            variance_ratios: Either a tuple (activation_variance, gradient_variance) for all layers,
                           or a dict mapping layer names to their specific variance ratios.
                           Values should be between 0 and 1, representing the fraction of 
                           variance to retain (not the rank ratio).
        """
        if isinstance(variance_ratios, tuple):
            ratios_dict = {name: variance_ratios for name in self.layer_names}
        else:
            ratios_dict = variance_ratios
        
        with torch.no_grad():  # Ensure no autograd tracking
            for layer_name in self.layer_names:
                if layer_name not in ratios_dict:
                    print(f"Warning: No variance ratios specified for {layer_name}, skipping")
                    continue
                
                var_A, var_G = ratios_dict[layer_name]
                info = self.kfac_info[layer_name]
                
                # Move eigenvalues to computation device if needed
                eva_A = info['eva_A']
                eva_G = info['eva_G']
                if eva_A.device != self.device:
                    eva_A = eva_A.to(self.device)
                if eva_G.device != self.device:
                    eva_G = eva_G.to(self.device)
                
                # Compute ranks based on variance retained
                rA, actual_var_A = self._compute_rank_for_variance(eva_A, var_A)
                rG, actual_var_G = self._compute_rank_for_variance(eva_G, var_G)
                
                # Apply K-FAC compression
                W_compressed = self._project_weight(info, rG, rA)
                
                # Update layer weights using copy_
                layer = self._get_layer_by_name(layer_name)
                layer.weight.copy_(W_compressed.to(self.original_weights[layer_name].dtype))
                
                # Store stats
                total_params = rA * rG
                original_params = info['W_orig'].numel()
                self.compression_stats[layer_name] = {
                    'variance_ratios': (var_A, var_G),
                    'actual_variances': (actual_var_A, actual_var_G),
                    'ranks': (rA, rG),
                    'total_params': total_params,
                    'original_params': original_params,
                    'compression_ratio': total_params / original_params
                }
                
                print(f"Applied K-FAC to {layer_name}:")
                print(f"  rA = {rA} (retains {actual_var_A:.1%} variance)")
                print(f"  rG = {rG} (retains {actual_var_G:.1%} variance)")
                print(f"  Total parameters: {total_params:,} / {original_params:,} ({total_params/original_params:.1%})")
    
    def restore_original_weights(self):
        """Restore all layers to their original weights."""
        with torch.no_grad():
            for layer_name, original_weight in self.original_weights.items():
                layer = self._get_layer_by_name(layer_name)
                layer.weight.copy_(original_weight)
        
        # Clear compression stats
        self.compression_stats = {}
        print("Restored original weights for all layers")
    
    def restore_layer(self, layer_name: str):
        """Restore a specific layer to its original weights."""
        if layer_name not in self.original_weights:
            raise ValueError(f"No original weights stored for {layer_name}")
        
        with torch.no_grad():
            layer = self._get_layer_by_name(layer_name)
            layer.weight.copy_(self.original_weights[layer_name])
        
        # Remove from compression stats
        if layer_name in self.compression_stats:
            del self.compression_stats[layer_name]
        
        print(f"Restored original weights for {layer_name}")
    
    def get_compression_summary(self) -> Dict:
        """Get summary of compression applied to each layer."""
        summary = {}
        for layer_name, stats in self.compression_stats.items():
            summary[layer_name] = {
                'variance_ratios': f"A:{stats['variance_ratios'][0]:.0%}, G:{stats['variance_ratios'][1]:.0%}",
                'actual_variances': f"A:{stats['actual_variances'][0]:.1%}, G:{stats['actual_variances'][1]:.1%}",
                'ranks': f"rA={stats['ranks'][0]}, rG={stats['ranks'][1]}",
                'compression_ratio': f"{stats['compression_ratio']:.1%}"
            }
        return summary


# Example usage:
# kfac = KFACTreatment(model, 
#                      ['model.layers.31.mlp.gate_proj', 'model.layers.31.mlp.up_proj'],
#                      'path/to/kfac_factors.pt')
# 
# # Apply compression using variance retained (not rank ratio!)
# kfac.apply_kfac((0.75, 0.50))  # Retain 75% activation variance, 50% gradient variance
# 
# # Sweep through different ratios
# for ratio in [0.5, 0.6, 0.7, 0.8, 0.9]:
#     kfac.apply_kfac((ratio, ratio))
#     # Evaluate model...
# 
# kfac.restore_original_weights()  # Restore all layers


class KFACTreatmentPairwise(KFACTreatment):
    """
    Extends KFACTreatment with product‑based variance retaining.

    New public API
    ---------------
    apply_kfac_by_product(variance_ratio: float)
        Retains a fraction `variance_ratio` of the joint eigenvalue mass of
        H ≈ G ⊗ A by selecting the largest λ_i μ_j pairs.
    """

    # ---------- private helpers -------------------------------------------

    @staticmethod
    def _top_pairs_by_product(evals_G: torch.Tensor,
                              evals_A: torch.Tensor,
                              ratio: float) -> List[Tuple[int, int]]:
        """
        k‑way merge over the outer‑product matrix to fetch the
        largest λ_i μ_j until cumulative ≥ ratio · total.
        Runs in O(k log(min(m,n))) memory instead of materialising
        the full m×n outer product.
        """
        m, n = len(evals_G), len(evals_A)
        # make sure descending
        g = evals_G
        a = evals_A
        total_mass = (g.sum() * a.sum()).item()
        target_mass = ratio * total_mass

        # max‑heap keyed by −product so we pop the largest each time
        heap = [(-(g[0] * a[0]).item(), 0, 0)]
        seen = {(0, 0)}
        cum_mass = 0.0
        selected = []

        while cum_mass < target_mass and heap:
            neg_prod, i, j = heapq.heappop(heap)
            prod = -neg_prod
            cum_mass += prod
            selected.append((i, j))

            # Push neighbours (i+1, j) and (i, j+1)
            if i + 1 < m and (i + 1, j) not in seen:
                heapq.heappush(heap, (-(g[i + 1] * a[j]).item(), i + 1, j))
                seen.add((i + 1, j))
            if j + 1 < n and (i, j + 1) not in seen:
                heapq.heappush(heap, (-(g[i] * a[j + 1]).item(), i, j + 1))
                seen.add((i, j + 1))

        return selected

    def _project_weight_pairs(self,
                              info: Dict,
                              pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Projection using an arbitrary set of (i,j) pairs.
        W_proj = Σ_{(i,j)∈S} (u_i^T W v_j) · u_i  v_j^T
        """
        Ug = info['evc_G']         # [out, m]
        Ua = info['evc_A']         # [in , n]
        W  = info['W_orig'].float()

        # Pre–compute all required dot products in one matmul for speed
        # C = U_G^T W U_A  →  shape [m, n]
        C = Ug.T @ W @ Ua          # on-device

        # Build a mask that keeps only the selected coefficients
        mask = torch.zeros_like(C, device=C.device)
        idx_i, idx_j = zip(*pairs) if pairs else ([], [])
        mask[idx_i, idx_j] = 1.0
        # Count how many unique eigenvectors from G (rows) and A (cols) are used
        used_rows = (mask.sum(dim=1) > 0).sum().item()
        used_cols = (mask.sum(dim=0) > 0).sum().item()

        # Shape diagnostics
        rows_full = int((mask.sum(dim=1) == mask.shape[1]).sum().item())
        cols_full = int((mask.sum(dim=0) == mask.shape[0]).sum().item())
        row0_pairs = int(mask[0].sum().item()) if mask.shape[0] > 0 else 0
        col0_pairs = int(mask[:, 0].sum().item()) if mask.shape[1] > 0 else 0
        full_cols_mask = (mask.sum(dim=0) == mask.shape[0])
        if full_cols_mask.any():
            full_block_J = int(full_cols_mask.nonzero().max().item() + 1)
        else:
            full_block_J = 0

        # Compute mass-based effective counts: how many G/A eigvecs explain
        # 95% of the selected joint mass (sum of g[i]*a[j] over selected pairs)
        eva_G = info['eva_G']
        eva_A = info['eva_A']
        if eva_G.device != mask.device:
            eva_G = eva_G.to(mask.device)
        if eva_A.device != mask.device:
            eva_A = eva_A.to(mask.device)

        # Per-row and per-column mass contributions for selected pairs
        # row_mass[i] = eva_G[i] * sum_j mask[i,j] * eva_A[j]
        # col_mass[j] = eva_A[j] * sum_i mask[i,j] * eva_G[i]
        row_mass = eva_G * (mask @ eva_A)
        col_mass = eva_A * (mask.T @ eva_G)
        selected_mass = row_mass.sum().item()
        total_mass = (eva_G.sum() * eva_A.sum()).item()

        def _eff_k_for_fraction(mass_vec: torch.Tensor, frac: float) -> int:
            if mass_vec.numel() == 0:
                return 0
            if mass_vec.sum() <= 0:
                return 0
            vals, _ = torch.sort(mass_vec, descending=True)
            csum = torch.cumsum(vals, dim=0)
            target = frac * csum[-1]
            k = int((csum >= target).nonzero()[0].item()) + 1
            return k

        eff95_G = _eff_k_for_fraction(row_mass, 0.95)
        eff95_A = _eff_k_for_fraction(col_mass, 0.95)

        # Prefix coverage (smallest prefix sizes that contain all used indices)
        max_i = int(max(idx_i)) + 1 if idx_i else 0
        max_j = int(max(idx_j)) + 1 if idx_j else 0
        C_masked = C * mask        # element‑wise

        W_proj = Ug @ C_masked @ Ua.T
        # Attach counts for caller via info for later reporting
        info['used_eig_G'] = used_rows
        info['used_eig_A'] = used_cols
        info['eff95_eig_G'] = eff95_G
        info['eff95_eig_A'] = eff95_A
        info['prefix_eig_G'] = max_i
        info['prefix_eig_A'] = max_j
        info['selected_mass'] = selected_mass
        info['total_mass'] = total_mass
        info['selected_mass_ratio'] = (selected_mass / total_mass) if total_mass > 0 else 0.0
        info['mass_ratio'] = (selected_mass / total_mass) if total_mass > 0 else 0.0
        info['rows_full'] = rows_full
        info['cols_full'] = cols_full
        info['row0_pairs'] = row0_pairs
        info['col0_pairs'] = col0_pairs
        info['full_block_J'] = full_block_J
        return W_proj

    # ---------- public method ---------------------------------------------

    def apply_kfac_by_product(self,
                              variance_ratio: Union[float,
                                                    Dict[str, float]]):
        """
        Project each registered layer onto the span of the largest
        λ_i μ_j products until the chosen fraction of total mass is kept.

        variance_ratio
            • single float → same ratio for every layer, or
            • dict[layer_name] = ratio
        """
        # Build a per‑layer ratio map
        if isinstance(variance_ratio, float):
            ratio_map = {name: variance_ratio for name in self.layer_names}
        else:
            ratio_map = variance_ratio

        with torch.no_grad():
            for layer_name in self.layer_names:
                rho = ratio_map[layer_name]
                info = self.kfac_info[layer_name]

                eva_G = info['eva_G'].to(self.device)
                eva_A = info['eva_A'].to(self.device)
                assert (eva_G[:-1] >= eva_G[1:]).all(), "eva_G must be sorted desc"
                assert (eva_A[:-1] >= eva_A[1:]).all(), "eva_A must be sorted desc"

                # Step 1: which (i,j) pairs?
                pairs = self._top_pairs_by_product(eva_G, eva_A, rho)

                # Step 2: build projection
                W_proj = self._project_weight_pairs(info, pairs)

                # Step 3: write back
                layer = self._get_layer_by_name(layer_name)
                layer.weight.copy_(W_proj.to(layer.weight.dtype))

                # Book‑keeping
                k = len(pairs)
                info_str = f"{layer_name}: {k} pairs ({k/ info['W_orig'].numel():.2%})"
                print(info_str)
                # Record stats for external reporting
                rect_params = int(info.get('prefix_eig_G', 0)) * int(info.get('prefix_eig_A', 0))
                orig_params = int(info['W_orig'].numel())
                self.compression_stats[layer_name] = {
                    'num_pairs': k,
                    'pair_ratio': k / info['W_orig'].numel(),
                    'uniq_G': int(info.get('used_eig_G', 0)),
                    'uniq_A': int(info.get('used_eig_A', 0)),
                    'eff95_G': int(info.get('eff95_eig_G', 0)),
                    'eff95_A': int(info.get('eff95_eig_A', 0)),
                    'prefix_G': int(info.get('prefix_eig_G', 0)),
                    'prefix_A': int(info.get('prefix_eig_A', 0)),
                    'dim_G': int(info['evc_G'].shape[1]),
                    'dim_A': int(info['evc_A'].shape[1]),
                    'rho': float(rho),
                    'mass_ratio': float(info.get('mass_ratio', 0.0)),
                    'rect_params': rect_params,
                    'rect_ratio': (rect_params / orig_params) if orig_params > 0 else 0.0,
                    'rows_full': int(info.get('rows_full', 0)),
                    'cols_full': int(info.get('cols_full', 0)),
                    'row0_pairs': int(info.get('row0_pairs', 0)),
                    'col0_pairs': int(info.get('col0_pairs', 0)),
                    'full_block_J': int(info.get('full_block_J', 0)),
                }