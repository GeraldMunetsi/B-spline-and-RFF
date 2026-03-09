import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy.interpolate import BSpline


# Constants
N=10000
n_knots = 20
n_timepoints=50


# RANDOM FOURIER FEATURES
class RandomFourierFeatures(nn.Module):
    """
    Random Fourier Features for kernel approximation (Rahimi & Recht, 2007).

    Maps a low-dimensional parameter vector into a high-dimensional feature
    space that approximates a shift-invariant (RBF) kernel.  This helps the
    subsequent MLP learn smooth, non-linear functions over the parameter space.
    """

    def __init__(self, input_dim=3, n_features=64, sigma=1.0, learnable=False):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.sigma = sigma

        W = torch.randn(input_dim, n_features) / sigma
        b = torch.rand(n_features) * 2 * np.pi

        if learnable:
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer('W', W)
            self.register_buffer('b', b)

        self.scale = math.sqrt(2.0 / n_features)

    def forward(self, x):
        projection = x @ self.W + self.b          # (batch, n_features)
        return self.scale * torch.cos(projection)  # (batch, n_features)


# ============================================================================
# PARAMETER ENCODER  (3-parameter SIR)
# ============================================================================

class FourierParameterEncoder(nn.Module):
    """
    Encodes the 3 SIR parameters [tau, gamma, rho] into a dense embedding.

    Pipeline:
        [tau, gamma, rho]  →  RandomFourierFeatures  →  MLP  →  embedding
    """

    def __init__(self, input_dim=3, n_fourier=64, hidden_dim=32, output_dim=16):
        super().__init__()

        self.rff = RandomFourierFeatures(
            input_dim=input_dim,   # ← 3 (was 7)
            n_features=n_fourier,
            sigma=1.0,
            learnable=False,
        )

        self.mlp = nn.Sequential(
            nn.Linear(n_fourier, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, params):
        """
        Args:
            params: (batch, 3)  → [tau, gamma, rho]
        Returns:
            encoding: (batch, output_dim)
        """
        fourier_features = self.rff(params)
        return self.mlp(fourier_features)


# ============================================================================
# B-SPLINE BASIS
# ============================================================================


class DifferentiableBSpline(nn.Module):
    """
    Differentiable cubic B-spline evaluated at fixed time points.

    The spline maps a set of learnable control-point coefficients onto a
    smooth, dense temporal curve.  This is what prevents the 'steppy'
    predictions you saw with plain MLP outputs.
    """
 
    # number of control points (degree=3 → 10 basis functions)
    def __init__(self, n_knots= n_knots, degree=3, n_eval_points=50):
        super().__init__()
        self.n_knots        = n_knots
        self.degree         = degree
        self.n_eval_points  = n_eval_points

        knots = self._create_knot_vector(n_knots, degree)
        self.register_buffer('knots', knots)

        eval_points = torch.linspace(0, 1, n_eval_points)
        self.register_buffer('eval_points', eval_points)

        basis_matrix = self._compute_basis_matrix_scipy()
        self.register_buffer('basis_matrix', basis_matrix)

    def _create_knot_vector(self, n_basis, degree):
        n_knots        = n_basis + degree + 1
        interior_count = n_knots - 2 * (degree + 1)
        interior       = torch.linspace(0, 1, interior_count + 2)[1:-1]
        return torch.cat([
            torch.zeros(degree + 1),
            interior,
            torch.ones(degree + 1),
        ])

    def _compute_basis_matrix_scipy(self):
        knots_np    = self.knots.cpu().numpy()
        k           = self.degree
        n_basis     = len(knots_np) - k - 1
        basis_matrix = np.zeros((self.n_eval_points, n_basis))
        eval_pts    = self.eval_points.cpu().numpy()

        for i in range(n_basis):
            coeff       = np.zeros(n_basis)
            coeff[i]    = 1.0
            spline      = BSpline(knots_np, coeff, k)
            basis_matrix[:, i] = spline(eval_pts)

        return torch.tensor(basis_matrix, dtype=torch.float32)

    def forward(self, coefficients):
        """
        Args:
            coefficients: (batch, n_knots)
        Returns:
            curve: (batch, n_eval_points)
        """
        return coefficients @ self.basis_matrix.T


# PHYSICS-INFORMED SPLINE DECODER


class SplineTemporalDecoderPhysics(nn.Module):
    """
    Physics-informed temporal decoder that outputs (S, I, R) curves.

    What the network actually learns (the ONLY free components):
        - How fast S decays (n_knots-1 retention rates)
        - The shape of the I bell/decay curve (n_knots - 1 free coefficients;
          first coefficient is pinned to I(0))
    """

    def __init__(self, input_dim: int = 64, n_knots: int = 12,
                 n_timepoints: int = 50, total_population: int = 10000):
        super().__init__()
        self.n_knots      = n_knots
        self.n_timepoints = n_timepoints
        self.N            = total_population

        # Two independent B-spline layers (S and I have different constraints)
        self.spline_S = DifferentiableBSpline(n_knots, degree=3, n_eval_points=n_timepoints)
        self.spline_I = DifferentiableBSpline(n_knots, degree=3, n_eval_points=n_timepoints)

        # ── S decoder: instead of learning sline coefficientss dirrectly we learn retention rates via MLP
        # Predicts n_knots-1 retention rates r₁…r_{K-1} ∈ (0,1)
        # S_coeff[k] = S₀ · r₁ · r₂ · … · r_k  guaranteed monotone decreasing
        self.predict_S_retention = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),  # feature extraction , ransform latent embedding into nonlinear representation.
            nn.Dropout(0.3), # avoid overfitting 
            nn.Linear(32, n_knots - 1), # output projection, mapping hidden represenation  to spline coefifient spcae
            # sigmoid applied in forward to map to (0, 1)
        )

        # I decoder: We directly learn B-spline coefficients execpt the firrst one (except the first one, pinned to I(0)=rho*N 
        # We use the coeffs as comtrol points of the curve
      
        self.predict_I_coeffs = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_knots),
            # No activation — raw logits, can be negative, spline output clamped later
        )

    def forward(self, z: torch.Tensor, rho_raw: torch.Tensor) -> tuple:
        """
        Args:
            z       : (batch, input_dim)  — latent vector from fusion MLP
            rho_raw : (batch,)            — raw (un-normalised) rho values in [0.001, 0.010]
             so that I₀ = rho * N gives the correct count in people.

        Returns:
            S_pred, I_pred, R_pred each (batch, n_timepoints)
        """
        batch_size = z.size(0)
        device     = z.device

        # S(0) = N·(1-rho),  I(0) = N·rho,  R(0) = 0 
        S_0 = ((1.0 - rho_raw) * self.N).unsqueeze(1)   # (batch, 1) adding  a dimension
        I_0 = (rho_raw * self.N).unsqueeze(1)            # (batch, 1)

        # S: monotonically decreasing curve 
        retention_raw = self.predict_S_retention(z)            # (batch, n_knots-1)
        retention_rates= torch.sigmoid(retention_raw)           # (batch, n_knots-1), in (0,1)

        ones= torch.ones(batch_size, 1, device=device)
        all_rates= torch.cat([ones, retention_rates], dim=1)  # (batch, n_knots)
        cum_product = torch.cumprod(all_rates, dim=1)         #decreasing numbers from 1  , so im guanrteed S is decreasing right  (cumprod of positives × S(0)> 0), 0 < cumprod ≤ 1

   
        S_coeffs = S_0 * cum_product        #s coeeficints start at S(0) since r(0)=1 and decay monotonically                  
        S_pred   = self.spline_S(S_coeffs)                       
    

        I_coeffs = self.predict_I_coeffs(z)                     
        I_coeffs = torch.cat([I_0, I_coeffs[:, 1:]], dim=1) #    Basis(0) =1, (clamped spline) , The remaining n_knots-1 coefficients are free to learn the curve shape.
        I_pred = self.spline_I(I_coeffs)                         # (batch, n_timepoints)
        I_pred = F.softplus(I_pred)                   # I(t) > 0 , non-negativity after spline

        
        # Conservation law: R = N - S - I  (exact, never predicted)
        # Do NOT apply softplus here — softplus(N-S-I) ≠ N-S-I,
        # which would break S+I+R=N and make R a free prediction instead
        # of a derived quantity.
        R_pred = self.N - S_pred - I_pred

        return S_pred, I_pred, R_pred

# MAIN MODEL 


class HybridSplineFourierMLPPhysics(nn.Module):
    """
    Physics-Informed Hybrid MLP for 3-Parameter SIR Emulation.

    Architecture:
        params [tau, gamma, rho]
            ↓
        FourierParameterEncoder   (RFF → MLP → embedding)
            ↓
        Fusion MLP                (embedding → latent z)
            ↓
        SplineTemporalDecoderPhysics  (z → S, I, R curves)

    The SpatialMLP (graph-stats branch) has been removed because we are
    working with a fixed Barabási-Albert network, so graph topology is
    constant and does not need to be fed as a dynamic input.
    """

    def __init__(
        self,
        n_params=3,              # tau, gamma, rhox
        n_fourier_features=64,
        fourier_hidden=32,
        param_output_dim=16,
        n_knots= n_knots,
        n_timepoints=n_timepoints,
        total_population=N,
        fusion_hidden=64, 
        fusion_dropout=0.3,
    ):
        super().__init__()
        self.n_timepoints = n_timepoints
        

        # ── Component 1: Parameter encoder ───────────────────────────────────
        self.param_encoder = FourierParameterEncoder(
            input_dim=n_params,          # 3
            n_fourier=n_fourier_features,
            hidden_dim=fourier_hidden,
            output_dim=param_output_dim,
        )

        # ── Component 2: Fusion (param_emb only; no graph branch) ────────────
        self.fusion = nn.Sequential(
            nn.Linear(param_output_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden), # recall this applies Applies Batch Normalization over a my 3D input.
            nn.ReLU(),
            nn.Dropout(fusion_dropout),# During training, randomly zeroes some of the elements of the input tensor with probability p.
            nn.Linear(fusion_hidden, fusion_hidden), #Applies an affine linear transformation to the incoming data:
        )

        # ── Component 3: Physics-informed decoder ────────────────────────────
        self.temporal_decoder = SplineTemporalDecoderPhysics(
            input_dim=fusion_hidden,
            n_knots=n_knots,
            n_timepoints=n_timepoints,
            total_population=total_population,
        )

    def forward(self, data, n_timesteps=None, **kwargs):
        """
        Forward pass.

        Args:
            data        : BatchWrapper from utils_SIR.collate_sir()
                          must have .params_norm (batch,3) and .rho_raw (batch,)
            n_timesteps : ignored (kept for API compatibility); actual
                          time resolution is set at construction time.
        Returns:
            predictions : (batch, n_timepoints, 3)   → [S, I, R]
        """
        # 1. Read batch fields produced by utils_SIR.collate_sir()
        #    params_norm : (batch, 3) normalised to [0,1] — for Fourier encoder
        #    rho_raw     : (batch,)   raw rho in [0.001,0.010] — for decoder ICs
        params_norm = data.params_norm                       # (batch, 3) in [0, 1]
        rho_raw     = data.rho_raw                           # (batch,)   raw

        # 2. Encode normalised parameters
        param_emb = self.param_encoder(params_norm)          # (batch, param_output_dim)

        # 3. Project to latent space
        z = self.fusion(param_emb)                           # (batch, fusion_hidden)

        # 4. Decode — rho_raw passed separately so decoder can pin S(0) and I(0) exactly
        S, I, R = self.temporal_decoder(z, rho_raw)          # each (batch, n_timepoints)

        # 5. Stack → (batch, n_timepoints, 3)
        return torch.stack([S, I, R], dim=2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_component_params(self):
        """Parameter count per component (useful for diagnostics)."""
        return {
            'param_encoder'   : sum(p.numel() for p in self.param_encoder.parameters()),
            'fusion'          : sum(p.numel() for p in self.fusion.parameters()),
            'temporal_decoder': sum(p.numel() for p in self.temporal_decoder.parameters()),
            'total'           : self.count_parameters(),
        }

# FACTORY FUNCTION


def create_hybrid_mlp_model(config):
    """
    Instantiate the 3-parameter SIR hybrid model from a config dict.

    Minimal required keys:
        n_fourier, fourier_hidden, param_hidden, temporal_hidden,
        dropout, n_knots, n_timepoints, total_population

    Keys that were relevant to the old 7-parameter age-structured model
    (mlp_input_dim, mlp_hidden, mlp_layers) are silently ignored if present,
    so old checkpoints/configs don't break on load.
    """
    model = HybridSplineFourierMLPPhysics(
        n_params=config.get('n_params', 3),
        n_fourier_features=config.get('n_fourier', 64),
        fourier_hidden=config.get('fourier_hidden', 32),
        param_output_dim=config.get('param_hidden', 16),
        n_knots=config.get('n_knots',  n_knots),
        n_timepoints=config.get('n_timepoints', n_timepoints),
        total_population=config.get('total_population', N),
        fusion_hidden=config.get('temporal_hidden', 64),
        fusion_dropout=config.get('dropout', 0.3),
    )
    return model



# QUICK TEST


if __name__ == "__main__":
 
    print("Physics-informed MLP ")

    config = {
        'n_params'        : 3,
        'n_fourier'       : 64,
        'fourier_hidden'  : 32,
        'param_hidden'    : 16,
        'temporal_hidden' : 64,
        'dropout'         : 0.3,
        'n_knots'         : n_knots ,
        'n_timepoints'    : n_timepoints,
        'total_population': N,
    }

    model = create_hybrid_mlp_model(config)
    comp  = model.get_component_params()

    print(f"\n  Total parameters : {comp['total']:,}")
    print(f"  param_encoder    : {comp['param_encoder']:,}")
    print(f"  fusion           : {comp['fusion']:,}")
    print(f"  temporal_decoder : {comp['temporal_decoder']:,}")

    
    #[tau, gamma, rho]  →  FourierEncoder  →  Fusion  →  SplineDecoder- Something to tell Alex")

    # Smoke test — FakeBatch must match BatchWrapper API from utils_SIR
    import types
    fake = types.SimpleNamespace(
        params_norm = torch.rand(4, 3),               # normalised [0,1]
        rho_raw     = torch.FloatTensor([0.005]*4),   # raw rho
    )
    model.eval()
    with torch.no_grad():
        out = model(fake)
    print(f"\n  Smoke test output shape : {out.shape}")   # expect (4, 50, 3)

    S, I, R = out[:,:,0], out[:,:,1], out[:,:,2]
    print(f"  Conservation check S+I+R=N  max error: {(S+I+R - N).abs().max().item():.4f}")
    print(f"  I(0) pinned to rho*N=50:    max error: {(I[:,0] - 50.0).abs().max().item():.4f}")
    print(f"  S(0) pinned to (1-rho)*N:   max error: {(S[:,0] - 9950.0).abs().max().item():.4f}")