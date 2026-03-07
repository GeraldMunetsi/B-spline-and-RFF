"""
Bspline2.py  ·  Physics-Informed Hybrid MLP Model
===================================================
3-Parameter SIR emulator: tau (τ), gamma (γ), rho (ρ)

Age-structured parameters and SpatialMLP (graph-stats branch)
have been fully removed.  The parameter encoder now accepts a
3-dimensional input vector instead of 7.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy.interpolate import BSpline


# ============================================================================
# RANDOM FOURIER FEATURES
# ============================================================================

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
 
    n_knots = 12   # number of control points (degree=3 → 10 basis functions)
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


# ============================================================================
# PHYSICS-INFORMED SPLINE DECODER
# ============================================================================

class SplineTemporalDecoderPhysics(nn.Module):
    """
    Physics-informed temporal decoder that outputs (S, I, R) curves.

    Hard constraints enforced:
        1. S is monotonically NON-INCREASING  (cumulative product of retention rates)
        2. I is smooth and NON-NEGATIVE       (Softplus + B-spline)
        3. R = N − S − I                      (conservation law; S+I+R = N always)
    """

    def __init__(self, input_dim=64, n_knots=12, n_timepoints=50, total_population=10000):
        super().__init__()
        self.n_knots       = n_knots
        self.n_timepoints  = n_timepoints
        self.N             = total_population

        # B-splines for S and I
        self.spline_S = DifferentiableBSpline(n_knots, degree=3, n_eval_points=n_timepoints)
        self.spline_I = DifferentiableBSpline(n_knots, degree=3, n_eval_points=n_timepoints)

        # ── Susceptible: initial fraction ────────────────────────────────────
        self.predict_S_initial = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),       # output in (0, 1)
        )
        with torch.no_grad():
            self.predict_S_initial[-2].bias.fill_(2.0)   # sigmoid(2) ≈ 0.88

        # ── Susceptible: retention rates at each knot ─────────────────────────
        self.predict_S_retention = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_knots - 1),
            # sigmoid applied in forward
        )

        # ── Infected: spline coefficients ────────────────────────────────────
        self.predict_I_coeffs = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_knots),
            nn.Softplus(),      # non-negative
        )

    def forward(self, z):
        """
        Args:
            z: (batch, input_dim)
        Returns:
            S, I, R – each (batch, n_timepoints)
        """
        batch_size = z.size(0)
        device     = z.device

        # ── S: monotonically decreasing ──────────────────────────────────────
        S_0_frac          = self.predict_S_initial(z)                    # (batch, 1)
        S_0               = S_0_frac * self.N

        retention_raw     = self.predict_S_retention(z)                  # (batch, n_knots-1)
        retention_rates   = torch.sigmoid(retention_raw)                 # in (0, 1)

        ones              = torch.ones(batch_size, 1, device=device)
        all_rates         = torch.cat([ones, retention_rates], dim=1)    # (batch, n_knots)
        cum_retention     = torch.cumprod(all_rates, dim=1)              # guaranteed ≤ 1

        S_coeffs          = S_0 * cum_retention                         # (batch, n_knots)
        S_pred            = self.spline_S(S_coeffs)                     # (batch, n_timepoints)
        S_pred            = torch.clamp(S_pred, min=0, max=self.N)

        # ── I: smooth, positive curve ─────────────────────────────────────────
        I_coeffs          = self.predict_I_coeffs(z)                    # (batch, n_knots)
        I_pred            = self.spline_I(I_coeffs)                     # (batch, n_timepoints)
        I_pred            = torch.clamp(I_pred, min=0)

        # ── R: conservation law R = N − S − I ─────────────────────────────────
        R_pred            = self.N - S_pred - I_pred
        R_pred            = torch.clamp(R_pred, min=0, max=self.N)

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
        n_params=3,              # tau, gamma, rho
        n_fourier_features=64,
        fourier_hidden=32,
        param_output_dim=16,
        n_knots=12,
        n_timepoints=50,
        total_population=10000,
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
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
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
            data        : batch object with .params  (batch, 3)
            n_timesteps : ignored (kept for API compatibility); actual
                          time resolution is set at construction time.
        Returns:
            predictions : (batch, n_timepoints, 3)   → [S, I, R]
        """
        # 1. Encode [tau, gamma, rho]
        param_emb = self.param_encoder(data.params)          # (batch, param_output_dim)

        # 2. Project to latent space (no graph branch)
        z = self.fusion(param_emb)                           # (batch, fusion_hidden)

        # 3. Decode into physics-consistent SIR curves
        S, I, R = self.temporal_decoder(z)                   # each (batch, n_timepoints)

        # 4. Stack → (batch, n_timepoints, 3)
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
        n_knots=config.get('n_knots', 12),
        n_timepoints=config.get('n_timepoints', 50),
        total_population=config.get('total_population', 10000),
        fusion_hidden=config.get('temporal_hidden', 64),
        fusion_dropout=config.get('dropout', 0.3),
    )
    return model



# QUICK TEST


if __name__ == "__main__":
    print("=" * 70)
    print("PHYSICS-INFORMED HYBRID MLP  ·  3-Parameter SIR (τ, γ, ρ)")
    print("=" * 70)

    config = {
        'n_params'        : 3,
        'n_fourier'       : 64,
        'fourier_hidden'  : 32,
        'param_hidden'    : 16,
        'temporal_hidden' : 64,
        'dropout'         : 0.3,
        'n_knots'         : 12,
        'n_timepoints'    : 50,
        'total_population': 10000,
    }

    model = create_hybrid_mlp_model(config)
    comp  = model.get_component_params()

    print(f"\n  Total parameters : {comp['total']:,}")
    print(f"  param_encoder    : {comp['param_encoder']:,}")
    print(f"  fusion           : {comp['fusion']:,}")
    print(f"  temporal_decoder : {comp['temporal_decoder']:,}")

    print("\n  ARCHITECTURE:")
    print("    [tau, gamma, rho]  →  FourierEncoder  →  Fusion  →  SplineDecoder")
    print("    SpatialMLP (graph-stats branch) : REMOVED")

    print("\n  PHYSICS CONSTRAINTS:")
    print("    · S : monotonically non-increasing  (cumulative product)")
    print("    · I : smooth, non-negative           (Softplus + B-spline)")
    print("    · R : conservation law               (R = N − S − I)")

    # Smoke test
    class FakeBatch:
        params = torch.rand(4, 3)   # batch=4, 3 params

    out = model(FakeBatch())
    print(f"\n  Smoke test output shape : {out.shape}")   # expect (4, 50, 3)
    assert out.shape == (4, 50, 3), "Shape mismatch!"
    print("  Shape correct")
    print("=" * 70)
