import numpy as np
from scipy.optimize import root_scalar

from ..cg import relative_residual_cg



def discrepancy_principle(F, R, y, noise_sigma, safeguard_fac=1.05, cg_tol=1e-4, cg_maxits=1000, lambda_0=1e-4, lambda_1=1e4):
    """Computes a regularization parameter estimate using the discrepancy principle.

    Uses bisection method for finding root of

        f(\lambda) = (1/m)\| F x_{\lambda} - y \|_2^2 - \sigma^2,

    where a unique root is assumed to lie in the interval [lambda_0, lambda_1].

    noise_sigma is an estimate of the standard deviation of i.i.d. Gaussian noise in the signal.
    """
    # Get # measurements
    m = len(y)

    # Setup
    x_lambda = None

    # Define objective
    def _objective(lam):
        x_lambda = None

        # Get current solution
        Q = (F.T @ F) + lam*(R.T @ R)
        sol = relative_residual_cg(Q, F.T @ y, eps=cg_tol, maxits=cg_maxits, x0=x_lambda)
        x_lambda = sol["x"]

        # Evaluate objective
        objective = (1/m)*( (np.linalg.norm( (F @ x_lambda) - y, ord=2)**2) ) - safeguard_fac*(noise_sigma**2)

        return objective
    
    # Call solver
    bisection_sol = root_scalar(_objective, method="bisect", bracket=[lambda_0, lambda_1])

    # Also return the solution for this given lambda
    Q = (F.T @ F) + bisection_sol.root*(R.T @ R)
    sol = relative_residual_cg(Q, F.T @ y, eps=cg_tol, maxits=cg_maxits, x0=x_lambda)
    x_lambda = sol["x"]

    # Form output
    data = {
        "scipy_solve": sol,
        "lambda": bisection_sol.root,
        "x_lambda": x_lambda,
    }

    return data


    