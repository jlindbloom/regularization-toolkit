import numpy as np

from tracelogdetdiag.util import AinvCGLinearOperator
from tracelogdetdiag.trace import hutch_plus_plus_trace, hutchinson_trace

from ..cg import relative_residual_cg




def maximum_expectation(F, R, y, n_iterations=5, trace_samples=30, cg_tol=1e-4, cg_maxits=1000, lambda_0=1e1):
    """Implements the maximum expectation method in [9].
    """

    # Parameters
    uk = None
    A = F # renaming
    T = R # renaming
    b = y # renaming
    d = A.shape[1]
    reg_lambda = lambda_0

    for k in range(n_iterations):
        
        # Make new H
        Hk = (A.T @ A) + reg_lambda*(T.T @ T)

        # Get current reconstruction
        uk = relative_residual_cg(Hk, A.T @ b, x0=uk, eps=cg_tol, maxits=cg_maxits)["x"]

        # Evaluate traces
        Hkinv = AinvCGLinearOperator(Hk, cg_tol=cg_tol, cg_maxits=cg_maxits, use_prev=True)
        #T_trace = hutch_plus_plus_trace(Hkinv @ ( A.T @ A ), sample_size=6 )
        T_trace = hutchinson_trace(Hkinv @ ( T.T @ T ), sample_size=trace_samples)
        A_trace = hutchinson_trace(Hkinv @ ( A.T @ A ), sample_size=trace_samples)

        # Improve trace estimate
        denom_trace = A_trace + reg_lambda*T_trace
        T_trace = d*T_trace/denom_trace
        A_trace = d*A_trace/denom_trace

        # Compute new sigma and eta
        residual = (A @ uk) - b
        sigma_squared = (residual.T @ residual)/(A.shape[0] - A_trace)
        Tuk = T @ uk
        eta_squared = (Tuk.T @ Tuk)/( (T.shape[0]) - reg_lambda*T_trace) # Here, really need to involve the rank of T
        reg_lambda = sigma_squared/eta_squared  


    # Get final reconstruction
    Hk = (A.T @ A) + reg_lambda*(T.T @ T)
    uk = relative_residual_cg(Hk, A.T @ b, x0=uk, eps=cg_tol, maxits=cg_maxits)["x"]


    data = {
        "lambda": reg_lambda,
        "noise_var": sigma_squared,
        "prior_var": eta_squared,
        "x_lambda": uk,
    }

    return data











