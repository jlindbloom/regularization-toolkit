import numpy as np
from scipy.interpolate import splev
from scipy.interpolate import splrep, BSpline

from ..cg import relative_residual_cg



def lcurve(F, R, y, lambda_min=1e-7, lambda_max=1e4, n_lambdas=200, cg_tol=1e-7, cg_maxits=1000):
    """Computes points on the L-curve corresponding to Tikhonov regularization. Here
            x_{\lambda} = \argmin_{x} \| F x -y \|_2^2 + \lambda \| R x \|_2^2,
            data_fidelity = \| F x_{\lambda} - y \|_2^2 
            reg_term = \| R x_{\lambda} \|_2^2
    and we output a dict of these terms, as well as the lambda values.
    """

    # Setup
    lambdas = np.logspace(np.log(lambda_min), np.log(lambda_max), num=n_lambdas)
    x_tikh = None
    fidelity_terms = []
    reg_terms = []
    x_tikhs = []
    
    # Compute terms
    for lam in lambdas:

        # Make mat
        Q = (F.T @ F) + lam*(R.T @ R)

        # Get the Tikhonov solution for the current value of lambda
        sol = relative_residual_cg(Q, F.T @ y, eps=cg_tol, maxits=cg_maxits, x0=x_tikh)
        x_tikh = sol["x"]

        # Compute norms
        fidelity_term = np.linalg.norm(F @ x_tikh - y, ord=2)
        reg_term = np.linalg.norm(R @ x_tikh, ord=2)
        fidelity_terms.append(fidelity_term)
        reg_terms.append(reg_term)
        x_tikhs.append(x_tikh)

    data = {
        "lambdas": lambdas,
        "x_lambdas": x_tikhs,
        "data_terms": np.asarray(fidelity_terms),
        "reg_terms": np.asarray(reg_terms),
    }

    return data



def lcurve_corner(lcurve_data):
    """Assuming the L curve actually has a corner, produces an estimate of the corner. Accepts
    output of lcurve as input, uses spline method to find point of max curvature."""
    
    # Get data
    data_terms = lcurve_data["data_terms"]
    reg_terms = lcurve_data["reg_terms"]
    lambdas = lcurve_data["lambdas"]
    x_lambdas = lcurve_data["x_lambdas"]

    # Reorder so that increasing in data_terms
    inds = data_terms.argsort()
    data_terms = data_terms[inds]
    reg_terms = reg_terms[inds]
    lambdas = lambdas[inds]

    # Make spline
    tck = splrep( np.log(data_terms), np.log(reg_terms), s=5)

    # Grid to evaluate curvatures on
    dom = np.linspace( np.log(np.amin(data_terms)), np.log(np.amax(data_terms)), num=500  )

    # Compute first and second derivatives
    first_derivs = splev(dom, tck, der=1)
    second_derivs = splev(dom, tck, der=2)
    curvatures = np.abs(second_derivs)/( (1 + (first_derivs**2) )**(1.5) )

    # Get index of max curvature
    max_curve_idx = np.argmax(curvatures)

    # Find closest data_term coord to point of max curvature
    max_curve_data_term = dom[max_curve_idx]
    closest_data_term_idx = np.abs(np.log(data_terms) - max_curve_data_term).argmin() 

    # Return the lambda corresponding to this value
    max_curvature_lambda_est = lambdas[closest_data_term_idx]
    max_curvature_x_lambda = x_lambdas[closest_data_term_idx]
    max_curvature_data_term = data_terms[closest_data_term_idx]
    max_curvature_reg_term = reg_terms[closest_data_term_idx]

    # Package output
    data = {
        "lambda": max_curvature_lambda_est,
        "x_lambda": max_curvature_x_lambda,
        "data_term": max_curvature_data_term,
        "reg_term": max_curvature_reg_term
    }

    return data


