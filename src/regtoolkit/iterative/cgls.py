import numpy as np



def cgls_early_stopping_regularization(A, b, noise_sigma, maxits=100, tau=1.03, early_stopping=True):
    """Performs CGLS with early stopping regularization applied to minimizing
                || A x - b ||_2 ,
    stopping at the first iteration such that
        || A x - b ||_2^2 <= tau * m * noise_var.
    """

    n = A.shape[1]
    m = A.shape[0]
    x = np.zeros(n)
    r_prev = b - (A @ x)
    d_prev = A.T @ r_prev

    # Tracking
    squared_residuals = []
    squared_res = np.linalg.norm((A @ x) - b)**2
    squared_residuals.append(squared_res)
    n_iterations = 0

    for k in range(maxits):

        # CGLS iteration
        alpha = (np.linalg.norm(A.T @ r_prev)**2)/(np.linalg.norm(A @ d_prev)**2)
        x = x + alpha*d_prev
        r_next = r_prev - alpha*(A @ d_prev)
        beta = (np.linalg.norm(A.T @ r_next)**2/(np.linalg.norm(A.T @ r_prev)**2))
        d_next = (A.T @ r_next) + beta*d_prev

        d_prev = d_next
        r_prev = r_next

        # Track this
        n_iterations += 1
        squared_res = np.linalg.norm((A @ x) - b)**2
        squared_residuals.append(squared_res)

        if early_stopping and (squared_res < tau*(m*(noise_sigma**2))):
            break

    data = {
        "x": x,
        "n_iterations": n_iterations,
        "squared_residuals": squared_residuals,
    }

    return data



def cgls_for_white_noise_tikhonov(A, b, noise_sigma, maxits=100, tau=1.02, early_stopping=True):
    """Performs CGLS with early stopping regularization applied to minimizing
                || \hat{A} x - \hat{b} ||_2^2 + || x ||_2.
    Here \hat{A}/\hat{b} are re-scalings of A and b by the noise variance. We seek the minimizer by applying
    CGLS towards the minimization of
                || \hat{A} x - \hat{b} ||_2^2
    and using early stopping. We stop at the first iteration such that
                || \hat{A} x - \hat{b} ||_2^2 <= tau * m,
    OR when the original objective function begins to increase.
    """

    # Whiten by noise variance
    A = (1/noise_sigma)*A
    b = (1/noise_sigma)*b

    #print(np.linalg.norm((A @ x) - y))
    n = A.shape[1]
    m = A.shape[0]
    x = np.zeros(n)
    r_prev = b - (A @ x)
    d_prev = A.T @ r_prev

    # Tracking
    squared_residuals = []
    squared_res = np.linalg.norm((A @ x) - b)**2
    squared_residuals.append(squared_res)
    n_iterations = 0

    # original objective?
    prev_original_obj = squared_res + (np.linalg.norm(x)**2)
    original_objs = []
    original_objs.append(prev_original_obj)

    # which stopping criterion satisfied?
    which_broke = "none"

    for k in range(maxits):

        # CGLS iteration
        alpha = (np.linalg.norm(A.T @ r_prev)**2)/(np.linalg.norm(A @ d_prev)**2)
        x = x + alpha*d_prev
        r_next = r_prev - alpha*(A @ d_prev)
        beta = (np.linalg.norm(A.T @ r_next)**2/(np.linalg.norm(A.T @ r_prev)**2))
        d_next = (A.T @ r_next) + beta*d_prev

        d_prev = d_next
        r_prev = r_next

        # Track this
        n_iterations += 1
        squared_res = np.linalg.norm((A @ x) - b)**2
        squared_residuals.append(squared_res)

        next_original_obj = squared_res + (np.linalg.norm(x)**2)
        original_objs.append(next_original_obj)

        if early_stopping and (squared_res < tau*m):
            which_broke = "cgls"
            break

        # print()
        # print(prev_original_obj)
        # print(next_original_obj)

        # print("next > tau*prev?")
        # print(next_original_obj > tau*prev_original_obj)
        # print()

        if early_stopping and (next_original_obj > tau*prev_original_obj):
            which_broke = "original"
            break
        
        # Replace with latest
        prev_original_obj = next_original_obj


    data = {
        "x": x,
        "n_iterations": n_iterations,
        "squared_residuals": np.asarray(squared_residuals),
        "original_objectives": np.asarray(original_objs),
        "which_broke": which_broke,
    }

    return data


