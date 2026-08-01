import torch
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from sklearn.neighbors import NearestNeighbors

from .graph import build_graph
# https://github.com/RobinMagnet/pyFM/blob/master/pyFM/refine/icp.py


def knn_query(X, Y, k=1, return_distance=False,metric='minkowski', n_jobs=1, device=None):
    """
    Query nearest neighbors.

    Parameters
    -------------------------------
    X : (n1,p) first collection
    Y : (n2,p) second collection
    k : int - number of neighbors to look for
    return_distance : whether to return the nearest neighbor distance
    n_jobs          : number of parallel jobs. Set to -1 to use all processes
    device          : torch device for GPU computation (None for CPU path)

    Output
    -------------------------------
    dists   : (n2,k) or (n2,) if k=1 - ONLY if return_distance is False. Nearest neighbor distance.
    matches : (n2,k) or (n2,) if k=1 - nearest neighbor
    """
    if device is not None and str(device) != 'cpu':
        return _knn_query_gpu(X, Y, k=k, return_distance=return_distance, metric=metric, device=device)

    if metric=='cosine':
        algorithm="auto"
    else:
        algorithm="kd_tree"
    tree = NearestNeighbors(n_neighbors=k, leaf_size=40,metric=metric, algorithm=algorithm, n_jobs=n_jobs)
    tree.fit(X)
    dists, matches = tree.kneighbors(Y)

    if k == 1:
        dists = dists.squeeze()
        matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches


def _knn_query_gpu(X, Y, k=1, return_distance=False, metric='minkowski', device='cuda'):
    """GPU KNN using torch.cdist."""
    X_t = _to_torch(X, device)
    Y_t = _to_torch(Y, device)

    if metric == 'cosine':
        X_n = torch.nn.functional.normalize(X_t, p=2, dim=1)
        Y_n = torch.nn.functional.normalize(Y_t, p=2, dim=1)
        dists = 1.0 - Y_n @ X_n.T  # (n2, n1)
    else:
        dists = torch.cdist(Y_t, X_t, p=2)  # (n2, n1)

    topk_dists, topk_idx = torch.topk(dists, k, largest=False, dim=1)

    if k == 1:
        topk_dists = topk_dists.squeeze(1)
        topk_idx = topk_idx.squeeze(1)

    matches = topk_idx.cpu().numpy()
    if return_distance:
        return topk_dists.cpu().numpy(), matches
    return matches


def p2p_to_FM(p2p_21, evects1, evects2, A2=None, device=None):
    """
    Compute a Functional Map from a vertex to vertex maps (with possible subsampling).
    Can compute with the pseudo inverse of eigenvectors (if no subsampling) or least square.

    Parameters
    ------------------------------
    p2p_21    : (n2,) vertex to vertex map from target to source.
                For each vertex on the target shape, gives the index of the corresponding vertex on mesh 1.
                Can also be presented as a (n2,n1) sparse matrix.
    eigvects1 : (n1,k1) eigenvectors on source mesh. Possibly subsampled on the first dimension.
    eigvects2 : (n2,k2) eigenvectors on target mesh. Possibly subsampled on the first dimension.
    A2        : (n2,n2) area matrix of the target mesh. If specified, the eigenvectors can't be subsampled
    device    : torch device for GPU computation (None for CPU/numpy path)

    Outputs
    -------------------------------
    FM_12       : (k2,k1) functional map corresponding to the p2p map given.
                  Solved with pseudo inverse if A2 is given, else using least square.
    """
    if device is not None and str(device) != 'cpu':
        return _p2p_to_FM_gpu(p2p_21, evects1, evects2, A2=A2, device=device)

    # Pulled back eigenvectors
    evects1_pb = evects1[p2p_21, :] if np.asarray(p2p_21).ndim == 1 else p2p_21 @ evects1

    if A2 is not None:
        if A2.shape[0] != evects2.shape[0]:
            raise ValueError("Can't compute exact pseudo inverse with subsampled eigenvectors")

        if A2.ndim == 1:
            return evects2.T @ (A2[:, None] * evects1_pb)  # (k2,k1)

        return evects2.T @ (A2 @ evects1_pb)  # (k2,k1)

    # Solve with least square
    return scipy.linalg.lstsq(evects2, evects1_pb)[0]  # (k2,k1)


def _p2p_to_FM_gpu(p2p_21, evects1, evects2, A2=None, device='cuda'):
    """GPU implementation of p2p_to_FM using torch.linalg.lstsq."""
    evects1_t = _to_torch(evects1, device)
    evects2_t = _to_torch(evects2, device)
    p2p = np.asarray(p2p_21)

    if p2p.ndim == 1:
        evects1_pb = evects1_t[p2p, :]
    else:
        p2p_t = _to_torch(p2p, device)
        evects1_pb = p2p_t @ evects1_t

    if A2 is not None:
        A2_t = _to_torch(A2, device)
        if A2_t.shape[0] != evects2_t.shape[0]:
            raise ValueError("Can't compute exact pseudo inverse with subsampled eigenvectors")
        if A2_t.ndim == 1:
            return (evects2_t.T @ (A2_t[:, None] * evects1_pb)).cpu().numpy()
        return (evects2_t.T @ (A2_t @ evects1_pb)).cpu().numpy()

    # lstsq on GPU
    result = torch.linalg.lstsq(evects2_t, evects1_pb).solution
    return result.cpu().numpy()


def FM_to_p2p(FM_12, evects1, evects2, use_adj=False, n_jobs=1,k=1,metric='minkowski',return_distance=False, device=None):
    """
    Obtain a point to point map from a functional map C.
    Compares embeddings of dirac functions on the second mesh Phi_2.T with embeddings
    of dirac functions of the first mesh Phi_1.T

    Either one can transport the first diracs with the functional map or the second ones with
    the adjoint, which leads to different results (adjoint is the mathematically correct way)

    Parameters
    --------------------------
    FM_12     : (k2,k1) functional map from mesh1 to mesh2 in reduced basis
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
                First dimension can be subsampled.
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)
                First dimension can be subsampled.
    use_adj   : use the adjoint method
    n_jobs    : number of parallel jobs. Use -1 to use all processes
    device    : torch device for GPU computation (None for CPU path)


    Outputs:
    --------------------------
    p2p_21     : (n2,) match vertex i on shape 2 to vertex p2p_21[i] on shape 1,
                 or equivalent result if the eigenvectors are subsampled.
    """
    k2, k1 = FM_12.shape

    assert k1 <= evects1.shape[1], \
        f'At least {k1} should be provided, here only {evects1.shape[1]} are given'
    assert k2 <= evects2.shape[1], \
        f'At least {k2} should be provided, here only {evects2.shape[1]} are given'

    if use_adj:
        emb1 = evects1[:, :k1]
        emb2 = evects2[:, :k2] @ FM_12

    else:
        emb1 = evects1[:, :k1] @ FM_12.T
        emb2 = evects2[:, :k2]
    if return_distance:
        p2p_21_dist,p2p_21 = knn_query(emb1, emb2,  k=k, n_jobs=n_jobs,metric=metric,return_distance=return_distance, device=device)
        return p2p_21_dist,p2p_21

    else:
        p2p_21 = knn_query(emb1, emb2,  k=k, n_jobs=n_jobs,metric=metric,return_distance=return_distance, device=device)
    return p2p_21  # (n2,)


import torch
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

# source : https://github.com/RobinMagnet/pyFM/blob/master/pyFM/optimize/base_functions.py#L221

# p: number of points
# d: dimension of latent space
# b: dimension of basis (number of eigenvectors)

# descr preservation
def descr_preservation(C, descr1_red, descr2_red):
    """
    Compute the descriptor preservation constraint

    Parameters
    ---------------------
    C      : (b2, b1) Functional map
    descr1 : (b1,p) descriptors on first basis
    descr2 : (b2,p) descriptros on second basis

    Output
    ---------------------
    energy : descriptor preservation squared norm
    """
    return 0.5 * np.square(C @ descr1_red - descr2_red).sum()

def descr_preservation_grad(C, descr1_red, descr2_red):
    """
    Compute the gradient of the descriptor preservation constraint

    Parameters
    ---------------------
    C      : (b2,b1) Functional map
    descr1 : (b1,p) descriptors on first basis
    descr2 : (b2,p) descriptros on second basis

    Output
    ---------------------
    gradient : gradient of the descriptor preservation squared norm
    """
    return (C @ descr1_red - descr2_red) @ descr1_red.T

# Laplacian commutation
def LB_commutation(C, ev_sqdiff):
    """
    Compute the LB commutativity constraint

    Parameters
    ---------------------
    C      : (K2,K1) Functional map
    ev_sqdiff : (K2,K1) [normalized] matrix of squared eigenvalue differences

    Output
    ---------------------
    energy : (float) LB commutativity squared norm
    """
    return 0.5 * (np.square(C) * ev_sqdiff).sum()


def LB_commutation_grad(C, ev_sqdiff):
    """
    Compute the gradient of the LB commutativity constraint

    Parameters
    ---------------------
    C         : (K2,K1) Functional map
    ev_sqdiff : (K2,K1) [normalized] matrix of squared eigenvalue differences

    Output
    ---------------------
    gradient : (K2,K1) gradient of the LB commutativity squared norm
    """
    return C * ev_sqdiff

# operator commutation
def op_commutation(C, op1, op2):
    """
    Compute the operator commutativity constraint.
    Can be used with descriptor multiplication operator

    Parameters
    ---------------------
    C   : (K2,K1) Functional map
    op1 : (K1,K1) operator on first basis
    op2 : (K2,K2) descriptros on second basis

    Output
    ---------------------
    energy : (float) operator commutativity squared norm
    """
    return 0.5 * np.square(C @ op1 - op2 @ C).sum()


def op_commutation_grad(C, op1, op2):
    """
    Compute the gradient of the operator commutativity constraint.
    Can be used with descriptor multiplication operator

    Parameters
    ---------------------
    C   : (K2,K1) Functional map
    op1 : (K1,K1) operator on first basis
    op2 : (K2,K2) descriptros on second basis

    Output
    ---------------------
    gardient : (K2,K1) gradient of the operator commutativity squared norm
    """
    return op2.T @ (op2 @ C - C @ op1) - (op2 @ C - C @ op1) @ op1.T

# descriptor commutation
def oplist_commutation(C, op_list):
    """
    Compute the operator commutativity constraint for a list of pairs of operators
    Can be used with a list of descriptor multiplication operator

    Parameters
    ---------------------
    C   : (K2,K1) Functional map
    op_list : list of tuple( (K1,K1), (K2,K2) ) operators on first and second basis

    Output
    ---------------------
    energy : (float) sum of operators commutativity squared norm
    """
    energy = 0
    for (op1, op2) in op_list:
        energy += op_commutation(C, op1, op2)

    return energy


def oplist_commutation_grad(C, op_list):
    """
    Compute the gradient of the operator commutativity constraint for a list of pairs of operators
    Can be used with a list of descriptor multiplication operator

    Parameters
    ---------------------
    C   : (K2,K1) Functional map
    op_list : list of tuple( (K1,K1), (K2,K2) ) operators on first and second basis

    Output
    ---------------------
    gradient : (K2,K1) gradient of the sum of operators commutativity squared norm
    """
    gradient = 0
    for (op1, op2) in op_list:
        gradient += op_commutation_grad(C, op1, op2)
    return gradient

def compute_descr_op(eigvec1, eigvec2, descr1, descr2, k1, k2):
    """
    Compute the multiplication operators associated with the descriptors

    Output
    ---------------------------
    operators : n_descr long list of ((k1,k1),(k2,k2)) operators.
    """

    pinv1 = eigvec1[:, :k1].T # (k1,n)
    pinv2 = eigvec2[:, :k2].T # (k2,n)

    list_descr = [
                    (pinv1@(descr1[:, i, None] * eigvec1[:, :k1]),
                    pinv2@(descr2[:, i, None] * eigvec2[:, :k2])
                    )
                    for i in range(descr1.shape[1])
                ]

    return list_descr


# energy function
def energy_func_std(C, descr_mu, descr1_red, descr2_red, lap_mu, ev_sqdiff, descr_comm_mu, list_descr, orient_mu, orient_op):
    """
    Evaluation of the energy for standard FM computation

    min_C descr_mu * ||C@A - B||^2 + descr_comm_mu * (sum_i ||C@D_Ai - D_Bi@C||^2)
              + lap_mu * ||C@L1 - L2@C||^2 + orient_mu * (sum_i ||C@G_Ai - G_Bi@C||^2)

    Parameters:
    ----------------------
    C               : (b2*b1) or (b2,b1) Functional map
    descr_mu        : scaling of the descriptor preservation term
    descr1          : (b1,p) descriptors on first basis
    descr2          : (b2,p) descriptros on second basis
    lap_mu          : scaling of the laplacian commutativity term
    ev_sqdiff       : (K2,K1) [normalized] matrix of squared eigenvalue differences
    descr_comm_mu   : scaling of the descriptor commutativity term
    list_descr      : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to descriptors.
    orient_mu       : scaling of the orientation preservation term
    orient_op       : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to orientation preservation operators.

    Output
    ------------------------
    energy : float - value of the energy
    """
    k1 = descr1_red.shape[0]
    k2 = descr2_red.shape[0]
    C = C.reshape((k2,k1))

    energy = 0

    if descr_mu > 0:
        energy += descr_mu * descr_preservation(C, descr1_red, descr2_red)

    if lap_mu > 0:
        energy += lap_mu * LB_commutation(C, ev_sqdiff)

    if descr_comm_mu > 0:
        energy += descr_comm_mu * oplist_commutation(C, list_descr)

    if orient_mu > 0:
        energy += orient_mu * oplist_commutation(C, orient_op)

    return energy



def grad_energy_std(C, descr_mu, descr1_red, descr2_red, lap_mu, ev_sqdiff, descr_comm_mu, list_descr, orient_mu, orient_op):
    """
    Evaluation of the gradient of the energy for standard FM computation

    Parameters:
    ----------------------
    C               : (b2*b1) or (b2,b1) Functional map
    descr_mu        : scaling of the descriptor preservation term
    descr1          : (b1,p) descriptors on first basis
    descr2          : (b2,p) descriptros on second basis
    lap_mu          : scaling of the laplacian commutativity term
    ev_sqdiff       : (K2,K1) [normalized] matrix of squared eigenvalue differences
    descr_comm_mu   : scaling of the descriptor commutativity term
    list_descr      : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to descriptors.
    orient_mu       : scaling of the orientation preservation term
    orient_op       : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to orientation preservation operators.

    Output
    ------------------------
    gradient : (b2*b1) - value of the energy
    """
    k1 = descr1_red.shape[0]
    k2 = descr2_red.shape[0]
    C = C.reshape((k2,k1))

    gradient = np.zeros_like(C)

    if descr_mu > 0:
        gradient += descr_mu * descr_preservation_grad(C, descr1_red, descr2_red)

    if lap_mu > 0:
        gradient += lap_mu * LB_commutation_grad(C, ev_sqdiff)

    if descr_comm_mu > 0:
        gradient += descr_comm_mu * oplist_commutation_grad(C, list_descr)

    if orient_mu > 0:
        gradient += orient_mu * oplist_commutation_grad(C, orient_op)

    gradient[:,0] = 0
    return gradient.reshape(-1)


def get_C0(k1, k2, eigvec1=None, eigvec2=None, optinit="zeros"):
        """
        Returns the initial functional map for optimization.

        Parameters
        ------------------------
        optinit : 'random' | 'identity' | 'zeros' initialization.
                  In any case, the first column of the functional map is computed by hand
                  and not modified during optimization

        Output
        ------------------------
        x0 : corresponding initial vector
        """
        if optinit == 'random':
            x0 = np.random.random((k2, k1))
        elif optinit == 'identity':
            x0 = np.eye(k2, k1)
        else:
            x0 = np.zeros((k2, k1))

        if eigvec1 is not None and eigvec2 is not None:
            # Sets the equivalence between the constant functions
            ev_sign = np.sign(eigvec1[0,0]*eigvec2[0,0])
            # area_ratio = np.sqrt(mesh2.area/mesh1.area)

            x0[:,0] = np.zeros(k2)
            x0[0,0] = ev_sign # * area_ratio

        return x0


def rfm(d1: torch.Tensor, d2: torch.Tensor, eigvec1, eigvec2, evals1, evals2, k1=0, k2=0, w_descr=1e-1, w_lap=1e-3, w_dcomm=1, w_orient=0, optinit="zeros", device=None):
        """Computes functional map from v1 to v2.

        Args:
            d1: (n1,nd) tensor, descriptors of v1
            d2: (n2,nd) tensor, descriptors of v2
            eigvec1: (n1, k1) tensor, eigenvectors of the first shape
            eigvec2: (n2, k2) tensor, eigenvectors of the second shape
            evals1: (k1,) tensor, eigenvalues of the first shape
            evals2: (k2,) tensor, eigenvalues of the second shape
            k1: int, number of eigenvectors to use for the first shape
            k2: int, number of eigenvectors to use for the second shape
            device: torch device for GPU computation (None for CPU/numpy path)
        Returns:
            (b2, b1) numpy array, functional map from v1 to v2
        """

        if k1 == 0:
            k1 = eigvec1.shape[1]
        if k2 == 0:
            k2 = eigvec2.shape[1]

        if device is not None and str(device) != 'cpu':
            return _rfm_gpu(d1, d2, eigvec1, eigvec2, evals1, evals2, k1, k2,
                            w_descr, w_lap, w_dcomm, w_orient, optinit, device)

        # --- CPU path (original) ---
        C0 = get_C0(k1, k2, optinit=optinit,eigvec1=eigvec1,eigvec2=eigvec2)

        dspec1 = eigvec1[:, :k1].T @ d1
        dspec2 = eigvec2[:, :k2].T @ d2

        list_descr = []
        if w_dcomm > 0:
            list_descr = compute_descr_op(eigvec1, eigvec2, d1, d2, k1, k2)

        ev_sqdiff = np.square(evals1[None, :k1] - evals2[:k2, None])
        ev_sqdiff /= ev_sqdiff.sum()

        orient_op = []

        args = (w_descr, dspec1, dspec2, w_lap, ev_sqdiff, w_dcomm, list_descr, w_orient, orient_op)

        res = fmin_l_bfgs_b(energy_func_std, C0.ravel(), fprime=grad_energy_std, args=args)

        FM = res[0].reshape((k2, k1))

        return FM


def _to_torch(arr, device):
    if isinstance(arr, torch.Tensor):
        return arr.float().to(device)
    return torch.tensor(np.asarray(arr), dtype=torch.float32, device=device)


def _rfm_gpu(d1, d2, eigvec1, eigvec2, evals1, evals2, k1, k2,
             w_descr, w_lap, w_dcomm, w_orient, optinit, device):
    """GPU implementation of rfm using torch.optim.LBFGS."""
    # Move everything to device
    eigvec1_t = _to_torch(eigvec1, device)
    eigvec2_t = _to_torch(eigvec2, device)
    d1_t = _to_torch(d1, device)
    d2_t = _to_torch(d2, device)
    evals1_t = _to_torch(evals1, device).flatten()
    evals2_t = _to_torch(evals2, device).flatten()

    # Spectral descriptors
    dspec1 = eigvec1_t[:, :k1].T @ d1_t  # (k1, nd)
    dspec2 = eigvec2_t[:, :k2].T @ d2_t  # (k2, nd)

    # Descriptor multiplication operators
    list_descr_t = []
    if w_dcomm > 0:
        pinv1 = eigvec1_t[:, :k1].T
        pinv2 = eigvec2_t[:, :k2].T
        for i in range(d1_t.shape[1]):
            op1 = pinv1 @ (d1_t[:, i:i+1] * eigvec1_t[:, :k1])
            op2 = pinv2 @ (d2_t[:, i:i+1] * eigvec2_t[:, :k2])
            list_descr_t.append((op1, op2))

    # Eigenvalue squared differences
    ev_sqdiff_t = (evals1_t[None, :k1] - evals2_t[:k2, None]).square()
    ev_sqdiff_t = ev_sqdiff_t / (ev_sqdiff_t.sum() + 1e-12)

    # Initialize C
    if optinit == 'identity':
        C = torch.eye(k2, k1, device=device, dtype=torch.float32)
    elif optinit == 'random':
        C = torch.rand(k2, k1, device=device, dtype=torch.float32)
    else:
        C = torch.zeros(k2, k1, device=device, dtype=torch.float32)

    # Set first column for constant function equivalence
    ev_sign = torch.sign(eigvec1_t[0, 0] * eigvec2_t[0, 0])
    C[:, 0] = 0.0
    C[0, 0] = ev_sign

    C = C.requires_grad_(True)

    optimizer = torch.optim.LBFGS([C], max_iter=1, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        energy = torch.tensor(0.0, device=device)

        if w_descr > 0:
            energy = energy + w_descr * 0.5 * (C @ dspec1 - dspec2).square().sum()

        if w_lap > 0:
            energy = energy + w_lap * 0.5 * (C.square() * ev_sqdiff_t).sum()

        if w_dcomm > 0:
            for (op1, op2) in list_descr_t:
                energy = energy + w_dcomm * 0.5 * (C @ op1 - op2 @ C).square().sum()

        energy.backward()

        # Fix first column gradient to 0 (same as CPU path)
        if C.grad is not None:
            C.grad[:, 0] = 0.0

        return energy

    for _ in range(100):
        optimizer.step(closure)

    FM = C.detach().cpu().numpy()
    return FM

## Refinement

from tqdm import tqdm

# def icp_refine(self, nit=10, tol=None, use_adj=False, overwrite=True, verbose=False, n_jobs=1):
#     """
#     Refines the functional map using ICP and saves the result

#     Parameters
#     -------------------
#     nit       : int - number of iterations of icp to apply
#     tol       : float - threshold of change in functional map in order to stop refinement
#                 (only applies if nit is None)
#     overwrite : bool - If True changes FM type to 'icp' so that next call of self.FM
#                 will be the icp refined FM
#     """
#     if not self.fitted:
#         raise ValueError("The Functional map must be fit before refining it")

#     self._FM_icp = pyFM.refine.mesh_icp_refine(self.FM, self.mesh1, self.mesh2, nit=nit, tol=tol,
#                                                 use_adj=use_adj, n_jobs=n_jobs, verbose=verbose)

#     if overwrite:
#         self.FM_type = 'icp'

def farthest_point_sampling_call(d_func, k, n_points=None, verbose=False):
    """
    Samples points using farthest point sampling, initialized randomly

    Parameters
    -------------------------
    d_func   : callable - for index i, d_func(i) is a (n_points,) array of geodesic distance to
               other points
    k        : int - number of points to sample
    n_points : Number of points. If not specified, checks d_func(0)

    Output
    --------------------------
    fps : (k,) array of indices of sampled points
    """
    rng = np.random.default_rng()

    if n_points is None:
        n_points = d_func(0).shape

    else:
        assert n_points > 0

    inds = [rng.integers(n_points).item(0)]
    dists = d_func(inds[0])

    iterable = range(k-1) if not verbose else tqdm(range(k))
    for i in iterable:
        if i == k-1:
            continue
        newid = np.argmax(dists)
        inds.append(newid)
        dists = np.minimum(dists, d_func(newid))

    # print(inds)
    return np.asarray(inds)

#REGION ZoomOut
def zoomout_iteration(FM_12, evects1, evects2, step=1, A2=None, n_jobs=1, device=None):
    """
    Performs an iteration of ZoomOut.

    Parameters
    --------------------
    FM_12    : (k2,k1) Functional map from evects1[:,:k1] to evects2[:,:k2]
    evects1  : (n1,k1') eigenvectors on source shape with k1' >= k1 + step.
                 Can be a subsample of the original ones on the first dimension.
    evects2  : (n2,k2') eigenvectors on target shape with k2' >= k2 + step.
                 Can be a subsample of the original ones on the first dimension.
    step     : int - step of increase of dimension.
    A2       : (n2,n2) sparse area matrix on target mesh, for vertex to vertex computation.
                 If specified, the eigenvectors can't be subsampled !
    device   : torch device for GPU computation (None for CPU path)

    Output
    --------------------
    FM_zo : zoomout-refined functional map
    """
    k2, k1 = FM_12.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step
    new_k1, new_k2 = k1 + step1, k2 + step2

    p2p_21 = FM_to_p2p(FM_12, evects1, evects2, n_jobs=n_jobs, device=device)  # (n2,)
    # Compute the (k2+step, k1+step) FM
    FM_zo = p2p_to_FM(p2p_21, evects1[:, :new_k1], evects2[:, :new_k2], A2=A2, device=device)

    return FM_zo

def zoomout_refine(FM_12, evects1, evects2, nit=10, step=1, A2=None, subsample=None,
                   return_p2p=False, n_jobs=1, verbose=False, device=None):
    """
    Refine a functional map with ZoomOut.
    Supports subsampling for each mesh, different step size, and approximate nearest neighbor.

    Parameters
    --------------------
    eigvects1  : (n1,k1) eigenvectors on source shape with k1 >= K + nit
    eigvects2  : (n2,k2) eigenvectors on target shape with k2 >= K + nit
    FM_12      : (K,K) Functional map from from shape 1 to shape 2
    nit        : int - number of iteration of zoomout
    step       : increase in dimension at each Zoomout Iteration
    A2         : (n2,n2) sparse area matrix on target mesh.
    subsample  : tuple or iterable of size 2. Each gives indices of vertices to sample
                 for faster optimization. If not specified, no subsampling is done.
    return_p2p : bool - if True returns the vertex to vertex map.
    device     : torch device for GPU computation (None for CPU path)

    Output
    --------------------
    FM_12_zo  : zoomout-refined functional map from basis 1 to 2
    p2p_21_zo : only if return_p2p is set to True - the refined pointwise map from basis 2 to basis 1
    """
    k2_0, k1_0 = FM_12.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step

    assert k1_0 + nit*step1 <= evects1.shape[1], \
        f"Not enough eigenvectors on source : \
        {k1_0 + nit*step1} are needed when {evects1.shape[1]} are provided"
    assert k2_0 + nit*step2 <= evects2.shape[1], \
        f"Not enough eigenvectors on target : \
        {k2_0 + nit*step2} are needed when {evects2.shape[1]} are provided"

    use_subsample = False
    if subsample is not None:
        use_subsample = True
        sub1, sub2 = subsample

    FM_12_zo = FM_12.copy()

    iterable = range(nit) if not verbose else tqdm(range(nit))
    for it in iterable:
        if use_subsample:
            FM_12_zo = zoomout_iteration(FM_12_zo, evects1[sub1], evects2[sub2], A2=None,
                                         step=step, n_jobs=n_jobs, device=device)

        else:
            FM_12_zo = zoomout_iteration(FM_12_zo, evects1, evects2, A2=A2,
                                         step=step, n_jobs=n_jobs, device=device)

    if return_p2p:
        p2p_21_zo = FM_to_p2p(FM_12_zo, evects1, evects2, n_jobs=n_jobs, device=device)  # (n2,)
        return FM_12_zo, p2p_21_zo

    return FM_12_zo

def graph_zoomout_refine(FM_12, evects1, evects2, G1=None, G2=None, nit=10, step=1, subsample=None, return_p2p=False, n_jobs=1, verbose=False, device=None):
    """
    Refines the functional map using ZoomOut and saves the result

    Parameters
    -------------------
    FM_12      : (K,K) Functional map from from shape 1 to shape 2
    eigvects1  : (n1,k1) eigenvectors on source shape with k1 >= K + nit
    eigvects2  : (n2,k2) eigenvectors on target shape with k2 >= K + nit
    nit       : int - number of iterations to do
    step      : increase in dimension at each Zoomout Iteration
    subsample : int - number of points to subsample for ZoomOut. If None or 0, no subsampling is done.
    return_p2p : bool - if True returns the vertex to vertex map.
    overwrite : bool - If True changes FM type to 'zoomout' so that next call of self.FM
                will be the zoomout refined FM (larger than the other 2)
    device    : torch device for GPU computation (None for CPU path)
    """
    if subsample is None or subsample == 0 or G1 is None or G2 is None:
        sub = None
    else:
        sub1 = G1.extract_fps(subsample)
        sub2 = G2.extract_fps(subsample)
        sub = (sub1,sub2)

    _FM_zo = zoomout_refine(FM_12, evects1, evects2, nit,step=step, subsample=sub,
                   return_p2p=False, n_jobs=1, verbose=verbose, device=device)

    return _FM_zo
#ENDREGION

# EVALUATION
# https://github.com/RobinMagnet/pyFM/blob/master/pyFM/eval/evaluate.py


def FM_geod_err(p2p, gt_p2p, D1_geod, return_all=False, sqrt_area=None):
    """
    Computes the geodesic accuracy of a vertex to vertex map. The map goes from
    the target shape to the source shape.

    Parameters
    ----------------------
    p2p        : (n2,) - vertex to vertex map giving the index of the matched vertex on the source shape
                 for each vertex on the target shape (from a functional map point of view)
    gt_p2p     : (n2,) - ground truth mapping between the pairs
    D1_geod    : (n1,n1) - geodesic distance between pairs of vertices on the source mesh
    return_all : bool - whether to return all the distances or only the average geodesic distance

    Output
    -----------------------
    acc   : float - average accuracy of the vertex to vertex map
    dists : (n2,) - if return_all is True, returns all the pairwise distances
    """

    dists = D1_geod[(p2p,gt_p2p)]
    if sqrt_area is not None:
        dists /= sqrt_area

    if return_all:
        return dists.mean(), dists

    return dists.mean()


def FM_continuity(p2p, D1_geod, D2_geod, edges):
    """
    Computes continuity of a vertex to vertex map. The map goes from
    the target shape to the source shape.

    Parameters
    ----------------------
    p2p     : (n2,) - vertex to vertex map giving the index of the matched vertex on the source shape
                 for each vertex on the target shape (from a functional map point of view)
    gt_p2p  : (n2,) - ground truth mapping between the pairs
    D1_geod : (n1,n1) - geodesic distance between pairs of vertices on the source mesh
    D2_geod : (n1,n1) - geodesic distance between pairs of vertices on the target mesh
    edges   : (n2,2) edges on the target shape

    Output
    -----------------------
    continuity : float - average continuity of the vertex to vertex map
    """
    source_len = D2_geod[(edges[:,0], edges[:,1])]
    target_len = D1_geod[(p2p[edges[:,0]], p2p[edges[:,1]])]

    continuity = np.mean(target_len / source_len)

    return continuity


def FM_coverage(p2p, A):
    """
    Computes coverage of a vertex to vertex map. The map goes from
    the target shape to the source shape.

    Parameters
    ----------------------
    p2p : (n2,) - vertex to vertex map giving the index of the matched vertex on the source shape
                 for each vertex on the target shape (from a functional map point of view)
    A   : (n1,n1) or (n1,) - area matrix on the source shape or array of per-vertex areas.

    Output
    -----------------------
    coverage : float - coverage of the vertex to vertex map
    """
    if len(A.shape) == 2:
        vert_area = np.asarray(A.sum(1)).flatten()
    coverage = vert_area[np.unique(p2p)].sum() / vert_area.sum()

    return coverage
