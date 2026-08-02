"""
gwr_core.py — Implementacion CANONICA y UNICA del estimador GWR-Ridge del proyecto.
====================================================================================
Fuente unica de verdad para TODOS los scripts que usan GWR (modelo principal y
analisis de sensibilidad). Garantiza que sensibilidades, controles y tablas usen
EXACTAMENTE el mismo estimador que el modelo defendido (corrige el hallazgo #2 de la
3a auditoria adversarial: antes las sensibilidades usaban lambda=0.1 con intercepto
penalizado, distinto del modelo principal).

Especificacion canonica (2026-06-20):
  - Ridge con intercepto NO penalizado (estandar de Ridge; `_ridge_eye`).
  - lambda seleccionado por CV ESPACIAL ANIDADO dentro del training de cada fold
    (`select_lambda`): KMeans sobre coordenadas + leave-one-cluster-out, criterio MAE
    USD/m2. Sin leakage del test externo; la particion interna es espacial (refleja
    extrapolacion). lambda es consecuencia del procedimiento, no eleccion post hoc.
  - bandwidth bisquare adaptativo por golden-section/AICc (`select_bw`), en encoding
    continuo; prediccion en one-hot (el llamador decide el encoding que pasa).

Predecir y_hat(x0) = x0' beta(x0),  beta(x0)=(X'W(x0)X + lam*I')^-1 X'W(x0) y,
I' sin penalizar el intercepto.
"""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    from mgwr.sel_bw import Sel_BW
    MGWR_AVAILABLE = True
except ImportError:
    MGWR_AVAILABLE = False

# ── Hiperparametros canonicos ────────────────────────────────────────────────
LAMBDA_GRID        = [0.01, 0.1, 1.0, 10.0]   # grilla para el CV anidado de lambda
LAMBDA_RIDGE       = 1.0     # fallback si el fold es demasiado pequeno para nested CV
N_INNER_LAMBDA     = 3       # clusters espaciales (KMeans) del CV anidado de lambda
PENALIZE_INTERCEPT = False   # NO penalizar el intercepto (columna 0)
BW_MIN, BW_MAX, BW_FALLBACK = 40, 500, 199
RANDOM_STATE       = 42


def add_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


def _ridge_eye(p, penalize_intercept=PENALIZE_INTERCEPT):
    """Matriz de regularizacion Ridge; si penalize_intercept=False la columna 0
    (intercepto) NO se penaliza (estandar de Ridge)."""
    I_p = np.eye(p)
    if not penalize_intercept:
        I_p[0, 0] = 0.0
    return I_p


def select_bw(coords_tr, X_int, y_tr, verbose=False):
    """Bandwidth bisquare adaptativo por golden-section sobre AICc (mgwr)."""
    if not MGWR_AVAILABLE:
        return BW_FALLBACK
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = Sel_BW(coords_tr, y_tr.reshape(-1, 1), X_int, fixed=False, kernel="bisquare")
            bw = sel.search(bw_min=BW_MIN, bw_max=BW_MAX,
                            search_method="golden_section", criterion="AICc")
        return max(BW_MIN, int(round(float(bw))))
    except Exception as exc:
        if verbose:
            print(f"    Sel_BW fallo ({exc}) -> BW_FALLBACK={BW_FALLBACK}", flush=True)
        return BW_FALLBACK


def fit_gwr_ridge(coords_tr, X_int, y_tr, bw, lam=LAMBDA_RIDGE,
                  penalize_intercept=PENALIZE_INTERCEPT):
    """Coeficientes locales beta(x_i) en cada punto de training (para residuales in-sample)."""
    n, p = X_int.shape
    k_q = min(bw + 1, n)
    nbrs = NearestNeighbors(n_neighbors=k_q, algorithm="ball_tree").fit(coords_tr)
    h = nbrs.kneighbors(coords_tr)[0][:, -1]
    params = np.zeros((n, p)); n_rc = 0; I_p = _ridge_eye(p, penalize_intercept)
    for i in range(n):
        d = np.sqrt(((coords_tr - coords_tr[i]) ** 2).sum(axis=1))
        u = d / (h[i] + 1e-10)
        w = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        Xw = X_int * w[:, None]
        A = Xw.T @ X_int + lam * I_p
        b = X_int.T @ (w * y_tr)
        try:
            params[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            n_rc += 1
            try:
                params[i] = np.linalg.solve(Xw.T @ X_int + (lam * 10) * I_p, b)
            except Exception:
                params[i] = np.zeros(p)
    return params, n_rc


def predict_gwr(coords_tr, X_tr_int, y_tr, coords_te, X_te_int, bw,
                lam=LAMBDA_RIDGE, penalize_intercept=PENALIZE_INTERCEPT):
    """Prediccion GWR estandar: para cada x0 de test resuelve el sistema ponderado-ridge
    sobre el training con kernel bisquare adaptativo centrado en x0."""
    n_te = len(coords_te); p = X_tr_int.shape[1]; I_p = _ridge_eye(p, penalize_intercept)
    k_q = min(bw, len(coords_tr))
    nbrs = NearestNeighbors(n_neighbors=k_q, algorithm="ball_tree").fit(coords_tr)
    h_te = nbrs.kneighbors(coords_te)[0][:, -1]
    preds = np.zeros(n_te); n_rc = 0
    for j in range(n_te):
        d = np.sqrt(((coords_tr - coords_te[j]) ** 2).sum(axis=1))
        u = d / (h_te[j] + 1e-10)
        w = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        Xw = X_tr_int * w[:, None]
        A = Xw.T @ X_tr_int + lam * I_p
        b = X_tr_int.T @ (w * y_tr)
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            n_rc += 1
            try:
                beta = np.linalg.solve(Xw.T @ X_tr_int + (lam * 10) * I_p, b)
            except Exception:
                beta = np.zeros(p)
        preds[j] = X_te_int[j] @ beta
    return preds, n_rc


def select_lambda(coords_tr, X_tr_int, y_tr, bw, grid=LAMBDA_GRID,
                  penalize_intercept=PENALIZE_INTERCEPT, n_inner=N_INNER_LAMBDA,
                  seed=RANDOM_STATE):
    """CV ESPACIAL ANIDADO para lambda dentro del training del outer fold.
    KMeans(n_inner) sobre coordenadas + leave-one-cluster-out; criterio MAE USD/m2
    pooled. Sin leakage del test externo; particion interna espacial. Devuelve
    (lambda*, {lambda: MAE_inner})."""
    from sklearn.cluster import KMeans
    n = len(coords_tr)
    if n < 150:
        return LAMBDA_RIDGE, {}
    labels = KMeans(n_clusters=n_inner, random_state=seed, n_init=10).fit(coords_tr).labels_
    mae_by_lam = {}
    for lam in grid:
        errs = []
        for c in range(n_inner):
            te = labels == c; tr = ~te
            if te.sum() < 10 or tr.sum() < 50:
                continue
            pl, _ = predict_gwr(coords_tr[tr], X_tr_int[tr], y_tr[tr],
                                coords_tr[te], X_tr_int[te], bw,
                                lam=lam, penalize_intercept=penalize_intercept)
            errs.append(np.abs(np.exp(y_tr[te]) - np.exp(pl)))
        if errs:
            mae_by_lam[lam] = float(np.mean(np.concatenate(errs)))
    if not mae_by_lam:
        return LAMBDA_RIDGE, {}
    return min(mae_by_lam, key=mae_by_lam.get), mae_by_lam


def local_condition_diagnostics(coords_tr, X_tr_int, coords_te, X_te_int, bw,
                                lam=LAMBDA_RIDGE, penalize_intercept=PENALIZE_INTERCEPT):
    """Diagnostico de condicionamiento (hallazgo #4): numero de condicion y eigenvalor
    minimo de las matrices locales A=(X'W X + lam I') en cada punto de test, y |z|-score
    maximo de las covariables de test respecto a media/desv. del training. Devuelve dict."""
    p = X_tr_int.shape[1]; I_p = _ridge_eye(p, penalize_intercept)
    k_q = min(bw, len(coords_tr))
    nbrs = NearestNeighbors(n_neighbors=k_q, algorithm="ball_tree").fit(coords_tr)
    h_te = nbrs.kneighbors(coords_te)[0][:, -1]
    conds, eigmins = [], []
    for j in range(len(coords_te)):
        d = np.sqrt(((coords_tr - coords_te[j]) ** 2).sum(axis=1))
        u = d / (h_te[j] + 1e-10)
        w = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        Xw = X_tr_int * w[:, None]
        A = Xw.T @ X_tr_int + lam * I_p
        try:
            ev = np.linalg.eigvalsh(A); ev = ev[ev > 0]
            if len(ev):
                conds.append(ev.max() / ev.min()); eigmins.append(ev.min())
        except np.linalg.LinAlgError:
            pass
    # |z|-score de covariables test vs media/std de training (excluye intercepto col 0)
    mu = X_tr_int[:, 1:].mean(axis=0); sd = X_tr_int[:, 1:].std(axis=0) + 1e-12
    z = np.abs((X_te_int[:, 1:] - mu) / sd)
    return {"cond_median": float(np.median(conds)) if conds else float("nan"),
            "cond_p95": float(np.percentile(conds, 95)) if conds else float("nan"),
            "cond_max": float(np.max(conds)) if conds else float("nan"),
            "eig_min": float(np.min(eigmins)) if eigmins else float("nan"),
            "zmax_test": float(z.max())}
