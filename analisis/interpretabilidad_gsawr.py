"""
interpretabilidad_gsawr.py
==========================
Permutation importance + mapa de error para GSAWR (mejor modelo reportado).

GSAWR (Xu et al., 2025, IJGIS): SANNet(AFCNN+SAJPNN) + SAWCNN + OLS-weighted output.
Toma como entrada un tensor imagen (n, 1+p, 20, 20): canal 0 = distancia espacial,
canales 1..p = |X_scaled - grid_attrs_idw|. Permutar la variable k afecta tanto el
peso multiplicativo X_int[:,k+1] como los canales de atributo del tensor imagen.

Metodo:
  1. Reconstruir el preprocesamiento exacto de gsawr_log.py (grid, IDW, scalers
     por canal ajustados en train) y cargar output_log/models/gsawr_log_final.pt
  2. Prediccion base sobre holdout 20% -> RMSE_base
  3. Para cada variable k: barajar columna k en X_test, reconstruir X_int y los
     canales de atributo del tensor imagen, medir degradacion RMSE
  4. Permutation importance = RMSE_permuted - RMSE_base (5 repeticiones)
  5. Mapa de error: |y_true - y_pred| por predio, en espacio

Salidas: analisis/output_log/
  - interp_gsawr_permutation_importance.csv
  - interp_error_por_predio.csv
  - interp_gsawr_perm_importance.png / .pdf
  - interp_gsawr_mapa_error.png / .pdf
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
ROOT       = BASE_DIR.parent
sys.path.insert(0, str(ROOT / "modelos"))
from features import build_feature_matrix
OUT_DIR    = BASE_DIR / "output_log"
OUT_DIR.mkdir(exist_ok=True)

GSAWR_DIR  = ROOT / "modelos" / "gsawr"
MODEL_PATH = GSAWR_DIR / "output_log" / "models" / "gsawr_log_final.pt"
DATA_PATH  = ROOT / "datos" / "dataset.gpkg"
SPLIT_PATH = ROOT / "data_split" / "split.csv"

CRS_UTM = "EPSG:32717"

# Hiperparametros identicos a gsawr_log.py
GRID_COLS = 20; GRID_ROWS = 20; IDW_POWER = 2.0
AFCNN_HIDDEN = [16]; SAJPNN_HIDDEN = 4
CNN_CHANNELS = [16, 16, 32]; CNN_KERNEL = (5, 5); DENSE_LAYERS = [256, 64]
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_REPEATS = 5
RANDOM_STATE = 42

def pr(m=""): print(m, flush=True)

# ── Arquitectura (identica a gsawr_log.py) ────────────────────────────────────
class AFCNN(nn.Module):
    def __init__(self, n_feat, hidden, bias=True, batch_norm=True, activate_func=None):
        super().__init__()
        if activate_func is None: activate_func = nn.ReLU()
        layers = nn.Sequential(); in_ch = n_feat
        for i, out_ch in enumerate(hidden):
            layers.add_module(f"conv{i+1}", nn.Conv2d(in_ch, out_ch, kernel_size=(1,1), bias=bias))
            if batch_norm: layers.add_module(f"bn{i+1}", nn.BatchNorm2d(out_ch))
            layers.add_module(f"act{i+1}", activate_func); in_ch = out_ch
        layers.add_module("conv_out", nn.Conv2d(in_ch, 1, kernel_size=(1,1), bias=bias))
        layers.add_module("act_out", nn.ReLU())
        self.afcnn = layers
    def forward(self, a): return self.afcnn(a)

class SAJPNN(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden, bias=True), nn.PReLU(init=0.1),
            nn.Linear(hidden, 1, bias=True), nn.PReLU(init=0.1))
    def forward(self, x): return self.net(x)

class SANNet(nn.Module):
    def __init__(self, n_feat, afcnn_hidden, sajpnn_hidden, bias=True, batch_norm=True):
        super().__init__()
        self.afcnn  = AFCNN(n_feat, afcnn_hidden, bias, batch_norm)
        self.sajpnn = SAJPNN(sajpnn_hidden)
    def forward(self, dis):
        b, c, h, w = dis.size()
        s_mat = dis[:, 0:1, :, :]; a_mat = dis[:, 1:, :, :]
        a_fused = self.afcnn(a_mat)
        sa = torch.cat([s_mat, a_fused], dim=1)
        sa_flat = sa.permute(0,2,3,1).reshape(-1, 2)
        out = self.sajpnn(sa_flat)
        return out.view(b, 1, h, w)

class SAWCNN(nn.Module):
    def __init__(self, rows, cols, n_features, cnn_channels, kernel_size, dense_layers):
        super().__init__()
        conv_blocks = nn.Sequential(); in_ch = 1
        pad = (kernel_size[0]//2, kernel_size[1]//2)
        for i, out_ch in enumerate(cnn_channels):
            block = nn.Sequential()
            block.add_module("conv", nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=True))
            block.add_module("act", nn.PReLU(init=0.1))
            block.add_module("pool", nn.MaxPool2d(kernel_size=2))
            conv_blocks.add_module(f"block{i+1}", block); in_ch = out_ch
        self.conv_blocks = conv_blocks
        last_size = self._flat(rows, cols, cnn_channels)
        fc = nn.Sequential(); prev = last_size
        for i, h in enumerate(dense_layers):
            fc.add_module(f"fc{i}", nn.Linear(prev, h)); fc.add_module(f"act{i}", nn.ReLU()); prev = h
        fc.add_module("fc_out", nn.Linear(prev, n_features))
        self.fc = fc
    @staticmethod
    def _flat(rows, cols, channels):
        h, w = rows, cols
        for _ in channels: h //= 2; w //= 2
        return channels[-1] * h * w
    def forward(self, x):
        b = x.size(0)
        for block in self.conv_blocks: x = block(x)
        return self.fc(x.view(b, -1))

class GSAWRModel(nn.Module):
    def __init__(self, n_feat, n_features, rows, cols, afcnn_hidden, sajpnn_hidden,
                 cnn_channels, kernel_size, dense_layers, ols_coeff):
        super().__init__()
        self.sannet = SANNet(n_feat, afcnn_hidden, sajpnn_hidden)
        self.sawcnn = SAWCNN(rows, cols, n_features, cnn_channels, kernel_size, dense_layers)
        self.out = nn.Linear(n_features, 1, bias=False)
        self.out.weight = nn.Parameter(
            torch.tensor(ols_coeff.reshape(1,-1), dtype=torch.float32), requires_grad=False)
    def forward(self, dis, x):
        sa_map = self.sannet(dis)
        feat_w = self.sawcnn(sa_map)
        return self.out(feat_w * x)

# ── Helpers de preprocesamiento (identicos a gsawr_log.py) ────────────────────
def add_intercept(X): return np.column_stack([np.ones(len(X)), X])

def build_reference_grid(coords_train, cols, rows):
    x_min, y_min = coords_train.min(axis=0); x_max, y_max = coords_train.max(axis=0)
    dx = (x_max-x_min)*0.05; dy = (y_max-y_min)*0.05
    x_lin = np.linspace(x_min-dx, x_max+dx, cols); y_lin = np.linspace(y_min-dy, y_max+dy, rows)
    gx, gy = np.meshgrid(x_lin, y_lin)
    return np.column_stack([gx.ravel(), gy.ravel()])

def compute_grid_attrs_idw(coords_train, X_train_scaled, grid_xy, power=IDW_POWER):
    dists = cdist(grid_xy, coords_train, "euclidean")
    w = 1.0/(dists**power+1e-10); w /= w.sum(axis=1, keepdims=True)
    return w @ X_train_scaled

def raw_image(coords_query, X_query_scaled, grid_xy, grid_attrs, rows, cols):
    n = len(coords_query); n_feat = X_query_scaled.shape[1]
    d_spatial = cdist(coords_query, grid_xy, "euclidean").reshape(n, 1, rows, cols)
    delta = np.abs(X_query_scaled[:,np.newaxis,:] - grid_attrs[np.newaxis,:,:])
    delta = delta.transpose(0,2,1).reshape(n, n_feat, rows, cols)
    return np.concatenate([d_spatial, delta], axis=1)

def predict(model, dis_img, X_int, bs=BATCH_SIZE):
    model.eval(); preds = []
    dis_t = torch.tensor(dis_img, dtype=torch.float32)
    x_t   = torch.tensor(X_int,   dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_int), bs):
            yb = model(dis_t[i:i+bs].to(DEVICE), x_t[i:i+bs].to(DEVICE))
            preds.append(yb.cpu().numpy().flatten())
    return np.concatenate(preds)

def rmse_orig(y_true_orig, pred_log): return float(np.sqrt(np.mean((y_true_orig - np.exp(pred_log))**2)))

# ── Cargar datos ───────────────────────────────────────────────────────────────
pr("Cargando datos ...")
gdf = gpd.read_file(DATA_PATH, layer="puntos_mercado").to_crs(CRS_UTM)
gdf["predio_join"] = gdf["predio_join"].astype(int)
split_df = pd.read_csv(SPLIT_PATH); split_df["predio_join"] = split_df["predio_join"].astype(int)
gdf = gdf.merge(split_df[["predio_join","split"]], on="predio_join", how="left")
gdf = gdf.sort_values("predio_join").reset_index(drop=True)

coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
y_orig = gdf["valor_m2"].values.astype(np.float64)
y_log  = np.log(y_orig)
X_raw, FEAT_NAMES = build_feature_matrix(gdf)
train_mask = (gdf["split"]=="train").values; test_mask = (gdf["split"]=="test").values
pr(f"n_train={train_mask.sum()}  n_test={test_mask.sum()}")

# ── Reconstruir preprocesamiento del modelo final ─────────────────────────────
pr("Reconstruyendo preprocesamiento (grid + IDW + scalers por canal) ...")
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_raw[train_mask])
X_te_sc = scaler.transform(X_raw[test_mask])
X_tr_int = add_intercept(X_tr_sc); X_te_int = add_intercept(X_te_sc)
n_feat = X_tr_sc.shape[1]; n_features = X_tr_int.shape[1]

coords_tr = coords[train_mask]; coords_te = coords[test_mask]
grid_xy    = build_reference_grid(coords_tr, GRID_COLS, GRID_ROWS)
grid_attrs = compute_grid_attrs_idw(coords_tr, X_tr_sc, grid_xy)

# Scalers por canal ajustados en train (igual que build_image_tensors)
dis_tr_raw = raw_image(coords_tr, X_tr_sc, grid_xy, grid_attrs, GRID_ROWS, GRID_COLS)
n_channels = dis_tr_raw.shape[1]
channel_scalers = []
for ch in range(n_channels):
    sc = StandardScaler()
    sc.fit(dis_tr_raw[:, ch, :, :].reshape(-1, 1))
    channel_scalers.append(sc)

def scale_image(raw):
    out = np.empty_like(raw, dtype=np.float32)
    n = raw.shape[0]
    for ch in range(n_channels):
        out[:, ch, :, :] = channel_scalers[ch].transform(
            raw[:, ch, :, :].reshape(-1, 1)).reshape(n, GRID_ROWS, GRID_COLS)
    return out

dis_te = scale_image(raw_image(coords_te, X_te_sc, grid_xy, grid_attrs, GRID_ROWS, GRID_COLS))

# ── Cargar modelo ──────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    pr(f"ERROR: No se encontro {MODEL_PATH}\nCorre gsawr_log.py primero."); sys.exit(1)

ols_coeff = LinearRegression(fit_intercept=False).fit(X_tr_int, y_log[train_mask]).coef_.flatten().astype(np.float32)
model = GSAWRModel(n_feat, n_features, GRID_ROWS, GRID_COLS, AFCNN_HIDDEN, SAJPNN_HIDDEN,
                   CNN_CHANNELS, CNN_KERNEL, DENSE_LAYERS, ols_coeff).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state); model.eval()
pr(f"Modelo cargado: {MODEL_PATH.name}")

# ── Prediccion base ──────────────────────────────────────────────────────────
y_te_orig = y_orig[test_mask]
pred_base = predict(model, dis_te, X_te_int)
rmse_base = rmse_orig(y_te_orig, pred_base)
pr(f"RMSE base holdout: {rmse_base:.4f}")

# ── Permutation importance ────────────────────────────────────────────────────
pr(f"\nPermutation Importance ({N_REPEATS} reps x {len(FEAT_NAMES)} vars) ...")
rng = np.random.default_rng(RANDOM_STATE)
# Canal espacial (0) no cambia al permutar atributos -> precomputar una vez
d_spatial_te = cdist(coords_te, grid_xy, "euclidean").reshape(len(coords_te), 1, GRID_ROWS, GRID_COLS)

importance_rows = []
for k, var_name in enumerate(FEAT_NAMES):
    deltas = []
    for _ in range(N_REPEATS):
        X_perm = X_raw[test_mask].copy()
        X_perm[:, k] = rng.permutation(X_perm[:, k])
        Xp_sc  = scaler.transform(X_perm)
        Xp_int = add_intercept(Xp_sc)
        # Reconstruir canales de atributo (1..p); canal 0 fijo
        delta = np.abs(Xp_sc[:,np.newaxis,:] - grid_attrs[np.newaxis,:,:])
        delta = delta.transpose(0,2,1).reshape(len(coords_te), n_feat, GRID_ROWS, GRID_COLS)
        raw = np.concatenate([d_spatial_te, delta], axis=1)
        dis_perm = scale_image(raw)
        pred_perm = predict(model, dis_perm, Xp_int)
        deltas.append(rmse_orig(y_te_orig, pred_perm) - rmse_base)
    md = float(np.mean(deltas)); sd = float(np.std(deltas))
    importance_rows.append({"variable": var_name, "importance_mean": round(md,4),
                            "importance_std": round(sd,4), "rmse_perm_mean": round(rmse_base+md,4)})
    pr(f"  [{k+1:2d}/{len(FEAT_NAMES)}] {var_name:30s}  dRMSE={md:+.3f} +- {sd:.3f}")

imp_df = pd.DataFrame(importance_rows).sort_values("importance_mean", ascending=False)
imp_df.to_csv(OUT_DIR / "interp_gsawr_permutation_importance.csv", index=False)
pr("\nTop-10 variables:")
pr(imp_df.head(10)[["variable","importance_mean","importance_std"]].to_string(index=False))

# ── Error por predio ──────────────────────────────────────────────────────────
abs_error = np.abs(y_te_orig - np.exp(pred_base))
rel_error = abs_error / y_te_orig * 100
pd.DataFrame({
    "x_utm": coords_te[:,0], "y_utm": coords_te[:,1],
    "y_true": y_te_orig, "y_pred": np.exp(pred_base),
    "abs_error": abs_error, "rel_error": rel_error,
}).to_csv(OUT_DIR / "interp_error_por_predio.csv", index=False)

# ── Figura 1: Permutation importance ──────────────────────────────────────────
n_show = min(20, len(imp_df)); top = imp_df.head(n_show).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#E53935" if v > 0 else "#43A047" for v in top["importance_mean"]]
ax.barh(range(n_show), top["importance_mean"], xerr=top["importance_std"], capsize=3,
        color=colors, alpha=0.82, edgecolor="black", lw=0.5, error_kw={"elinewidth":1.0})
ax.axvline(0, color="black", lw=1.0)
ax.set_yticks(range(n_show)); ax.set_yticklabels(top["variable"], fontsize=9)
ax.set_xlabel("ΔRMSE (USD/m²) al permutar la variable", fontsize=11)
ax.set_title("Permutation Importance — GSAWR (Xu et al., 2025)\n"
             f"Holdout 20% (n=1,011)  |  5 repeticiones  |  RMSE base={rmse_base:.1f} USD/m²",
             fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(OUT_DIR / "interp_gsawr_perm_importance.png", dpi=180, bbox_inches="tight")
fig.savefig(OUT_DIR / "interp_gsawr_perm_importance.pdf", bbox_inches="tight")
plt.close(fig)
pr("  -> interp_gsawr_perm_importance.png / .pdf")

# ── Figura 2: Mapa de error espacial ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
ax = axes[0]
sc = ax.scatter(coords_te[:,0]/1000, coords_te[:,1]/1000, c=abs_error, cmap="RdYlGn_r",
                s=8, alpha=0.7, norm=LogNorm(vmin=max(1, abs_error.min()), vmax=abs_error.max()))
plt.colorbar(sc, ax=ax, label="Error absoluto USD/m² (log)")
ax.set_xlabel("Easting (km)", fontsize=10); ax.set_ylabel("Northing (km)", fontsize=10)
ax.set_title("Error Absoluto por Predio", fontsize=11, fontweight="bold"); ax.set_aspect("equal")
ax = axes[1]
p50 = np.percentile(abs_error, 50); p90 = np.percentile(abs_error, 90)
cat = np.where(abs_error <= p50, "bajo (<p50)",
        np.where(abs_error <= p90, "medio (p50-p90)", "alto (>p90)"))
cmap_cat = {"bajo (<p50)": "#1B7837", "medio (p50-p90)": "#FFA500", "alto (>p90)": "#D32F2F"}
for label, color in cmap_cat.items():
    mask = cat == label
    ax.scatter(coords_te[mask,0]/1000, coords_te[mask,1]/1000, c=color, s=8, alpha=0.7,
               label=f"{label} (n={mask.sum()})")
ax.set_xlabel("Easting (km)", fontsize=10); ax.set_ylabel("Northing (km)", fontsize=10)
ax.set_title("Percentil de Error por Zona", fontsize=11, fontweight="bold")
ax.set_aspect("equal"); ax.legend(fontsize=9, loc="upper right")
fig.suptitle("Mapa de Error Espacial — GSAWR (Xu et al., 2025)\n"
             f"Holdout 20% (n=1,011)  |  RMSE={rmse_base:.1f} USD/m²  |  p50={p50:.0f}  p90={p90:.0f}",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "interp_gsawr_mapa_error.png", dpi=180, bbox_inches="tight")
fig.savefig(OUT_DIR / "interp_gsawr_mapa_error.pdf", bbox_inches="tight")
plt.close(fig)
pr("  -> interp_gsawr_mapa_error.png / .pdf")
pr(f"\n[Listo] Resultados en {OUT_DIR}")
