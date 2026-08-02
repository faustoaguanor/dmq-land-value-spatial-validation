"""
spatial_cv_strategies.py
========================
Tres estrategias de validacion cruzada espacial — Roberts et al. (2017)

  Estrategia 1 | RandomKFoldCV   — K-Fold aleatorio (baseline)
  Estrategia 2 | SpatialBlockCV  — Bloques espaciales + checkerboard
  Estrategia 3 | BufferedLOO     — Leave-One-Out con buffer de exclusion

Las tres heredan de sklearn.model_selection.BaseCrossValidator y son
compatibles con cualquier flujo de scikit-learn:

    cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    GridSearchCV(model, param_grid, cv=cv)

Tambien incluye:
  - estimate_autocorrelation_range() : calcula el rango del semivariograma
  - run_spatial_cv()                 : ciclo CV con MAE, RMSE, MAPE por fold

Referencia:
  Roberts, M. et al. (2017). "Cross-validation strategies for data with
  temporal, spatial, hierarchical, or phylogenetic structure."
  Ecography, 40(8), 913–929. https://doi.org/10.1111/ecog.02881
"""

from __future__ import annotations

from typing import Generator, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.model_selection import BaseCrossValidator, KFold
import skgstat as skg


# =============================================================================
# Funcion auxiliar: estimacion del rango de autocorrelacion espacial
# =============================================================================

def estimate_autocorrelation_range(
    coords: np.ndarray,
    values: np.ndarray,
    models: Tuple[str, ...] = ("spherical", "exponential", "gaussian"),
    n_lags: int = 25,
) -> Tuple[float, str]:
    """
    Estima el rango de autocorrelacion espacial mediante semivariograma
    empirico. Equivalente Python de autofitVariogram() de R (paquete automap).

    Prueba multiples modelos y selecciona el de menor RMSE de ajuste.
    El rango es la distancia a partir de la cual la autocorrelacion espacial
    se reduce a cero — es el parametro clave para dimensionar los bloques
    en SpatialBlockCV y el buffer en BufferedLOO.

    Parametros
    ----------
    coords  : np.ndarray, shape (n, 2)
        Coordenadas (x, y) en metros (CRS proyectado, e.g. EPSG:32717).
    values  : np.ndarray, shape (n,)
        Variable objetivo (e.g. valor_m2).
    models  : tuple de str
        Modelos de variograma a probar. Por defecto: spherical, exponential,
        gaussian.
    n_lags  : int
        Numero de lags del semivariograma empirico.

    Retorna
    -------
    range_m    : float — rango de autocorrelacion en metros
    best_model : str   — nombre del modelo con menor RMSE
    """
    best_rmse, best_range, best_model = np.inf, None, None

    for model in models:
        try:
            V = skg.Variogram(
                coordinates=coords,
                values=values,
                model=model,
                maxlag="median",
                n_lags=n_lags,
            )
            if V.rmse < best_rmse:
                best_rmse  = V.rmse
                best_range = V.parameters[0]
                best_model = model
        except Exception:
            continue

    if best_range is None:
        raise RuntimeError(
            "No se pudo ajustar ningun modelo de semivariograma. "
            "Verifica que coords este en metros (CRS proyectado)."
        )

    return float(best_range), str(best_model)


# =============================================================================
# Estrategia 1: Random K-Fold (baseline)
# =============================================================================

class RandomKFoldCV(BaseCrossValidator):
    """
    Estrategia 1 (baseline): Random K-Fold.

    Wrapper sobre sklearn.model_selection.KFold que hereda de
    BaseCrossValidator para ser intercambiable con SpatialBlockCV
    y BufferedLOO en cualquier flujo de scikit-learn.

    Se usa UNICAMENTE como linea de base para demostrar como el CV
    aleatorio subestima el error real en datos espacialmente
    correlacionados ("structural overfitting", Roberts et al. 2017).
    Las observaciones cercanas en el espacio quedan tanto en train como
    en test, lo que permite al modelo "hacer trampa" memorizando la
    estructura espacial.

    Parametros
    ----------
    n_splits     : int  — numero de folds (default: 5)
    shuffle      : bool — mezclar antes de dividir (default: True)
    random_state : int  — semilla de aleatoriedad (default: 42)

    Ejemplo
    -------
    >>> cv = RandomKFoldCV(n_splits=5)
    >>> for train_idx, test_idx in cv.split(X):
    ...     model.fit(X[train_idx], y[train_idx])
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> None:
        self._n_splits    = n_splits
        self.shuffle      = shuffle
        self.random_state = random_state
        self._kfold       = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        return self._n_splits

    def _iter_test_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[np.ndarray]:
        n = len(X) if X is not None else 0
        for _, test_idx in self._kfold.split(np.arange(n)):
            yield test_idx

    def __repr__(self) -> str:
        return (
            f"RandomKFoldCV(n_splits={self._n_splits}, "
            f"shuffle={self.shuffle}, random_state={self.random_state})"
        )


# =============================================================================
# Estrategia 2: Spatial Block K-Fold con patron checkerboard
# =============================================================================

class SpatialBlockCV(BaseCrossValidator):
    """
    Estrategia 2: Spatial Block K-Fold con patron checkerboard.

    Implementacion de la metodologia de bloques espaciales de Roberts et al.
    (2017). Divide el area de estudio en una cuadricula regular y asigna
    bloques completos a cada fold, buscando que train y test queden separados
    por distancias del orden de block_size. NO garantiza una distancia minima
    punto-a-punto: los bordes de bloques contiguos pueden quedar mucho mas cerca
    (diagnostico: minimos ~37.5 m). Solo SpatialBlockBufferedCV garantiza separacion.

    Logica del patron checkerboard
    ------------------------------
    Se superpone una cuadricula sobre el area de estudio con:
        block_size = rango_autocorrelacion (del semivariograma)

    Cada punto se asigna a su celda:
        block_col = floor((x - x_min) / block_size)
        block_row = floor((y - y_min) / block_size)

    Los bloques se distribuyen a K folds con formula sistematica:
        fold = (block_col + block_row) % K

    Este patron produce franjas diagonales intercaladas (tablero de ajedrez
    extendido a K colores) que:
      - Maximizan la separacion espacial entre folds vecinos.
      - Evitan aislar por completo zonas extremas del mapa (la asignacion
        diagonal distribuye los folds en toda la extension geografica).
      - Mitigan el riesgo de extrapolacion ambiental pura, ya que cada fold
        contiene observaciones de distintas zonas del DMQ.

    Equivalente R (Box 4, grid.fun):
        m.init  <- matrix(c(T,F), dim.3, dim.2)   # patron binario
        m.full  <- expand_to_cell_size(m.init)     # expandir
        fold_id <- (block_col + block_row) %% K    # K folds

    Los folds deben calcularse previamente en spatial_cv_pipeline.py
    y pasarse al constructor.

    Parametros
    ----------
    folds : np.ndarray, shape (n_samples,)
        Etiqueta de fold (0 ... K-1) para cada observacion.
        Generado por: fold = (block_col + block_row) % K

    Ejemplo
    -------
    >>> folds = pd.read_csv("output/fold_assignments.csv")["fold"].values
    >>> cv = SpatialBlockCV(folds=folds)
    >>> for train_idx, test_idx in cv.split(X):
    ...     model.fit(X[train_idx], y[train_idx])
    """

    def __init__(self, folds: np.ndarray) -> None:
        self.folds   = np.asarray(folds, dtype=int)
        self._unique = np.unique(self.folds)

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        return len(self._unique)

    def _iter_test_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[np.ndarray]:
        """
        En cada iteracion, un fold geografico completo actua como hold-out.
        Los puntos del fold de test quedan separados de los de entrenamiento por
        distancias del orden de block_size en promedio; esto reduce (no elimina)
        la proximidad y NO garantiza independencia espacial.
        """
        all_idx = np.arange(len(self.folds))
        for k in self._unique:
            yield all_idx[self.folds == k]

    def __repr__(self) -> str:
        counts = [int((self.folds == k).sum()) for k in self._unique]
        return (
            f"SpatialBlockCV(k={len(self._unique)}, "
            f"n_total={len(self.folds)}, "
            f"fold_sizes={counts})"
        )


# =============================================================================
# Estrategia 2b: Spatial Block K-Fold con BUFFER de exclusion (dead-zone)
# =============================================================================

class SpatialBlockBufferedCV(BaseCrossValidator):
    """
    SpatialBlock K-Fold con zona de exclusion (buffer/dead-zone), siguiendo la
    recomendacion EXPLICITA de Roberts et al. (2017, Box 1 y 4) para GARANTIZAR
    separacion espacial train-test a nivel de cada punto individual.

    Problema que corrige
    --------------------
    En SpatialBlockCV el patron checkerboard pone bloques contiguos en folds
    distintos, por lo que dos puntos a ambos lados de un borde de bloque pueden
    quedar a ~0 m (en el DMQ: minimo observado 37.5 m). El tamano de bloque
    controla el tamano de las regiones homogeneas, NO la separacion train-test.

    Solucion
    --------
    Para cada fold de test se ELIMINAN del entrenamiento todos los puntos que
    esten a <= buffer metros de CUALQUIER punto de test. Garantiza:
        min_{i in train, j in test} dist(i, j) > buffer

    Eleccion del buffer
    -------------------
    Debe ser >= rango de autocorrelacion de los RESIDUOS (serie estacionaria),
    no del target crudo. En el DMQ el rango residual (variograma exponencial
    sobre residuos OLS) es ~2.5 km; mas alla, la dependencia espacial se anula
    y no hay fuga de informacion. buffer=2530 m elimina la autocorrelacion
    remanente reteniendo ~60-80% del training (viable). Buffers >> rango
    (p.ej. 5.6 km, rango del target crudo inflado por la tendencia) inanizan el
    training (folds con <600 obs) sin ganar independencia adicional.

    El conjunto de TEST de cada fold es identico al de SpatialBlockCV; solo se
    reduce el TRAIN. Asi la comparacion es "mismo test, training espacialmente
    independiente".

    Parametros
    ----------
    folds  : np.ndarray (n,)    etiqueta de bloque/fold por observacion.
    coords : np.ndarray (n, 2)  coordenadas (x, y) en metros (CRS proyectado).
    buffer : float              radio de exclusion en metros (>= rango residual).
    """

    def __init__(self, folds: np.ndarray, coords: np.ndarray,
                 buffer: float) -> None:
        self.folds   = np.asarray(folds, dtype=int)
        self.coords  = np.asarray(coords, dtype=float)
        self.buffer  = float(buffer)
        self._unique = np.unique(self.folds)
        if len(self.coords) != len(self.folds):
            raise ValueError("coords y folds deben tener la misma longitud")

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self._unique)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        all_idx = np.arange(len(self.folds))
        for k in self._unique:
            yield all_idx[self.folds == k]

    def split(self, X=None, y=None, groups=None):
        """Genera (train_idx, test_idx) con separacion garantizada > buffer."""
        all_idx = np.arange(len(self.folds))
        for k in self._unique:
            test_idx   = all_idx[self.folds == k]
            train_full = all_idx[self.folds != k]
            # Excluir del train todo punto a <= buffer de cualquier test
            d, _ = KDTree(self.coords[test_idx]).query(self.coords[train_full], k=1)
            train_idx = train_full[d > self.buffer]
            yield train_idx, test_idx

    def __repr__(self) -> str:
        return (f"SpatialBlockBufferedCV(k={len(self._unique)}, "
                f"n_total={len(self.folds)}, buffer={self.buffer:.0f} m)")


# =============================================================================
# Estrategia 3: Buffered Leave-One-Out
# =============================================================================

class BufferedLOO(BaseCrossValidator):
    """
    Estrategia 3: Buffered Leave-One-Out (LOO).

    Basado en Roberts et al. (2017), Box 1 y Box 4. Variante del LOO clasico
    que anade un radio de exclusion (buffer) alrededor del punto de prueba
    para garantizar una separacion geometrica minima superior al buffer (no
    independencia probabilistica exacta).

    Algoritmo por iteracion i
    -------------------------
        test_idx     = [i]
        buffered_idx = { j : dist(i, j) <= buffer_dist }   # incluye i
        train_idx    = todos los puntos fuera del buffer

    El buffer elimina los vecinos cercanos al punto de prueba del conjunto
    de entrenamiento, evitando que el modelo "aprenda" de observaciones
    espacialmente correlacionadas con el punto evaluado.

    Equivalente R (Box 1, LOO con buffer):
        for (i in 1:n) {
            test  <- i
            dists <- as.vector(dist.mat[i, ])
            train <- which(dists > scaled.buffer)
        }

    Implementacion eficiente con scipy.spatial.KDTree
    -------------------------------------------------
    - KDTree se construye UNA SOLA VEZ en el constructor: O(n log n)
    - query_ball_point() por punto: O(log n + k), k = vecinos en buffer
    - Complejidad total: O(n log n) en lugar de O(n^2) naive

    Parametros
    ----------
    coords      : np.ndarray, shape (n_samples, 2)
        Coordenadas (x, y) en metros (CRS proyectado, e.g. EPSG:32717).
    buffer_dist : float
        Radio de exclusion en metros. Por defecto debe ser el rango del
        semivariograma (distancia donde la autocorrelacion se anula).

    Nota de rendimiento
    -------------------
    Para n=5051 este metodo genera 5051 iteraciones. Con modelos rapidos
    (Ridge, LinearRegression) el ciclo completo tarda ~30 segundos.
    Para SANNWR/GNNWR se recomienda evaluar un subconjunto representativo.

    Ejemplo
    -------
    >>> cv = BufferedLOO(coords=coords, buffer_dist=5621.8)
    >>> for train_idx, test_idx in cv.split(X):
    ...     model.fit(X[train_idx], y[train_idx])
    """

    def __init__(
        self,
        coords: np.ndarray,
        buffer_dist: float,
    ) -> None:
        self.coords      = np.asarray(coords, dtype=float)
        self.buffer_dist = float(buffer_dist)
        self._n          = len(self.coords)
        # KDTree construido una sola vez: O(n log n)
        self._tree       = KDTree(self.coords)

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        return self._n

    def _iter_test_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[np.ndarray]:
        """
        Implementacion requerida por BaseCrossValidator.
        Para LOO, cada punto es su propio test set.
        El buffer de exclusion se aplica en split() sobreescrito.
        """
        for i in range(self._n):
            yield np.array([i])

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Genera (train_idx, test_idx) con buffer de exclusion espacial.

        Sobreescribe BaseCrossValidator.split() para aplicar el buffer:
        el conjunto de entrenamiento excluye todos los puntos dentro
        del radio buffer_dist del punto de prueba i.
        """
        all_idx = np.arange(self._n)

        for i in range(self._n):
            # Vecinos dentro del buffer (incluye el propio punto i)
            buffered = np.array(
                self._tree.query_ball_point(self.coords[i], r=self.buffer_dist),
                dtype=int,
            )
            mask_train        = np.ones(self._n, dtype=bool)
            mask_train[buffered] = False

            yield all_idx[mask_train], np.array([i])

    def __repr__(self) -> str:
        return (
            f"BufferedLOO(n={self._n}, "
            f"buffer_dist={self.buffer_dist:.1f} m "
            f"({self.buffer_dist / 1000:.2f} km))"
        )


# =============================================================================
# Funcion de evaluacion: ciclo CV con MAE, RMSE, MAPE
# =============================================================================

def run_spatial_cv(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
    coords: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Ciclo de evaluacion de validacion cruzada espacial.

    Compatible con las tres estrategias: RandomKFoldCV, SpatialBlockCV
    y BufferedLOO. Equivalente Python del bucle CV en Roberts et al.
    (2017), Box 1:

        for (fold in 1:K) {
            mod  <- fit(train_data)
            pred <- predict(mod, test_data)
            RMSE[fold] <- sqrt(mean((pred - test$y)^2))
        }

    Parametros
    ----------
    model   : objeto con .fit(X, y) y .predict(X)
              Compatible con sklearn. Para SANNWR/GNNWR pasar coords.
    X       : np.ndarray, shape (n, p) — covariables
    y       : np.ndarray, shape (n,)   — variable objetivo
    cv      : instancia de RandomKFoldCV, SpatialBlockCV o BufferedLOO
    coords  : np.ndarray, shape (n, 2), opcional
              Coordenadas en metros. Si se provee, el modelo se llama como
              model.fit(X, y, coords) y model.predict(X, coords)
              (interfaz extendida para modelos geograficos SANNWR/GNNWR).
    verbose : bool — imprimir progreso (default: True, se omite si n > 50)

    Retorna
    -------
    results : pd.DataFrame
        Columnas: fold, n_train, n_test, MAE, RMSE, MAPE
    summary : dict
        Promedios ponderados por n_test:
        MAE_mean, RMSE_mean, MAPE_mean, MAE_std, RMSE_std
    """
    records: list[dict] = []
    n_splits = cv.get_n_splits(X, y)
    show_progress = verbose and (n_splits <= 50)

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        if show_progress:
            print(
                f"    Fold {fold_id + 1}/{n_splits}  "
                f"train={len(train_idx):,}  test={len(test_idx):,}"
            )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if coords is not None:
            # Interfaz extendida para modelos geograficos (SANNWR/GNNWR)
            model.fit(X_train, y_train, coords[train_idx])
            y_pred = model.predict(X_test, coords[test_idx])
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        errors = y_test - y_pred
        mae    = float(np.mean(np.abs(errors)))
        rmse   = float(np.sqrt(np.mean(errors ** 2)))
        mask   = y_test != 0
        mape   = (
            float(np.mean(np.abs(errors[mask] / y_test[mask])) * 100)
            if mask.any()
            else float("nan")
        )

        records.append({
            "fold":    fold_id,
            "n_train": len(train_idx),
            "n_test":  len(test_idx),
            "MAE":     round(mae,  4),
            "RMSE":    round(rmse, 4),
            "MAPE":    round(mape, 4),
        })

    results = pd.DataFrame(records)
    w       = results["n_test"].values / results["n_test"].sum()

    summary = {
        "MAE_mean":  round(float(np.average(results["MAE"],  weights=w)), 4),
        "RMSE_mean": round(float(np.average(results["RMSE"], weights=w)), 4),
        "MAPE_mean": round(float(np.average(results["MAPE"], weights=w)), 4),
        "MAE_std":   round(float(results["MAE"].std()),  4),
        "RMSE_std":  round(float(results["RMSE"].std()), 4),
    }

    return results, summary
