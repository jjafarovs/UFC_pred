"""Phase 3: model training + probability calibration.

Baseline logistic regression and a gradient boosting alternative
(HistGradientBoostingClassifier), both wrapped in isotonic/Platt (sigmoid)
calibration, evaluated with a reliability diagram (calibration_table /
plot_calibration) alongside accuracy/log loss/Brier score/AUC.

Walk-forward discipline applies here too, not just in backtest.py: the
train/calibration/test split is chronological (chronological_split), never a
random shuffle. Per-fight features are already as-of-date safe (features.py),
but evaluating a model on a random mix of old and new fights would still
overstate how it'd perform in practice -- a realistic estimate requires
training on the past and evaluating on the (chronologically) future, same as
the real backtest in Phase 5.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_MATRIX_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "feature_matrix.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURE_COLUMNS = [
    "diff_win_pct",
    "diff_current_streak",
    "diff_sig_str_acc",
    "diff_sig_str_def",
    "diff_td_acc",
    "diff_td_def",
    "diff_days_since_last_fight",
    "diff_height_in",
    "diff_reach_in",
    "diff_age_years",
    "same_stance",
    # Rolling-window grappling-control-time share (features.fighter_rolling_features'
    # control_time_pct, 97.9% coverage from fight_stats.control_time_sec) --
    # the one Tier 1 candidate (of control time / split-decision rate /
    # title-fight / book-divergence) that actually validated: improved ROI
    # and tightened the bootstrap CI for BOTH logistic and GBM without
    # regressing accuracy/brier/calibration. The other three either helped
    # one model while hurting the other, or (scheduled_rounds) flipped
    # logistic's backtest into a proven significant loss -- see README's
    # Walk-forward backtest section.
    "diff_control_time_pct",
    # NOTE: diff_total_prior_fights/diff_finish_rate/diff_times_finished_rate
    # (features.fighter_career_features) were tried here and REMOVED --
    # no validated backtest improvement, GBM specifically got worse (full
    # drawdown). Kept computed in features.py for the dashboard's stat
    # display, deliberately excluded from the trained feature set. See
    # README's Walk-forward backtest section. Do not re-add without a new
    # walk-forward + bootstrap CI result that actually justifies it.
    # De-vigged closing market probability (see features.market_prob_feature) --
    # NaN for the ~91% of historical fights with no matched odds.
    # HistGradientBoostingClassifier handles this natively; the logistic
    # pipeline's median-imputer treats missing as "no information" via the
    # column median, which is a reasonable neutral fallback for a probability.
    "market_prob_fighter_1",
]


def load_feature_matrix(path: Path = FEATURE_MATRIX_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["same_stance"] = df["same_stance"].astype("float")  # True/False/None -> 1.0/0.0/NaN
    return df


def chronological_split(df: pd.DataFrame, calib_frac: float = 0.2, test_frac: float = 0.2):
    """Sorts by event_date and splits train/calibration/test in time order --
    never randomly. A random split would place an earlier fight in the test
    set and a later one in train; that's a leak in spirit even though each
    row's own features are already as-of-date safe, because it lets the model
    "see" market/meta conditions from later in time during training.
    """
    df = df.sort_values("event_date").reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_calib = int(n * calib_frac)
    n_train = n - n_test - n_calib
    if n_train <= 0 or n_calib <= 0 or n_test <= 0:
        raise ValueError(
            f"Not enough rows ({n}) to split into non-empty train/calibration/test sets "
            f"with calib_frac={calib_frac}, test_frac={test_frac}"
        )
    return df.iloc[:n_train], df.iloc[n_train : n_train + n_calib], df.iloc[n_train + n_calib :]


def make_xy(df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS):
    X = df[feature_columns].astype(float)
    y = df["label_fighter_1_win"].astype(int)
    return X, y


def build_logistic_pipeline(C: float = 0.01) -> Pipeline:
    """C is LogisticRegression's inverse regularization strength (smaller =
    stronger L2 penalty). C=0.01 is a validated default, not sklearn's raw
    default of 1.0: a walk-forward-safe TimeSeriesSplit search (scored on log
    loss, never touching the actual backtest window) found C<=0.03 as the
    shallow optimum, and re-running the real backtest confirmed it -- the
    proven-significant ROI loss at C=1.0 (CI=[-0.192,-0.003]) became
    not-significant at C=0.01 (CI=[-0.181,+0.003]) with no other metric
    regressing. See README's Walk-forward backtest section.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=1000)),
        ]
    )


def build_gradient_boosting_model(
    max_depth: int | None = 3,
    learning_rate: float = 0.03,
    max_iter: int = 100,
    l2_regularization: float = 1.0,
    min_samples_leaf: int = 50,
) -> HistGradientBoostingClassifier:
    """Defaults are a validated configuration, not sklearn's raw un-tuned
    defaults (max_depth=None, learning_rate=0.1, l2_regularization=0.0,
    min_samples_leaf=20) -- those let GBM fit noise on this modest tabular
    dataset (see README: the one time a feature addition made GBM's backtest
    strictly worse, ending in full drawdown, is consistent with this).

    A walk-forward-safe TimeSeriesSplit log-loss search over depth/learning
    rate/l2/min_samples_leaf pointed at stronger regularization, but its
    top pick by log loss alone (l2=0.0, leaf=20) actually made the real
    backtest's ROI CI flip from not-significant to a proven loss. Checking
    several more-regularized neighbors directly against the real backtest
    found this config (l2=1.0, leaf=50) improved every axis over sklearn's
    defaults: ROI -0.073->-0.046, CI [-0.171,+0.025]->[-0.143,+0.053]
    (tighter and higher), calibration gap 0.0839->0.0786. Lesson: log loss
    on a held-out slice is a decent starting point but not a substitute for
    checking the actual backtest -- see README's Walk-forward backtest
    section.

    Natively supports NaN features -- no imputer needed. That matters here
    because missingness is itself informative (e.g. "no tracked prior
    fights" likely means a promotional debut), and median-imputing it away
    would erase that signal rather than just filling a gap.
    """
    return HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_regularization=l2_regularization,
        min_samples_leaf=min_samples_leaf,
        random_state=0,
    )


def calibrate_model(fitted_model, X_calib, y_calib, method: str = "sigmoid"):
    """Wraps an already-fitted model with post-hoc calibration on a held-out
    calibration slice. FrozenEstimator replaces the older `cv='prefit'` API,
    which scikit-learn removed in 1.6 -- it tells CalibratedClassifierCV to
    treat `fitted_model` as already trained and only fit the calibrator.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    calibrated.fit(X_calib, y_calib)
    return calibrated


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    y_arr = np.asarray(y)
    metrics = {
        "n": len(y_arr),
        "accuracy": float(((proba >= 0.5).astype(int) == y_arr).mean()),
        "brier_score": float(brier_score_loss(y_arr, proba)),
    }
    if len(set(y_arr)) > 1:
        metrics["log_loss"] = float(log_loss(y_arr, proba, labels=[0, 1]))
        metrics["roc_auc"] = float(roc_auc_score(y_arr, proba))
    else:
        metrics["log_loss"] = None
        metrics["roc_auc"] = None
    return metrics


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Buckets predictions into `n_bins` equal-width bins and reports the
    empirical win rate per bin -- the numbers behind a reliability diagram.
    A well-calibrated model has mean_predicted ~= empirical_win_rate in every
    bin with enough rows to be meaningful.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_low": bins[b],
                "bin_high": bins[b + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(y_prob[mask].mean()),
                "empirical_win_rate": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_calibration(table: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    if not table.empty:
        ax.plot(table["mean_predicted"], table["empirical_win_rate"], marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_model_artifact(model, feature_columns: list[str], metrics: dict, name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Every artifact is timestamp-versioned so a backtest run can always be
    traced back to the exact model (and its metrics/feature list) that
    produced it -- never overwritten in place.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}_{ts}.joblib"
    joblib.dump(model, model_path)
    meta_path = models_dir / f"{name}_{ts}.json"
    meta_path.write_text(
        json.dumps({"feature_columns": feature_columns, "metrics": metrics, "trained_at": ts}, indent=2)
    )
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + calibrate a baseline win-probability model")
    parser.add_argument("--features", type=Path, default=FEATURE_MATRIX_PATH)
    parser.add_argument("--model", choices=["logistic", "gbm"], default="logistic")
    parser.add_argument("--calibration", choices=["isotonic", "sigmoid"], default="sigmoid")
    parser.add_argument("--calib-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    df = load_feature_matrix(args.features)
    train_df, calib_df, test_df = chronological_split(df, args.calib_frac, args.test_frac)
    print(f"train/calibration/test sizes: {len(train_df)}/{len(calib_df)}/{len(test_df)}")

    X_train, y_train = make_xy(train_df)
    X_calib, y_calib = make_xy(calib_df)
    X_test, y_test = make_xy(test_df)

    base_model = build_logistic_pipeline() if args.model == "logistic" else build_gradient_boosting_model()
    base_model.fit(X_train, y_train)

    calibrated_model = calibrate_model(base_model, X_calib, y_calib, method=args.calibration)

    raw_metrics = evaluate(base_model, X_test, y_test)
    calibrated_metrics = evaluate(calibrated_model, X_test, y_test)
    print("raw model test metrics:       ", raw_metrics)
    print("calibrated model test metrics:", calibrated_metrics)

    table = calibration_table(y_test.values, calibrated_model.predict_proba(X_test)[:, 1])
    print(table.to_string(index=False))

    plot_path = MODELS_DIR / f"calibration_plot_{args.model}.png"
    plot_calibration(table, plot_path, title=f"{args.model} + {args.calibration} calibration")
    print(f"model.py: wrote calibration plot -> {plot_path}")

    model_path = save_model_artifact(
        calibrated_model,
        FEATURE_COLUMNS,
        {
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
            # Persisted so report.py can ground its confidence label in how many
            # held-out test examples actually validated this probability range,
            # rather than guessing from feature completeness alone.
            "calibration_table": table.to_dict(orient="records"),
        },
        name=args.model,
    )
    print(f"model.py: saved model artifact -> {model_path}")


if __name__ == "__main__":
    main()
