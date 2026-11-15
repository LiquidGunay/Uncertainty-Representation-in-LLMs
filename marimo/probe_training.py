import marimo


app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import joblib
    from pathlib import Path
    from typing import Tuple
    from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.calibration import CalibratedClassifierCV
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        import torch
        import torch.nn as nn
        from skorch import NeuralNetBinaryClassifier
        HAVE_SKORCH = True
        TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        HAVE_SKORCH = False
        TORCH_DEVICE = "cpu"
        torch = None
        nn = None
        NeuralNetBinaryClassifier = None

    try:
        from thundersvm import SVC as ThunderSVC
        HAVE_TSVC = True
    except Exception:
        ThunderSVC = None
        HAVE_TSVC = False

    sns.set_theme(style="whitegrid")

    return (
        mo,
        np,
        pd,
        pl,
        joblib,
        Path,
        StratifiedKFold,
        train_test_split,
        GridSearchCV,
        StandardScaler,
        PCA,
        Pipeline,
        LogisticRegression,
        accuracy_score,
        average_precision_score,
        classification_report,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
        CalibratedClassifierCV,
        plt,
        HAVE_SKORCH,
        TORCH_DEVICE,
        torch,
        nn,
        NeuralNetBinaryClassifier,
        ThunderSVC,
        HAVE_TSVC,
    )


@app.cell
def _(mo, Path):
    data_dir = Path(".").resolve()
    x_path_input = mo.ui.text(
        value=str(data_dir / "X_4096.npy"),
        label="Hidden state features (.npy)",
        full_width=True,
    )
    y_path_input = mo.ui.text(
        value=str(data_dir / "y_labels.npy"),
        label="Labels (.npy)",
        full_width=True,
    )
    uids_path_input = mo.ui.text(
        value=str(data_dir / "uids.npy"),
        label="UIDs (.npy, optional)",
        full_width=True,
    )

    mo.vstack(
        [
            mo.md("### Probe dataset"),
            x_path_input,
            y_path_input,
            uids_path_input,
        ]
    )

    return x_path_input, y_path_input, uids_path_input


@app.cell
def _(mo, np, Path, pl, x_path_input, y_path_input, uids_path_input):
    def load_array(path_text: str) -> Tuple[np.ndarray, Path]:
        path = Path(path_text).expanduser().resolve()
        mo.stop(not path.exists(), mo.md(f"**Missing file:** `{path}`"))
        array = np.load(path)
        return array, path

    X, X_path = load_array(x_path_input.value)
    y, y_path = load_array(y_path_input.value)

    uids_path = Path(uids_path_input.value).expanduser().resolve()
    uids = None
    if uids_path.exists():
        uids = np.load(uids_path, allow_pickle=True)

    summary = pl.DataFrame(
        {
            "stat": ["samples", "features", "positives", "positivity_rate"],
            "value": [
                int(X.shape[0]),
                int(X.shape[1]),
                int(y.sum()),
                float(np.mean(y)),
            ],
        }
    )
    mo.hstack(
        [
            mo.md(
                f"Loaded **{X_path.name}** and **{y_path.name}** from `{X_path.parent}`"
            ),
            mo.ui.dataframe(summary),
        ]
    )

    return X, y, uids, X_path, y_path, uids_path


@app.cell
def _(mo, HAVE_SKORCH, HAVE_TSVC):
    class_weight_options = [("Balanced", "balanced"), ("Uniform", "none")]
    model_options = [("Logistic Regression (scikit-learn)", "logreg_sklearn")]
    if HAVE_SKORCH:
        model_options.append(("Logistic Regression (skorch/PyTorch)", "logreg_skorch"))
    if HAVE_TSVC:
        model_options.append(("Linear SVM (ThunderSVM)", "svm_linear"))

    model_selector = mo.ui.dropdown(
        options=model_options, value=model_options[0][1], label="Model type"
    )
    class_weight_selector = mo.ui.dropdown(
        options=class_weight_options, value="balanced", label="Class weighting"
    )
    test_size_slider = mo.ui.slider(
        start=0.1, stop=0.4, value=0.2, step=0.05, label="Test split size"
    )
    n_splits_slider = mo.ui.slider(
        start=3, stop=8, value=5, step=1, label="StratifiedKFold splits"
    )
    random_state_input = mo.ui.number(value=42, label="Random seed")
    calibrate_checkbox = mo.ui.checkbox(
        label="Calibrate probabilities (SVM only)", value=True
    )
    train_button = mo.ui.run_button(label="Train probe")

    mo.vstack(
        [
            mo.md("### Training controls"),
            mo.hstack([model_selector, class_weight_selector]),
            mo.hstack([test_size_slider, n_splits_slider]),
            mo.hstack([random_state_input, calibrate_checkbox]),
            train_button,
        ]
    )

    return (
        model_selector,
        class_weight_selector,
        test_size_slider,
        n_splits_slider,
        random_state_input,
        calibrate_checkbox,
        train_button,
    )


@app.cell
def _(
    NeuralNetBinaryClassifier,
    Pipeline,
    StandardScaler,
    StratifiedKFold,
    ThunderSVC,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    joblib,
    LogisticRegression,
    mo,
    nn,
    np,
    roc_auc_score,
    torch,
    train_test_split,
    GridSearchCV,
    CalibratedClassifierCV,
    HAVE_SKORCH,
    TORCH_DEVICE,
):
    def _build_skorch_pipeline(class_weight: str, random_state: int):
        if not HAVE_SKORCH:
            raise RuntimeError("skorch is unavailable in this environment.")

        pos_weight_val = 1.0
        if class_weight == "balanced":
            pos_weight_val = None  # computed during fit

        class TorchLogisticRegression(nn.Module):
            def __init__(self, input_dim: int = 4096):
                super().__init__()
                self.linear = nn.Linear(input_dim, 1)

            def forward(self, X):
                return self.linear(X)

        def _make_net(input_dim: int, pos_weight):
            net = NeuralNetBinaryClassifier(
                module=TorchLogisticRegression,
                module__input_dim=input_dim,
                criterion=nn.BCEWithLogitsLoss,
                criterion__pos_weight=pos_weight,
                optimizer=torch.optim.Adam,
                lr=1e-3,
                max_epochs=40,
                batch_size=256,
                train_split=None,
                device=TORCH_DEVICE,
                verbose=0,
            )
            return net

        def make_pipeline(X_train: np.ndarray, y_train: np.ndarray):
            pos_weight_tensor = None
            if class_weight == "balanced":
                num_pos = max(int((y_train == 1).sum()), 1)
                num_neg = max(int((y_train == 0).sum()), 1)
                pos_weight_tensor = torch.tensor(
                    num_neg / num_pos, device=TORCH_DEVICE, dtype=torch.float32
                )

            net = _make_net(X_train.shape[1], pos_weight_tensor)
            pipe = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("clf", net),
                ]
            )
            param_grid = {
                "clf__lr": [1e-3, 3e-3, 1e-2],
                "clf__max_epochs": [30, 60],
                "clf__optimizer__weight_decay": [0.0, 1e-4, 1e-3],
            }
            return pipe, param_grid, {}

        return make_pipeline

    def train_linear_probe(
        X: np.ndarray,
        y: np.ndarray,
        *,
        model_choice: str = "logreg_sklearn",
        test_size: float = 0.2,
        random_state: int = 42,
        class_weight: str = "balanced",
        n_splits: int = 5,
        do_calibrate: bool = True,
    ):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("y must be binary labels {0,1}.")

        train_idx, test_idx = train_test_split(
            np.arange(X.shape[0]),
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if class_weight == "none":
            class_weight_arg = None
        elif class_weight == "balanced":
            class_weight_arg = "balanced"
        else:
            raise ValueError(f"Unsupported class_weight: {class_weight}")

        if model_choice == "logreg_sklearn":
            pipe = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "clf",
                        LogisticRegression(
                            solver="saga",
                            penalty="l2",
                            max_iter=3000,
                            class_weight=class_weight_arg,
                            n_jobs=None,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            param_grid = {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__penalty": ["l1", "l2"],
            }
            needs_calibration = False
            fit_kwargs = {}

        elif model_choice == "logreg_skorch":
            make_pipeline = _build_skorch_pipeline(class_weight, random_state)
            pipe, param_grid, fit_kwargs = make_pipeline(X_train, y_train)
            needs_calibration = False

        elif model_choice == "svm_linear":
            if ThunderSVC is None:
                raise RuntimeError("ThunderSVM is unavailable in this environment.")
            pipe = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "clf",
                        ThunderSVC(
                            kernel="linear",
                            C=1.0,
                            probability=False,
                        ),
                    ),
                ]
            )
            param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
            needs_calibration = do_calibrate
            if class_weight_arg == "balanced":
                n1 = max(int((y_train == 1).sum()), 1)
                n0 = max(int((y_train == 0).sum()), 1)
                w1 = 0.5 / n1
                w0 = 0.5 / n0
                sample_weight = np.where(y_train == 1, w1, w0).astype(np.float32)
                fit_kwargs = {"clf__sample_weight": sample_weight}
            else:
                fit_kwargs = {}

        else:
            raise ValueError(f"Unknown model choice: {model_choice}")

        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",
            refit=True,
            n_jobs=None,
        )
        grid.fit(X_train, y_train, **fit_kwargs)

        best_model = grid.best_estimator_
        calibrated_model = None
        if needs_calibration:
            calibrated_model = CalibratedClassifierCV(
                estimator=best_model,
                cv=5,
                method="sigmoid",
            )
            calibrated_model.fit(X_train, y_train)
            model_for_eval = calibrated_model
        else:
            model_for_eval = best_model

        preds_train = model_for_eval.predict(X_train)
        preds_test = model_for_eval.predict(X_test)

        if hasattr(model_for_eval, "predict_proba"):
            probs_test = model_for_eval.predict_proba(X_test)[:, 1]
        elif hasattr(model_for_eval, "decision_function"):
            scores = model_for_eval.decision_function(X_test)
            probs_test = 1.0 / (1.0 + np.exp(-scores))
        else:
            probs_test = preds_test.astype(float)

        if hasattr(model_for_eval, "decision_function"):
            margins = model_for_eval.decision_function(X_test)
        else:
            eps = 1e-6
            margins = np.log(
                np.clip(probs_test, eps, 1 - eps)
                / np.clip(1 - probs_test, eps, 1 - eps)
            )

        metrics = {
            "train_accuracy": float(accuracy_score(y_train, preds_train)),
            "test_accuracy": float(accuracy_score(y_test, preds_test)),
            "train_f1": float(f1_score(y_train, preds_train)),
            "test_f1": float(f1_score(y_test, preds_test)),
            "roc_auc": float(roc_auc_score(y_test, probs_test)),
            "average_precision": float(average_precision_score(y_test, probs_test)),
            "best_params": grid.best_params_,
            "best_cv_score": float(grid.best_score_),
            "calibrated": bool(calibrated_model is not None),
            "model_choice": model_choice,
            "scoring": "f1",
        }

        report = classification_report(y_test, preds_test, digits=4)

        cv_results = grid.cv_results_

        return {
            "model": model_for_eval,
            "best_pipeline": best_model,
            "metrics": metrics,
            "classification_report": report,
            "eval_artifacts": {
                "X_test": X_test,
                "y_test": y_test,
                "probs_test": probs_test,
                "preds_test": preds_test,
                "margins": margins,
            },
            "cv_results": cv_results,
            "train_indices": train_idx,
            "test_indices": test_idx,
        }

    return train_linear_probe


@app.cell
def _(np, pd, PCA, plt, sns):
    def recover_linear_plane(pipeline):
        scaler = pipeline.named_steps.get("scaler")
        clf = pipeline.named_steps.get("clf")
        if scaler is None or clf is None:
            return None

        if hasattr(clf, "module_"):
            w_tensor = clf.module_.linear.weight.detach().cpu().numpy().reshape(-1)
            b_tensor = (
                clf.module_.linear.bias.detach().cpu().numpy().reshape(-1)[0]
            )
        elif hasattr(clf, "coef_"):
            w_tensor = np.array(clf.coef_).reshape(-1)
            intercept = getattr(clf, "intercept_", np.array([0.0]))
            b_tensor = float(np.array(intercept).reshape(-1)[0])
        else:
            return None

        std = getattr(scaler, "scale_", None)
        mean = getattr(scaler, "mean_", None)
        if std is None or mean is None:
            return None

        w_orig = w_tensor / (std + 1e-12)
        b_orig = b_tensor - float(np.dot(w_tensor, mean / (std + 1e-12)))
        return {"w": w_orig, "b": b_orig}

    def _project_to_w_perp(X, w, *, random_state: int = 0):
        w = np.asarray(w, dtype=float)
        w_norm = w / (np.linalg.norm(w) + 1e-12)
        X_center = X - X.mean(axis=0, keepdims=True)
        projection = X_center - (X_center @ w_norm[:, None]) * w_norm[None, :]
        pca = PCA(n_components=2, random_state=random_state)
        Z = pca.fit_transform(projection)
        return Z, pca

    def plot_margin_vs_pc1_perp(
        X,
        y,
        w,
        b,
        *,
        hue=None,
        title: str = "Margin vs. PC1 in w⊥",
        random_state: int = 0,
    ):
        X = np.asarray(X)
        y = np.asarray(y)
        w = np.asarray(w, dtype=float)
        margin = X @ w + float(b)
        Z, _ = _project_to_w_perp(X, w, random_state=random_state)
        pc1 = Z[:, 0]
        frame_dict = {"pc1_perp": pc1, "margin": margin, "label": y}
        if hue is not None:
            frame_dict["hue"] = hue
        data = pd.DataFrame(frame_dict)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=data,
            x="pc1_perp",
            y="margin",
            hue="hue" if hue is not None else "label",
            palette="viridis",
            s=20,
            alpha=0.7,
            ax=ax,
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("PC1 in w⊥")
        ax.set_ylabel("Margin w·x + b")
        ax.legend()
        return {"pc1": pc1, "margin": margin, "figure": fig, "ax": ax}

    def plot_margin_vs_pc2_perp(
        X,
        y,
        w,
        b,
        *,
        hue=None,
        title: str = "Margin vs. PC2 in w⊥",
        random_state: int = 0,
    ):
        X = np.asarray(X)
        y = np.asarray(y)
        w = np.asarray(w, dtype=float)
        margin = X @ w + float(b)
        Z, _ = _project_to_w_perp(X, w, random_state=random_state)
        if Z.shape[1] < 2:
            return None
        pc2 = Z[:, 1]
        frame_dict = {"pc2_perp": pc2, "margin": margin, "label": y}
        if hue is not None:
            frame_dict["hue"] = hue
        data = pd.DataFrame(frame_dict)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=data,
            x="pc2_perp",
            y="margin",
            hue="hue" if hue is not None else "label",
            palette="viridis",
            s=20,
            alpha=0.7,
            ax=ax,
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("PC2 in w⊥")
        ax.set_ylabel("Margin w·x + b")
        ax.legend()
        return {"pc2": pc2, "margin": margin, "figure": fig, "ax": ax}

    def get_misclassified_examples(
        model,
        X,
        y,
        *,
        uids=None,
        top_k: int = 100,
        base_indices=None,
    ):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        if hasattr(model, "predict_proba"):
            prob_pos = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            prob_pos = 1.0 / (1.0 + np.exp(-scores))
        else:
            prob_pos = model.predict(X).astype(float)

        if hasattr(model, "decision_function"):
            margin_all = model.decision_function(X)
        else:
            eps = 1e-6
            margin_all = np.log(
                np.clip(prob_pos, eps, 1 - eps)
                / np.clip(1 - prob_pos, eps, 1 - eps)
            )

        preds = (prob_pos >= 0.5).astype(int)
        missed = np.where(preds != y)[0]
        if base_indices is None:
            dataset_indices = missed
        else:
            dataset_indices = np.asarray(base_indices)[missed]

        data = {
            "dataset_index": dataset_indices,
            "y_true": y[missed],
            "y_pred": preds[missed],
            "prob_pos": prob_pos[missed],
            "margin": margin_all[missed],
        }
        if uids is not None:
            data["uid"] = np.asarray(uids)[missed]

        df = pd.DataFrame(data).sort_values("prob_pos", ascending=False)
        if top_k is not None and len(df) > top_k:
            df = df.head(top_k)
        return df.reset_index(drop=True)

    return recover_linear_plane, plot_margin_vs_pc1_perp, plot_margin_vs_pc2_perp, get_misclassified_examples


@app.cell
def _(
    calibrate_checkbox,
    class_weight_selector,
    mo,
    model_selector,
    n_splits_slider,
    random_state_input,
    test_size_slider,
    train_button,
    train_linear_probe,
    X,
    TORCH_DEVICE,
    y,
):
    mo.stop(
        not train_button.value,
        mo.md("Configure parameters above, then click **Train probe**."),
    )

    result = train_linear_probe(
        X=X,
        y=y,
        model_choice=model_selector.value,
        test_size=test_size_slider.value,
        random_state=int(random_state_input.value),
        class_weight=class_weight_selector.value,
        n_splits=int(n_splits_slider.value),
        do_calibrate=calibrate_checkbox.value,
    )

    mo.md("### Training complete")
    result["metrics"]

    return result


@app.cell
def _(mo, pl, result):
    metrics = result["metrics"]
    metrics_rows = []
    for key, value in metrics.items():
        if key == "best_params":
            continue
        metrics_rows.append({"metric": key, "value": value})
    metrics_df = pl.DataFrame(metrics_rows)
    best_params_md = mo.md(
        "**Best parameters:** "
        + ", ".join(f"`{k}={v}`" for k, v in metrics["best_params"].items())
    )
    mo.vstack([metrics_df, best_params_md])
    return metrics_df, best_params_md


@app.cell
def _(mo, result):
    report = result["classification_report"]
    mo.md("### Classification report")
    mo.md(f"```\n{report}\n```")
    return report


@app.cell
def _(plt, result, roc_curve):
    artifacts = result["eval_artifacts"]
    fpr, tpr, _ = roc_curve(artifacts["y_test"], artifacts["probs_test"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {result['metrics']['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax
    return fpr, tpr


@app.cell
def _(plt, precision_recall_curve, result):
    artifacts = result["eval_artifacts"]
    precision, recall, _ = precision_recall_curve(
        artifacts["y_test"], artifacts["probs_test"]
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, label=f"AP = {result['metrics']['average_precision']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    ax
    return precision, recall


@app.cell
def _(mo, pd, pl, result):
    cv_df = pd.DataFrame(result["cv_results"])
    keep_cols = [
        c
        for c in cv_df.columns
        if c.startswith("param_")
        or c in {"mean_train_score", "std_train_score", "mean_test_score", "std_test_score", "rank_test_score"}
    ]
    table = (
        cv_df[keep_cols]
        .sort_values("rank_test_score", ascending=True)
        .head(10)
    )
    mo.md("### Cross-validation summary (top 10)")
    mo.ui.dataframe(pl.DataFrame(table))
    return table


@app.cell
def _(mo, plot_margin_vs_pc1_perp, recover_linear_plane, result, X, y):
    plane = recover_linear_plane(result["best_pipeline"])
    mo.stop(
        plane is None,
        mo.md("This visualization is available for linear models with accessible weights."),
    )
    viz = plot_margin_vs_pc1_perp(X, y, plane["w"], plane["b"])
    viz["ax"]
    return plane, viz


@app.cell
def _(mo, plot_margin_vs_pc2_perp, plane, X, y):
    mo.stop(
        plane is None,
        mo.md("PC2 visualization is only available when a linear plane was recovered."),
    )
    viz_pc2 = plot_margin_vs_pc2_perp(X, y, plane["w"], plane["b"])
    mo.stop(viz_pc2 is None, mo.md("PCA(2) could not be computed for this dataset."))
    viz_pc2["ax"]
    return viz_pc2


@app.cell
def _(Path, X_path, mo):
    default_dir = X_path.parent if X_path is not None else Path(".")
    model_output = mo.ui.text(
        value=str(default_dir / "linear_probe.joblib"),
        label="Model artifact path",
        full_width=True,
    )
    plane_output = mo.ui.text(
        value=str(default_dir / "classification_plane.npz"),
        label="Plane artifact path",
        full_width=True,
    )
    cv_output = mo.ui.text(
        value=str(default_dir / "cv_f1_table.csv"),
        label="Cross-validation table path",
        full_width=True,
    )
    missed_output = mo.ui.text(
        value=str(default_dir / "misclassified_examples.csv"),
        label="Misclassified CSV path",
        full_width=True,
    )
    save_button = mo.ui.run_button(label="Save artifacts")
    mo.vstack(
        [
            mo.md("### Persist artifacts"),
            model_output,
            plane_output,
            cv_output,
            missed_output,
            save_button,
        ]
    )
    return model_output, plane_output, cv_output, missed_output, save_button


@app.cell
def _(get_misclassified_examples, mo, np, pl, result, uids):
    artifacts = result["eval_artifacts"]
    test_indices = result["test_indices"]
    uids_subset = None
    if uids is not None:
        uids_subset = np.asarray(uids)[test_indices]
    missed_df = get_misclassified_examples(
        result["model"],
        artifacts["X_test"],
        artifacts["y_test"],
        uids=uids_subset,
        base_indices=test_indices,
        top_k=200,
    )
    mo.md("### Misclassified test examples (top 200)")
    mo.ui.dataframe(pl.DataFrame(missed_df))
    return missed_df
