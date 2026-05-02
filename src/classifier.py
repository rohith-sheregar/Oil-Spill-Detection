"""
Module 2 — Look-alike Rejection Classifier
A Random Forest classifier that distinguishes real oil spill patches
from natural look-alike phenomena (low wind zones, algal blooms, biogenic films)
using geometric and contextual features extracted from the segmentation mask.
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.features import PatchFeatures, features_to_array


class LookAlikeClassifier:
    """
    Random Forest binary classifier for oil spill look-alike rejection.

    Labels:
        1 = oil_spill  (positive class)
        0 = look_alike (negative class)
    """

    FEATURE_NAMES = [
        "area_pixels",
        "area_km2",
        "perimeter",
        "elongation",
        "aspect_ratio",
        "compactness",
        "solidity",
        "extent",
        "hu_moment_1",
        "hu_moment_2",
        "mean_intensity",
        "std_intensity",
        "is_night",
    ]

    CLASSES = ["look_alike", "oil_spill"]

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_trained: bool = False
        self.feature_names: list[str] = self.FEATURE_NAMES
        self.classes: list[str] = self.CLASSES

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the classifier on pre-extracted feature arrays.

        Args:
            X: Feature matrix of shape (N, 13).
            y: Binary labels — 1 = oil_spill, 0 = look_alike — shape (N,).

        Returns:
            Dict with keys: accuracy, report, confusion_matrix, feature_importances.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale: fit on train only, apply to both
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)

        self.clf.fit(X_train_sc, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.clf.predict(X_test_sc)
        accuracy = float(np.mean(y_pred == y_test))
        report   = classification_report(
            y_test, y_pred, target_names=self.classes, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)

        # Feature importances sorted descending
        importances = self.clf.feature_importances_
        fi_dict = dict(
            sorted(
                {name: float(imp) for name, imp in zip(self.feature_names, importances)}.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )

        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm,
            "feature_importances": fi_dict,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: list) -> list:
        """
        Classify a list of PatchFeatures objects.

        Args:
            features: List of PatchFeatures instances.

        Returns:
            List of dicts, each with keys: label (str), confidence (float), is_oil (bool).

        Raises:
            RuntimeError: If classifier has not been trained or loaded.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Classifier not trained. Call train() or load() first."
            )

        if not features:
            return []

        X = features_to_array(features)
        X_sc = self.scaler.transform(X)

        labels_idx    = self.clf.predict(X_sc)
        probabilities = self.clf.predict_proba(X_sc)

        # class_index for "oil_spill" — RandomForest orders by sorted unique labels
        # Our labels are 0 (look_alike) and 1 (oil_spill) → oil_spill proba is column 1
        oil_col = list(self.clf.classes_).index(1) if 1 in self.clf.classes_ else 1

        results = []
        for i, (label_idx, proba) in enumerate(zip(labels_idx, probabilities)):
            confidence = float(proba[oil_col])
            label_str  = "oil_spill" if int(label_idx) == 1 else "look_alike"
            results.append({
                "label":      label_str,
                "confidence": confidence,
                "is_oil":     label_str == "oil_spill",
            })
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialise the scaler + classifier to a single pickle file.

        Args:
            path: Destination file path (e.g. outputs/checkpoints/module2_classifier.pkl).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "scaler":       self.scaler,
            "clf":          self.clf,
            "is_trained":   self.is_trained,
            "feature_names": self.feature_names,
            "classes":      self.classes,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Classifier saved → {path}")

    def load(self, path: str) -> None:
        """
        Restore a previously saved classifier from disk.

        Args:
            path: Path to the pickle file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Classifier checkpoint not found at '{path}'. "
                "Run train_module2.py first to generate it."
            )
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.scaler        = payload["scaler"]
        self.clf           = payload["clf"]
        self.is_trained    = True
        self.feature_names = payload.get("feature_names", self.FEATURE_NAMES)
        self.classes       = payload.get("classes", self.CLASSES)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_feature_importance_report(self) -> str:
        """
        Return a ranked, formatted feature importance table.

        Returns:
            Multi-line string with feature importances ranked highest to lowest.
        """
        if not self.is_trained:
            return "Classifier not trained yet."

        importances = self.clf.feature_importances_
        ranked = sorted(
            zip(self.feature_names, importances),
            key=lambda kv: kv[1],
            reverse=True,
        )
        lines = ["Feature Importances (ranked):"]
        for rank, (name, importance) in enumerate(ranked, start=1):
            bar = "█" * int(importance * 40)
            lines.append(f"  {rank:2d}. {name:<20s} {importance:.4f} {bar}")
        return "\n".join(lines)
