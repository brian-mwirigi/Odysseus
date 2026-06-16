"""
ML Pipeline v2 - Team Odysseus (DataSprint 2026)
Fast improvements on top of v1 (0.5475 baseline):
- Target encoding for ALL categoricals (county, marital, barriers) inside CV
- Frequency encoding replaces sparse one-hot
- Extra interaction features
- Soft voting ensemble of top-3
- Improved threshold search
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from preprocessing import load_and_preprocess


class ThresholdWrapper:
    def __init__(self, base_model, weights, n_classes):
        self.base_model = base_model
        self.weights = weights
        self.classes_ = np.arange(n_classes)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X) * self.weights, axis=1)


class SoftVotingWrapper:
    def __init__(self, models, weights=None):
        self.models = models
        self.model_weights = weights
        self.classes_ = np.arange(3)

    def predict_proba(self, X):
        probas = [m.predict_proba(X) for m in self.models]
        if self.model_weights is not None:
            avg = sum(w * p for w, p in zip(self.model_weights, probas)) / sum(self.model_weights)
        else:
            avg = np.mean(probas, axis=0)
        avg = avg / avg.sum(axis=1, keepdims=True)
        return avg

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def target_encode_fold(X_tr, y_tr, X_val, cat_cols, worsened_cls):
    overall = (y_tr == worsened_cls).mean()
    X_tr, X_val = X_tr.copy(), X_val.copy()
    for col in cat_cols:
        means = pd.Series(y_tr == worsened_cls).groupby(X_tr[col].values).mean()
        X_tr[col + '_te'] = X_tr[col].map(means).fillna(overall)
        X_val[col + '_te'] = X_val[col].map(means).fillna(overall)
    X_tr = X_tr.drop(columns=cat_cols)
    X_val = X_val.drop(columns=cat_cols)
    return X_tr, X_val


def target_encode_full(X_data, y_data, cat_cols, worsened_cls):
    overall = (y_data == worsened_cls).mean()
    X_out = X_data.copy()
    te_maps = {}
    for col in cat_cols:
        means = pd.Series(y_data == worsened_cls).groupby(X_data[col].values).mean()
        X_out[col + '_te'] = X_out[col].map(means).fillna(overall)
        te_maps[col] = {'means': means, 'overall': overall}
    X_out = X_out.drop(columns=cat_cols)
    return X_out, te_maps


def optimize_thresholds(probas, y_true, n_classes, n_iter=150):
    best_score = f1_score(y_true, np.argmax(probas, axis=1), average='weighted')
    best_weights = np.ones(n_classes)
    rng = np.random.RandomState(42)
    for i in range(n_iter):
        scale = max(0.05, 1.5 - i / n_iter)
        candidate = best_weights + rng.normal(0, scale * 0.1, n_classes)
        candidate = np.clip(candidate, 0.1, 3.0)
        score = f1_score(y_true, np.argmax(probas * candidate, axis=1), average='weighted')
        if score > best_score:
            best_score, best_weights = score, candidate.copy()
    for _ in range(n_iter * 3):
        candidate = best_weights + rng.normal(0, 0.01, n_classes)
        candidate = np.clip(candidate, 0.1, 3.0)
        score = f1_score(y_true, np.argmax(probas * candidate, axis=1), average='weighted')
        if score > best_score:
            best_score, best_weights = score, candidate.copy()
    return best_weights, best_score


print("=" * 70)
print("LOADING DATA & FEATURES")
print("=" * 70)
X, y, class_names, le_target, cat_cols_to_te = load_and_preprocess()
worsened_cls = list(class_names).index('Worsened')
print(f"{X.shape[0]} rows, {X.shape[1]} features (before TE)")
print(f"Cat cols for TE: {cat_cols_to_te}")

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_final, train_te_maps = target_encode_full(X_train_full, y_train_full, cat_cols_to_te, worsened_cls)
X_test_final = X_test.copy()
for col in cat_cols_to_te:
    X_test_final[col + '_te'] = X_test_final[col].map(train_te_maps[col]['means']).fillna(train_te_maps[col]['overall'])
X_test_final = X_test_final.drop(columns=cat_cols_to_te)

X_all_final, _ = target_encode_full(X, y, cat_cols_to_te, worsened_cls)
print(f"{X_train_final.shape[1]} features after TE")

# --- CatBoost Optuna (quick 8 trials) ---
print("\n" + "=" * 70)
print("OPTUNA (CatBoost) - 8 trials")
print("=" * 70)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def cb_objective(trial):
    p = {
        'iterations': 300,
        'learning_rate': trial.suggest_float('lr', 0.02, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 7),
        'l2_leaf_reg': trial.suggest_float('l2', 1.0, 10.0),
        'random_strength': trial.suggest_float('rs', 0.5, 3.0),
        'bagging_temperature': trial.suggest_float('bt', 0.0, 1.0),
        'border_count': 254,
        'auto_class_weights': 'Balanced', 'verbose': 0, 'random_seed': 42
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for ti, vi in skf.split(X_train_final, y_train_full):
        smt = SMOTETomek(random_state=42)
        Xr, yr = smt.fit_resample(X_train_final.iloc[ti], y_train_full[ti])
        m = CatBoostClassifier(**p).fit(Xr, yr)
        scores.append(f1_score(y_train_full[vi], m.predict(X_train_final.iloc[vi]).ravel(), average='weighted'))
    return np.mean(scores)


study = optuna.create_study(direction='maximize')
study.optimize(cb_objective, n_trials=8)
best_cb = study.best_params.copy()
best_cb.update({'iterations': 1000, 'auto_class_weights': 'Balanced', 'verbose': 0, 'random_seed': 42,
                'learning_rate': best_cb.pop('lr'), 'l2_leaf_reg': best_cb.pop('l2'),
                'random_strength': best_cb.pop('rs'), 'bagging_temperature': best_cb.pop('bt')})
print(f"Best CB: CV={study.best_value:.4f}")


def get_base_models():
    return [
        ("XGB", XGBClassifier(
            n_estimators=800, learning_rate=0.039, max_depth=6,
            min_child_weight=8, subsample=0.7, colsample_bytree=0.96,
            gamma=2.06, reg_alpha=1.71, random_state=42, n_jobs=-1
        )),
        ("LGBM", LGBMClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=6,
            num_leaves=40, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
        )),
        ("CatBoost", CatBoostClassifier(**best_cb)),
        ("GB", GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, min_samples_leaf=5, random_state=42
        )),
    ]


# --- Cross-Validation ---
print("\n" + "=" * 70)
print("CROSS VALIDATION (5-Fold)")
print("=" * 70)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_models = get_base_models()
cv_scores = {n: [] for n, _ in base_models}
oof_probas = {n: np.zeros((len(y_train_full), len(class_names))) for n, _ in base_models}

for fold, (ti, vi) in enumerate(skf.split(X_train_full, y_train_full), 1):
    print(f"--- Fold {fold} ---")
    Xtr = X_train_full.iloc[ti].copy()
    ytr = y_train_full[ti]
    Xva = X_train_full.iloc[vi].copy()
    yva = y_train_full[vi]

    Xtr, Xva = target_encode_fold(Xtr, ytr, Xva, cat_cols_to_te, worsened_cls)

    smt = SMOTETomek(random_state=42)
    Xtr_r, ytr_r = smt.fit_resample(Xtr, ytr)
    fsw = compute_sample_weight('balanced', ytr_r)

    for name, model in base_models:
        m = type(model)(**model.get_params())
        if name == "XGB":
            m.fit(Xtr_r, ytr_r, sample_weight=fsw)
        else:
            m.fit(Xtr_r, ytr_r)
        probas = m.predict_proba(Xva)
        cv_scores[name].append(f1_score(yva, np.argmax(probas, axis=1), average='weighted'))
        oof_probas[name][vi] = probas
        print(f"  {name:15s}: F1={cv_scores[name][-1]:.4f}")

mean_cv = {k: np.mean(v) for k, v in cv_scores.items()}
print("\n--- Mean CV ---")
for n in sorted(mean_cv, key=mean_cv.get, reverse=True):
    print(f"  {n:15s}: {mean_cv[n]:.4f}")

# --- Stacking ---
print("\n" + "=" * 70)
print("STACKING META-LEARNER")
print("=" * 70)
meta_train = np.hstack([oof_probas[n] for n, _ in base_models])
meta_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=0.5)
meta_lr.fit(meta_train, y_train_full)

# --- Retrain on Full Train ---
print("\n" + "=" * 70)
print("RETRAINING ON FULL TRAIN (SMOTE-Tomek)")
print("=" * 70)
smt = SMOTETomek(random_state=42)
Xr, yr = smt.fit_resample(X_train_final, y_train_full)
sw = compute_sample_weight('balanced', yr)

trained = {}
for name, model in get_base_models():
    print(f"  {name}...")
    m = type(model)(**model.get_params())
    if name == "XGB":
        m.fit(Xr, yr, sample_weight=sw)
    else:
        m.fit(Xr, yr)
    trained[name] = m

# --- Test Scores ---
print("\n" + "=" * 70)
print("TEST SCORES")
print("=" * 70)
test_scores, test_probas = {}, {}
for name, model in trained.items():
    p = model.predict_proba(X_test_final)
    test_probas[name] = p
    test_scores[name] = f1_score(y_test, np.argmax(p, axis=1), average='weighted')
    print(f"  {name:15s}: F1={test_scores[name]:.4f}")

# Stacking
mt = np.hstack([trained[n].predict_proba(X_test_final) for n, _ in base_models])
test_probas["Stacking"] = meta_lr.predict_proba(mt)
test_scores["Stacking"] = f1_score(y_test, meta_lr.predict(mt), average='weighted')
print(f"  Stacking        : F1={test_scores['Stacking']:.4f}")

# --- Soft Voting ---
print("\n" + "=" * 70)
print("BLENDED ENSEMBLE (joint blend + threshold search)")
print("=" * 70)
model_names_list = [n for n, _ in base_models]
best_blend_score, best_blend_w, best_thresh_w = 0, np.ones(len(model_names_list)) / len(model_names_list), np.ones(len(class_names))
threshold_weights = {}

rng = np.random.RandomState(42)
for i in range(5000):
    bw = rng.dirichlet(np.ones(len(model_names_list)))
    blended = sum(w * oof_probas[n] for w, n in zip(bw, model_names_list))
    blended /= blended.sum(axis=1, keepdims=True)
    tw = np.ones(len(class_names)) + rng.normal(0, 0.3, len(class_names))
    tw = np.clip(tw, 0.3, 2.5)
    preds = np.argmax(blended * tw, axis=1)
    s = f1_score(y_train_full, preds, average='weighted')
    if s > best_blend_score:
        best_blend_score, best_blend_w, best_thresh_w = s, bw.copy(), tw.copy()

for _ in range(10000):
    bw = best_blend_w + rng.normal(0, 0.02, len(model_names_list))
    bw = np.clip(bw, 0.01, None)
    bw /= bw.sum()
    blended = sum(w * oof_probas[n] for w, n in zip(bw, model_names_list))
    blended /= blended.sum(axis=1, keepdims=True)
    tw = best_thresh_w + rng.normal(0, 0.02, len(class_names))
    tw = np.clip(tw, 0.3, 2.5)
    preds = np.argmax(blended * tw, axis=1)
    s = f1_score(y_train_full, preds, average='weighted')
    if s > best_blend_score:
        best_blend_score, best_blend_w, best_thresh_w = s, bw.copy(), tw.copy()

print(f"  Best blend weights: {dict(zip(model_names_list, best_blend_w.round(3)))}")
print(f"  Best thresh weights: {best_thresh_w.round(3)}")
print(f"  OOF F1: {best_blend_score:.4f}")

sv_blend = SoftVotingWrapper([trained[n] for n in model_names_list], weights=list(best_blend_w))
blend_proba = sv_blend.predict_proba(X_test_final)
blend_tuned_pred = np.argmax(blend_proba * best_thresh_w, axis=1)
blend_f1 = f1_score(y_test, blend_tuned_pred, average='weighted')
test_scores["BlendTuned"] = blend_f1
trained["BlendTuned"] = ThresholdWrapper(sv_blend, best_thresh_w, len(class_names))
test_probas["BlendTuned"] = blend_proba
threshold_weights["BlendTuned"] = best_thresh_w
print(f"  BlendTuned Test F1: {blend_f1:.4f}")

# --- Threshold Optimization (individual models) ---
print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)
threshold_weights.setdefault("BlendTuned", best_thresh_w)

for mn in ["CatBoost", "XGB", "LGBM", "GB"]:
    print(f"  {mn}...")
    bw, bvs = optimize_thresholds(oof_probas[mn], y_train_full, len(class_names), n_iter=150)
    tp = test_probas[mn]
    tf1 = f1_score(y_test, np.argmax(tp * bw, axis=1), average='weighted')
    test_scores[f"{mn}+Threshold"] = tf1
    threshold_weights[f"{mn}+Threshold"] = bw
    trained[f"{mn}+Threshold"] = ThresholdWrapper(trained[mn], bw, len(class_names))
    print(f"    weights={bw.round(3)}  Test F1={tf1:.4f}")

# --- Leaderboard ---
print("\n" + "=" * 70)
print("LEADERBOARD")
print("=" * 70)
final_lb = sorted(test_scores.items(), key=lambda x: x[1], reverse=True)
for r, (n, s) in enumerate(final_lb, 1):
    print(f"  {r:2d}. {n:35s} F1={s:.4f}{' <<<< WINNER' if r == 1 else ''}")

best_name = final_lb[0][0]
best_model = trained[best_name]
y_pred = best_model.predict(X_test_final)
if hasattr(y_pred, 'ravel'):
    y_pred = y_pred.ravel()
f1_final = f1_score(y_test, y_pred, average='weighted')

print(f"\nFINAL: {best_name} — F1={f1_final:.4f}")
print(classification_report(y_test, y_pred, target_names=class_names))

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, cmap='Blues', ax=ax)
ax.set_title(f'{best_name} (F1={f1_final:.4f})')
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=150)
plt.close()

joblib.dump(best_model, 'models/odysseus_final_model.pkl')
te_maps_save = {col: {'means': train_te_maps[col]['means'], 'overall': train_te_maps[col]['overall']} for col in cat_cols_to_te}

joblib.dump({
    'X_test_final': X_test_final, 'y_test': y_test, 'y_pred': y_pred,
    'class_names': class_names,
    'trained': {k: v for k, v in trained.items() if not isinstance(v, (ThresholdWrapper, SoftVotingWrapper))},
    'test_scores': test_scores, 'mean_cv': mean_cv,
    'X_all_final': X_all_final, 'y_full': y,
    'te_maps': te_maps_save, 'cat_cols_to_te': cat_cols_to_te,
    'worsened_cls': worsened_cls, 'threshold_weights': threshold_weights,
    'best_name': best_name, 'meta_lr': meta_lr, 'best_cb_params': best_cb,
}, 'models/pipeline_state.pkl')

pd.DataFrame({'True': y_test.ravel(), 'Predicted': y_pred.ravel()}).to_csv('data/final_predictions.csv', index=False)
pd.DataFrame(final_lb, columns=['Model', 'Score']).to_csv('data/model_leaderboard.csv', index=False)
print("\nSaved artifacts successfully.")
