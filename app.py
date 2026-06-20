from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─── Detectors ────────────────────────────────────────────────────────────────

def robust_zscore(series):
    med = np.median(series)
    mad = np.median(np.abs(series - med))
    if mad == 0:
        mad = 1e-9
    scores = np.abs(series - med) / (1.4826 * mad)
    return scores / (scores.max() + 1e-9)

def iqr_fence(series):
    q1, q3 = np.percentile(series, 25), np.percentile(series, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    scores = np.zeros(len(series))
    for i, v in enumerate(series):
        if v < lower:
            scores[i] = (lower - v) / (iqr + 1e-9)
        elif v > upper:
            scores[i] = (v - upper) / (iqr + 1e-9)
    mx = scores.max()
    return scores / (mx + 1e-9)

def isolation_forest(series):
    from sklearn.ensemble import IsolationForest
    n = len(series)
    window = min(50, n // 4)
    feats = []
    for i in range(n):
        w = series[max(0, i-window):i+1]
        feats.append([w.mean(), w.std()+1e-9, series[i], i/n])
    feats = np.array(feats)
    clf = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
    clf.fit(feats)
    raw = -clf.score_samples(feats)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

def forecast_residual(series, p=10):
    n = len(series)
    if n < p + 2:
        p = max(1, n // 3)
    X, y = [], []
    for i in range(p, n):
        X.append(series[i-p:i][::-1])
        y.append(series[i])
    X, y = np.array(X), np.array(y)
    lam = 1.0
    XtX = X.T @ X + lam * np.eye(p)
    Xty = X.T @ y
    coef = np.linalg.solve(XtX, Xty)
    preds = np.concatenate([series[:p], X @ coef])
    residuals = np.abs(series - preds)
    mad = np.median(residuals)
    scores = residuals / (1.4826 * mad + 1e-9)
    return scores / (scores.max() + 1e-9)

def level_shift(series, window=20):
    n = len(series)
    scores = np.zeros(n)
    for i in range(window, n - window):
        left = series[i-window:i]
        right = series[i:i+window]
        pooled_std = np.sqrt((left.std()**2 + right.std()**2) / 2) + 1e-9
        scores[i] = abs(right.mean() - left.mean()) / pooled_std
    return scores / (scores.max() + 1e-9)

def matrix_profile_discord(series, m=20):
    n = len(series)
    if m >= n // 2:
        m = max(4, n // 4)
    k = n - m + 1
    if k < 2:
        return np.zeros(n)
    # z-normalize all subsequences at once
    subs = np.array([series[i:i+m] for i in range(k)], dtype=float)
    mu = subs.mean(axis=1, keepdims=True)
    std = subs.std(axis=1, keepdims=True) + 1e-9
    subs_z = (subs - mu) / std
    # Sample random neighbors instead of O(n^2) exhaustive search
    rng = np.random.default_rng(42)
    excl = max(1, m // 4)
    mp = np.full(k, np.inf)
    n_samples = min(30, k - 1)
    for i in range(k):
        pool = np.arange(k)
        pool = pool[np.abs(pool - i) > excl]
        if len(pool) == 0:
            mp[i] = 0.0
            continue
        chosen = pool[rng.choice(len(pool), size=min(n_samples, len(pool)), replace=False)]
        diffs = subs_z[chosen] - subs_z[i]
        mp[i] = float(np.sqrt((diffs**2).sum(axis=1)).min())
    scores = np.zeros(n)
    scores[:k] = mp
    scores[k:] = mp[-1] if k > 0 else 0.0
    return scores / (scores.max() + 1e-9)

DETECTORS = {
    'Robust Z-Score': robust_zscore,
    'IQR Fence': iqr_fence,
    'Isolation Forest': isolation_forest,
    'Forecast Residual': forecast_residual,
    'Level Shift': level_shift,
    'Matrix Profile': matrix_profile_discord,
}

# ─── Synthetic injection ───────────────────────────────────────────────────────

def inject_anomalies(series, seed=42):
    rng = np.random.default_rng(seed)
    n = len(series)
    std = series.std()
    injected = series.copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(3):
        idx = rng.integers(10, n-10)
        injected[idx] += rng.choice([-1, 1]) * std * rng.uniform(4, 6)
        labels[idx] = 1
    start = rng.integers(n//4, n//2)
    end = min(start + rng.integers(15, 30), n)
    injected[start:end] += std * rng.uniform(2.5, 4)
    labels[start:end] = 1
    vstart = rng.integers(n//2, 3*n//4)
    vend = min(vstart + rng.integers(10, 20), n)
    injected[vstart:vend] += rng.normal(0, std * 2, vend - vstart)
    labels[vstart:vend] = 1
    return injected, labels

# ─── Evaluation ───────────────────────────────────────────────────────────────

def compute_f1(preds, labels):
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return float(p), float(r), float(f1)

def point_adjusted_f1(preds, labels):
    adj_preds = preds.copy()
    in_anom, start = False, 0
    windows = []
    for i in range(len(labels)):
        if labels[i] == 1 and not in_anom:
            in_anom = True; start = i
        elif labels[i] == 0 and in_anom:
            in_anom = False; windows.append((start, i))
    if in_anom:
        windows.append((start, len(labels)))
    for s, e in windows:
        if np.any(preds[s:e] == 1):
            adj_preds[s:e] = 1
    return compute_f1(adj_preds, labels)

def auto_threshold(scores, labels):
    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.1, 0.95, 50):
        preds = (scores >= t).astype(int)
        _, _, f1 = compute_f1(preds, labels)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

# ─── Demo data ────────────────────────────────────────────────────────────────

def make_demo_csv():
    rng = np.random.default_rng(0)
    n = 480
    t = np.arange(n)
    temp = 30 + 0.02*t + 5*np.sin(2*np.pi*t/48) + rng.normal(0,1,n)
    pressure = 100 - 0.01*t + 3*np.sin(2*np.pi*t/24) + rng.normal(0,0.5,n)
    vibration = 1 + 0.5*np.sin(2*np.pi*t/12) + rng.normal(0,0.2,n)
    flow = 50 + 2*np.sin(2*np.pi*t/36) + rng.normal(0,2,n)
    temp[100:115] += 15
    vibration[220] += 5; vibration[221] += 4
    flow[350:360] -= 20
    pressure[400] += 10
    times = pd.date_range('2024-01-01', periods=n, freq='h')
    df = pd.DataFrame({'timestamp': times, 'temp_sensor': np.round(temp,2),
                       'pressure': np.round(pressure,2),
                       'vibration': np.round(vibration,3),
                       'flow_rate': np.round(flow,2)})
    return df.to_csv(index=False)

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo_csv')
def demo_csv():
    return make_demo_csv(), 200, {'Content-Type': 'text/csv'}

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        csv_content = data.get('csv', '')
        selected_vars = data.get('variables', [])
        n_votes = int(data.get('n_votes', 2))
        eval_mode = data.get('eval_mode', 'synthetic')
        active_detectors = data.get('active_detectors', list(DETECTORS.keys()))

        df = pd.read_csv(StringIO(csv_content))
        time_col = None
        for c in df.columns:
            if any(k in c.lower() for k in ['time','date','timestamp','datetime','ts']):
                time_col = c
                break

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_col:
            numeric_cols = [c for c in numeric_cols if c != time_col]

        if not selected_vars:
            selected_vars = numeric_cols[:6]
        else:
            selected_vars = [v for v in selected_vars if v in numeric_cols]

        if not selected_vars:
            return jsonify({'error': 'No numeric columns found'}), 400

        n = len(df)

        if time_col:
            try:
                times = pd.to_datetime(df[time_col]).dt.strftime('%Y-%m-%d %H:%M').tolist()
            except:
                times = list(range(n))
        else:
            times = list(range(n))

        results = {}
        all_consensus_votes = np.zeros(n)

        for var in selected_vars:
            series = df[var].ffill().fillna(0).values.astype(float)

            scores_per_det = {}
            thresholds = {}
            inj_series, gt_labels = inject_anomalies(series)

            for det_name in active_detectors:
                if det_name not in DETECTORS:
                    continue
                det_fn = DETECTORS[det_name]
                try:
                    sc = det_fn(series)
                    inj_sc = det_fn(inj_series)
                    t = auto_threshold(inj_sc, gt_labels)
                    scores_per_det[det_name] = sc.tolist()
                    thresholds[det_name] = float(t)
                except Exception:
                    scores_per_det[det_name] = np.zeros(n).tolist()
                    thresholds[det_name] = 0.5

            vote_matrix = np.zeros((len(active_detectors), n))
            for i, det_name in enumerate(active_detectors):
                if det_name in scores_per_det:
                    sc = np.array(scores_per_det[det_name])
                    t = thresholds.get(det_name, 0.5)
                    vote_matrix[i] = (sc >= t).astype(float)

            votes = vote_matrix.sum(axis=0)
            consensus_anom = (votes >= n_votes).astype(int)
            all_consensus_votes += votes

            eval_result = {}
            if eval_mode == 'synthetic':
                inj_scores_combined = np.zeros(n)
                inj_scores_list = {}
                for det_name in active_detectors:
                    if det_name not in DETECTORS:
                        continue
                    try:
                        inj_sc = DETECTORS[det_name](inj_series)
                        t = thresholds.get(det_name, 0.5)
                        inj_scores_list[det_name] = inj_sc
                        inj_scores_combined += (inj_sc >= t).astype(float)
                    except:
                        pass
                inj_consensus = (inj_scores_combined >= n_votes).astype(int)
                p, r, f1 = compute_f1(inj_consensus, gt_labels)
                pa_p, pa_r, pa_f1 = point_adjusted_f1(inj_consensus, gt_labels)

                det_f1s = {}
                for det_name in active_detectors:
                    if det_name in inj_scores_list:
                        inj_sc = inj_scores_list[det_name]
                        t = thresholds.get(det_name, 0.5)
                        preds = (inj_sc >= t).astype(int)
                        _, _, df1 = compute_f1(preds, gt_labels)
                        det_f1s[det_name] = round(df1, 3)

                eval_result = {
                    'mode': 'synthetic',
                    'n_injected': int(gt_labels.sum()),
                    'f1': round(f1, 3), 'precision': round(p, 3), 'recall': round(r, 3),
                    'pa_f1': round(pa_f1, 3), 'pa_precision': round(pa_p, 3), 'pa_recall': round(pa_r, 3),
                    'det_f1s': det_f1s,
                    'gt_labels': gt_labels.tolist(),
                }

            p_ar = min(10, n // 4)
            try:
                X_ar, y_ar = [], []
                for i in range(p_ar, n):
                    X_ar.append(series[i-p_ar:i][::-1])
                    y_ar.append(series[i])
                X_ar = np.array(X_ar)
                y_ar = np.array(y_ar)
                coef = np.linalg.solve(X_ar.T @ X_ar + np.eye(p_ar), X_ar.T @ y_ar)
                ar_pred = np.concatenate([series[:p_ar], X_ar @ coef])
            except:
                ar_pred = series.copy()

            results[var] = {
                'series': series.tolist(),
                'ar_pred': ar_pred.tolist(),
                'scores': scores_per_det,
                'thresholds': thresholds,
                'votes': votes.tolist(),
                'consensus_anomalies': consensus_anom.tolist(),
                'n_anomalies': int(consensus_anom.sum()),
                'anomaly_rate': round(float(consensus_anom.sum()) / n * 100, 1),
                'eval': eval_result,
            }

        vote_dist = {}
        for kk in range(len(active_detectors) + 1):
            vote_dist[str(kk)] = int((all_consensus_votes == kk).sum())

        return jsonify({
            'ok': True,
            'n_points': n,
            'times': times,
            'variables': selected_vars,
            'all_variables': numeric_cols,
            'time_col': time_col,
            'results': results,
            'vote_distribution': vote_dist,
            'detector_names': active_detectors,
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
