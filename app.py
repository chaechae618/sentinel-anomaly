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
    if mad == 0: mad = 1e-9
    scores = np.abs(series - med) / (1.4826 * mad)
    return scores / (scores.max() + 1e-9)

def iqr_fence(series):
    q1, q3 = np.percentile(series, 25), np.percentile(series, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    scores = np.zeros(len(series))
    for i, v in enumerate(series):
        if v < lower: scores[i] = (lower - v) / (iqr + 1e-9)
        elif v > upper: scores[i] = (v - upper) / (iqr + 1e-9)
    return scores / (scores.max() + 1e-9)

def isolation_forest(series):
    from sklearn.ensemble import IsolationForest
    n = len(series)
    window = min(50, n // 4)
    feats = []
    for i in range(n):
        w = series[max(0, i-window):i+1]
        feats.append([w.mean(), w.std()+1e-9, series[i], i/n])
    feats = np.array(feats)
    clf = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
    clf.fit(feats)
    raw = -clf.score_samples(feats)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

def forecast_residual(series, p=10):
    n = len(series)
    if n < p + 2: p = max(1, n // 3)
    X, y = [], []
    for i in range(p, n):
        X.append(series[i-p:i][::-1])
        y.append(series[i])
    X, y = np.array(X), np.array(y)
    coef = np.linalg.solve(X.T @ X + np.eye(p), X.T @ y)
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
    if m >= n // 2: m = max(4, n // 4)
    k = n - m + 1
    if k < 2: return np.zeros(n)
    subs = np.array([series[i:i+m] for i in range(k)], dtype=float)
    mu = subs.mean(axis=1, keepdims=True)
    std = subs.std(axis=1, keepdims=True) + 1e-9
    subs_z = (subs - mu) / std
    rng = np.random.default_rng(42)
    excl = max(1, m // 4)
    mp = np.full(k, np.inf)
    n_samples = min(30, k - 1)
    for i in range(k):
        pool = np.arange(k)
        pool = pool[np.abs(pool - i) > excl]
        if len(pool) == 0: mp[i] = 0.0; continue
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

POINT_DETECTORS = ['Robust Z-Score', 'IQR Fence', 'Isolation Forest', 'Forecast Residual']

# ─── Synthetic injection ───────────────────────────────────────────────────────

def inject_anomalies(series, seed=42):
    rng = np.random.default_rng(seed)
    n = len(series)
    std = series.std()
    injected = series.copy()
    labels = np.zeros(n, dtype=int)
    idxs = rng.choice(np.arange(10, n-10), size=10, replace=False)
    for idx in idxs:
        injected[idx] += rng.choice([-1, 1]) * std * rng.uniform(4, 7)
        labels[idx] = 1
    return injected, labels

# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_f1(preds, labels):
    tp = np.sum((preds==1)&(labels==1))
    fp = np.sum((preds==1)&(labels==0))
    fn = np.sum((preds==0)&(labels==1))
    p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9)
    f1 = 2*p*r/(p+r+1e-9)
    return float(p), float(r), float(f1)

def point_adjusted_f1(preds, labels):
    adj = preds.copy()
    in_a, start = False, 0
    windows = []
    for i in range(len(labels)):
        if labels[i]==1 and not in_a: in_a=True; start=i
        elif labels[i]==0 and in_a: in_a=False; windows.append((start,i))
    if in_a: windows.append((start, len(labels)))
    for s,e in windows:
        if np.any(preds[s:e]==1): adj[s:e]=1
    return compute_f1(adj, labels)

def auto_threshold(scores, labels):
    best_f1, best_t = 0, 0.6
    for t in np.linspace(0.2, 0.95, 50):
        preds = (scores >= t).astype(int)
        p, r, f1 = compute_f1(preds, labels)
        if f1 > best_f1 and p >= 0.6:
            best_f1, best_t = f1, t
    return best_t

# ─── Demo data (univariate) ───────────────────────────────────────────────────

def make_demo_csv():
    rng = np.random.default_rng(0)
    n = 480
    t = np.arange(n)
    # 단변량: 온도 센서 하나
    series = 30 + 0.02*t + 5*np.sin(2*np.pi*t/48) + rng.normal(0, 1, n)
    # 이상 주입
    series[100:115] += 15   # 레벨 상승
    series[220] += 12       # 스파이크
    series[350:360] -= 10   # 급락
    series[420] += 10       # 스파이크
    times = pd.date_range('2024-01-01', periods=n, freq='h')
    df = pd.DataFrame({'timestamp': times, 'temp_sensor': np.round(series, 2)})
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
        target_var = data.get('variable', '')
        n_votes = int(data.get('n_votes', 2))
        eval_mode = data.get('eval_mode', 'synthetic')
        active_detectors = data.get('active_detectors', list(DETECTORS.keys()))

        df = pd.read_csv(StringIO(csv_content))

        # 시간 컬럼 자동 감지
        time_col = None
        for c in df.columns:
            if any(k in c.lower() for k in ['time','date','timestamp','datetime','ts']):
                time_col = c; break

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_col:
            numeric_cols = [c for c in numeric_cols if c != time_col]

        if not numeric_cols:
            return jsonify({'error': 'No numeric columns found'}), 400

        # 단변량: 변수 1개 선택
        if target_var and target_var in numeric_cols:
            var = target_var
        else:
            var = numeric_cols[0]

        n = len(df)

        if time_col:
            try:
                times = pd.to_datetime(df[time_col]).dt.strftime('%Y-%m-%d %H:%M').tolist()
            except:
                times = list(range(n))
        else:
            times = list(range(n))

        series = df[var].ffill().fillna(0).values.astype(float)
        inj_series, gt_labels = inject_anomalies(series)

        # 각 탐지기 점수 계산
        scores_per_det = {}
        thresholds = {}
        for det_name in active_detectors:
            if det_name not in DETECTORS: continue
            try:
                sc = DETECTORS[det_name](series)
                inj_sc = DETECTORS[det_name](inj_series)
                t = max(0.5, auto_threshold(inj_sc, gt_labels))
                scores_per_det[det_name] = sc.tolist()
                thresholds[det_name] = float(t)
            except Exception:
                scores_per_det[det_name] = np.zeros(n).tolist()
                thresholds[det_name] = 0.6

        # 합의 투표 (점 이상 강한 탐지기 4개로)
        vote_detectors = [d for d in active_detectors if d in POINT_DETECTORS]
        if not vote_detectors: vote_detectors = active_detectors
        effective_votes = min(n_votes, len(vote_detectors))

        vote_matrix = np.zeros((len(vote_detectors), n))
        for i, det_name in enumerate(vote_detectors):
            if det_name in scores_per_det:
                sc = np.array(scores_per_det[det_name])
                vote_matrix[i] = (sc >= thresholds.get(det_name, 0.6)).astype(float)

        votes = vote_matrix.sum(axis=0)
        consensus_anom = (votes >= effective_votes).astype(int)

        # 합의 분포
        max_v = len(vote_detectors)
        vote_dist = {str(k): int((votes == k).sum()) for k in range(max_v + 1)}

        # 평가
        eval_result = {}
        if eval_mode == 'synthetic':
            inj_combined = np.zeros(n)
            inj_scores_list = {}
            for det_name in vote_detectors:
                if det_name not in DETECTORS: continue
                try:
                    inj_sc = DETECTORS[det_name](inj_series)
                    t = thresholds.get(det_name, 0.6)
                    inj_scores_list[det_name] = inj_sc
                    inj_combined += (inj_sc >= t).astype(float)
                except: pass

            inj_consensus = (inj_combined >= effective_votes).astype(int)
            p, r, f1 = compute_f1(inj_consensus, gt_labels)
            pa_p, pa_r, pa_f1 = point_adjusted_f1(inj_consensus, gt_labels)

            det_f1s = {}
            for det_name in active_detectors:
                if det_name not in DETECTORS: continue
                try:
                    inj_sc = DETECTORS[det_name](inj_series)
                    t = thresholds.get(det_name, 0.6)
                    preds = (inj_sc >= t).astype(int)
                    _, _, df1 = compute_f1(preds, gt_labels)
                    det_f1s[det_name] = round(df1, 3)
                except: det_f1s[det_name] = 0.0

            eval_result = {
                'mode': 'synthetic',
                'n_injected': int(gt_labels.sum()),
                'f1': round(f1, 3), 'precision': round(p, 3), 'recall': round(r, 3),
                'pa_f1': round(pa_f1, 3), 'pa_precision': round(pa_p, 3), 'pa_recall': round(pa_r, 3),
                'det_f1s': det_f1s,
                'gt_labels': gt_labels.tolist(),
            }

        # AR 예측선
        p_ar = min(10, n // 4)
        try:
            X_ar, y_ar = [], []
            for i in range(p_ar, n):
                X_ar.append(series[i-p_ar:i][::-1])
                y_ar.append(series[i])
            X_ar, y_ar = np.array(X_ar), np.array(y_ar)
            coef = np.linalg.solve(X_ar.T @ X_ar + np.eye(p_ar), X_ar.T @ y_ar)
            ar_pred = np.concatenate([series[:p_ar], X_ar @ coef])
        except:
            ar_pred = series.copy()

        # 기술통계
        stats = {
            'mean': round(float(series.mean()), 3),
            'std': round(float(series.std()), 3),
            'min': round(float(series.min()), 3),
            'max': round(float(series.max()), 3),
            'n_anomalies': int(consensus_anom.sum()),
            'anomaly_rate': round(float(consensus_anom.sum()) / n * 100, 1),
        }

        return jsonify({
            'ok': True,
            'n_points': n,
            'times': times,
            'variable': var,
            'all_variables': numeric_cols,
            'time_col': time_col,
            'series': series.tolist(),
            'ar_pred': ar_pred.tolist(),
            'scores': scores_per_det,
            'thresholds': thresholds,
            'votes': votes.tolist(),
            'consensus_anomalies': consensus_anom.tolist(),
            'vote_distribution': vote_dist,
            'detector_names': active_detectors,
            'stats': stats,
            'eval': eval_result,
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
