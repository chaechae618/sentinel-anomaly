# SENTINEL — 다변량 시계열 이상탐지

**다변량 CSV 시계열을 업로드하면 6개 탐지기가 자동으로 이상을 감지합니다.**

## 기능
- CSV 업로드 (드래그 & 드롭 또는 클릭)
- 데모 데이터 내장 (4변수 · 480포인트)
- 6개 탐지기: Robust Z-Score, IQR Fence, Isolation Forest, Forecast Residual, Level Shift, Matrix Profile
- 합의 투표 (N표 이상일 때만 이상 확정)
- 합성 주입 자동 평가 (Point F1 / Point-Adjusted F1)
- 탐지기 히트맵, 변수별 이상 밀도, 합의 분포 시각화

## 로컬 실행
```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

## Render 배포
1. GitHub에 이 폴더 push
2. Render → New Web Service → GitHub 연결
3. `render.yaml`이 있으면 자동 설정됨
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `gunicorn app:app --workers 2 --timeout 120`
