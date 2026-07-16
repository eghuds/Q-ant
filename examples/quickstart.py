"""
    python examples/quickstart.py

data/ 에 sample 데이터셋이 없으면 먼저 생성:
    python scripts/make_ml_dataset.py
"""

import sys
from pathlib import Path

# 레포를 pip install 하지 않고도 예제가 돌도록 경로 추가 (설치했다면 불필요)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backtesting_utils import (
    Dataset, Runner, MLModelAgent, MovingAverageCrossAgent, BuyAndHoldAgent,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ----------------------------------------------------------------------
# 초간단 ML 모델 (외부 라이브러리 없이 numpy 로지스틱 회귀)
# 실전에서는 B가 sklearn/xgboost 등으로 학습한 모델을 그대로 꽂으면 된다.
# predict / predict_proba 인터페이스만 맞으면 MLModelAgent가 감싸준다.
# ----------------------------------------------------------------------
class TinyLogisticModel:
    def __init__(self, lr=0.1, epochs=300):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu, self.sigma = X.mean(0), X.std(0) + 1e-9  # 표준화 파라미터
        Xn = (X - self.mu) / self.sigma
        self.w = np.zeros(Xn.shape[1])
        self.b = 0.0
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-(Xn @ self.w + self.b)))
            grad = Xn.T @ (p - y) / len(y)
            self.w -= self.lr * grad
            self.b -= self.lr * (p - y).mean()
        return self

    def predict_proba(self, X):
        Xn = (np.asarray(X, dtype=float) - self.mu) / self.sigma
        p = 1 / (1 + np.exp(-(Xn @ self.w + self.b)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def main():
    # ===== 회의록의 4줄 =====================================================
    # 1) 학습 데이터로 모델 학습
    train = Dataset(type="ML", subset="train-sample", data_dir=DATA_DIR)
    model = TinyLogisticModel().fit(train.X(), train.y())

    # 2) 테스트 데이터 + Runner + 에이전트 + 실행
    data = Dataset(type="ML", subset="test-sample", data_dir=DATA_DIR)
    runner = Runner(data, periods_per_year=252 * 381)  # 1분봉 기준 연율화

    agent = MLModelAgent(model, feature_columns=train.feature_columns)

    result = runner.submit(agent)
    # ========================================================================

    print("데이터:", data)
    print("ML 에이전트     :", result)

    # 기준선과 비교 (같은 데이터, 다른 에이전트)
    print("이동평균 교차   :", runner.submit(MovingAverageCrossAgent(5, 20)))
    print("바이앤홀드      :", runner.submit(BuyAndHoldAgent()))

    # 거래 로그 샘플
    print("\n거래 로그 앞 5건:")
    print(result.trade_log.head().to_string(index=False))


if __name__ == "__main__":
    main()
