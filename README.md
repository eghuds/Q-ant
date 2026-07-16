# backtesting-utils

백테스팅 SDK. 에이전트(ML/룰베이스)를 공통 규격으로 정의하고,
CSV 데이터셋 위에서 백테스트를 실행해 성과 지표를 반환한다.

## 설치

```bash
# 방법 1: GitHub에서 바로 설치
pip install git+https://github.com/<아이디>/backtesting-utils.git

# 방법 2: 레포를 클론해서 개발 모드로 설치 (코드 수정하며 쓸 때)
git clone https://github.com/<아이디>/backtesting-utils.git
cd backtesting-utils
pip install -e .
```

## 30초 사용법

```python
from backtesting_utils import Dataset, Runner, MLModelAgent

data = Dataset(type='ML', subset='test-sample')   # csv data 로드
runner = Runner(data)

agent = MLModelAgent(model, feature_columns=[...])  # 학습된 모델을 감싼 에이전트
result = runner.submit(agent)

print(result)                # 수익률, MDD, 샤프, 거래수, 승률
print(result.trade_log)      # 거래 상세 로그 (DataFrame)
result.equity_series()       # 자산 곡선 (시각화용 Series)
```

바로 실행해보기:

```bash
python scripts/make_ml_dataset.py     # sample 데이터셋 생성 (합성 데이터)
python examples/quickstart.py        # ML/이동평균/바이앤홀드 에이전트 비교 실행
```

## 구성

| 경로 | 내용 |
|---|---|
| `backtesting_utils/dataset.py` | `Dataset` — CSV 로드, 타입별 변환, DataFrame 래퍼 |
| `backtesting_utils/agent.py` | `BaseAgent` 규격 + 예시 에이전트 3종 |
| `backtesting_utils/runner.py` | `Runner` — 백테스트 루프, `BacktestResult` |
| `backtesting_utils/metrics.py` | 수익률 / MDD / 샤프 / 승률 |
| `scripts/make_ml_dataset.py` | OHLCV CSV → 피쳐+타깃 → train/test 분할 |
| `examples/quickstart.py` | 회의록 4줄 예시가 실제로 도는 파일 |
| `tests/` | 기본 동작 검증 (`pytest tests/ -v`) |

## 데이터 규약

`Dataset(type='ML', subset='test-1234')` → `data/ml_test-1234.csv` 를 읽는다.

ML 타입 CSV 컬럼:

- `datetime` — 필수. 시간순 정렬 기준
- `open, high, low, close, volume` — 시세 (Runner 체결가 계산)
- `feat_*` — 피쳐 (접두사로 자동 인식)
- `target` — 타깃 (예: 다음 봉 상승=1). train에는 필수

**train/test는 반드시 시점 기준으로 분할한다** (무작위 분할 금지 — 룩어헤드 편향).
`scripts/make_ml_dataset.py --split-date 2026-07-01` 이 이 규칙을 강제한다.

실데이터로 만들려면 (KIS 수집기의 종목별 CSV 사용):

```bash
# 한 종목만
python scripts/make_ml_dataset.py \
    --input ../KIS/Data/1min/005930.csv \
    --split-date 2026-07-15          # 이 시각 이후가 test. 데이터 날짜 범위 안이어야 함

# 폴더 통째로 (200종목 일괄) — tag는 파일명(종목코드)이 자동으로 붙는다
python scripts/make_ml_dataset.py --input ../KIS/Data/1min --split-date 2026-07-15
#   → data/ml_train-005930.csv, ml_test-005930.csv, … 생성
#   → Dataset(type='ML', subset='test-005930') 로 로드
#   특정 종목만: --codes 005930,000660
```

> KIS 수집기의 통합본(`kospi200_*_all.csv`)은 쓰지 않는다. 200종목이 한 파일에
> 세로로 쌓여 있어 피쳐(rolling/pct_change)가 종목 경계를 넘어 오염되고, `code`의
> 앞자리 0도 사라진다. 반드시 종목별 개별 CSV(`Data/1min/{code}.csv`)를 쓸 것.

## 에이전트 규격

```python
from backtesting_utils import BaseAgent

class MyAgent(BaseAgent):
    def reset(self):
        """백테스트 시작 전 내부 메모리 초기화 (Runner가 자동 호출)"""

    def act(self, observation: dict) -> str:
        """한 시점 데이터를 받아 'BUY' / 'SELL' / 'HOLD' 반환"""
        return "HOLD"
```

sklearn 스타일 모델(`predict`/`predict_proba`)은 `MLModelAgent`로 감싸면 바로 사용 가능.

## Runner 동작 방식과 v0.1 가정

- **체결**: 기본 `next_open` — t시점 데이터를 보고 낸 주문은 t+1 시가에 체결 (룩어헤드 방지).
  `Runner(..., execution='same_close')` 로 단순 모드 선택 가능.
- **수수료**: 편도 `fee_rate` (기본 0.015%).
- 단일 종목, 전량 매수/전량 매도, 공매도·슬리피지 없음 — 이후 버전에서 확장 예정 (TODO).