"""
OHLCV CSV → ML용 train/test 데이터셋 생성기
============================================
KIS 수집기가 만든 CSV(예: Data/1min/005930.csv)를 입력으로 받아
피쳐(feat_*)와 타깃(target)을 붙이고, '시점 기준'으로 train/test를 나눠
backtesting_utils의 Dataset이 바로 읽을 수 있는 형식으로 저장한다.

사용 예
-------
# (1) 한 종목 파일
python scripts/make_ml_dataset.py \\
    --input ../KIS/Data/1min/005930.csv \\
    --split-date 2026-07-08 \\
    --tag samsung1m \\
    --outdir ./data

→ ./data/ml_train-samsung1m.csv, ./data/ml_test-samsung1m.csv 생성
→ Dataset(type='ML', subset='train-samsung1m') 으로 로드 가능

# (2) 폴더를 통째로 (200개 종목 CSV 일괄 변환) — KIS 개별 종목 폴더 지정
python scripts/make_ml_dataset.py \\
    --input ../KIS/Data/1min \\
    --split-date 2026-07-15

→ 폴더 안 000660.csv, 005930.csv … 각각에 대해
   ./data/ml_train-000660.csv / ml_test-000660.csv … 생성 (tag = 파일명=종목코드)
→ Dataset(type='ML', subset='test-005930') 으로 로드
   특정 종목만: --codes 005930,000660

--input 없이 실행하면 합성(가짜) 시세를 만들어 sample 데이터셋을 생성한다.
(레포에 실데이터를 올리기 애매할 때 파이프라인 검증용)

주의: KIS 수집기의 통합본(kospi200_*_all.csv)은 쓰지 말 것.
200종목이 한 파일에 세로로 쌓여 있어 피쳐(rolling/pct_change)가 종목 경계를
넘어 계산돼 오염된다. 반드시 종목별 개별 CSV(Data/1min/{code}.csv)를 쓴다.

주의: 시계열은 절대 무작위로 섞어 나누지 않는다.
split-date 이전 = train, 이후 = test (룩어헤드 방지).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 피쳐 생성
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """결정론적 피쳐 엔지니어링. 모든 피쳐는 '해당 시점까지의 정보'만 사용한다."""
    df = df.copy()
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    # 수익률 계열
    df["feat_ret_1"] = c.pct_change(1)
    df["feat_ret_5"] = c.pct_change(5)
    df["feat_ret_20"] = c.pct_change(20)

    # 이동평균 괴리율 (현재가가 이동평균 대비 몇 % 위/아래인가)
    for w in (5, 20, 60):
        ma = c.rolling(w).mean()
        df[f"feat_ma{w}_gap"] = c / ma - 1.0

    # 변동성 (수익률 표준편차)
    df["feat_vol_20"] = df["feat_ret_1"].rolling(20).std()

    # 거래량 변화율 (20봉 평균 대비)
    df["feat_vol_ratio"] = v / v.rolling(20).mean()

    # RSI(14) — 표준 지표 예시
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["feat_rsi_14"] = 100 - 100 / (1 + rs)

    return df


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """타깃: horizon봉 뒤 종가가 오르면 1, 아니면 0.

    shift(-horizon)은 미래 값을 참조하지만, 이것은 '정답 라벨'이므로 허용.
    (피쳐에 미래 정보가 들어가면 안 되는 것과 구분할 것)
    마지막 horizon개 행은 정답을 알 수 없으므로 제거된다.
    """
    df = df.copy()
    future = df["close"].shift(-horizon)
    df["target"] = (future > df["close"]).astype(int)
    return df.iloc[:-horizon] if horizon > 0 else df


# ------------------------------------------------------------- 합성 데이터
def make_synthetic_ohlcv(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """실데이터가 없을 때 쓰는 합성 분봉. 랜덤워크 + 약한 추세/평균회귀 혼합."""
    rng = np.random.default_rng(seed)
    dt_index = pd.date_range("2026-06-01 09:00", periods=n, freq="1min")

    ret = rng.normal(0, 0.0015, n)
    trend = 0.00002 * np.sin(np.arange(n) / 150)  # 완만한 사이클
    close = 70000 * np.exp(np.cumsum(ret + trend))

    spread = np.abs(rng.normal(0, 0.001, n)) * close
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(1_000, 50_000, n).astype(float)

    return pd.DataFrame(
        {"datetime": dt_index, "open": open_.round(0), "high": high.round(0),
         "low": low.round(0), "close": close.round(0), "volume": volume}
    )


# --------------------------------------------------------------------- main
def build(input_csv: str | None, split_date: str, tag: str, outdir: str,
          horizon: int = 1) -> tuple[Path, Path]:
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    if input_csv:
        df = pd.read_csv(input_csv)
        keep = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
    else:
        print("[입력 없음] 합성 데이터를 생성합니다 (파이프라인 검증용)")
        df = make_synthetic_ohlcv()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df = add_features(df)
    df = add_target(df, horizon=horizon)
    df = df.dropna().reset_index(drop=True)  # 롤링 워밍업 구간 제거

    split = pd.Timestamp(split_date)
    # 지정한 분할 시점이 데이터 범위 밖이면(특히 합성 데이터) 80% 지점으로 자동 조정
    if not (df["datetime"].min() < split < df["datetime"].max()):
        split = df["datetime"].quantile(0.8)
        print(f"[자동 조정] split-date가 범위 밖 → 80% 지점 {split} 사용")
    train = df[df["datetime"] < split]
    test = df[df["datetime"] >= split]
    if train.empty or test.empty:
        raise SystemExit(
            f"분할 결과가 비었습니다 (train {len(train)} / test {len(test)}). "
            f"--split-date {split_date} 가 데이터 범위 안인지 확인하세요: "
            f"{df['datetime'].min()} ~ {df['datetime'].max()}"
        )

    train_path = outdir_p / f"ml_train-{tag}.csv"
    test_path = outdir_p / f"ml_test-{tag}.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    n_feat = sum(c.startswith("feat_") for c in df.columns)
    print(f"[완료] 피쳐 {n_feat}개, 타깃 상승비율 train {train['target'].mean():.1%} / test {test['target'].mean():.1%}")
    print(f"  train: {train_path}  ({len(train):,}행, ~{split} 이전)")
    print(f"  test : {test_path}  ({len(test):,}행, {split} 이후)")
    print(f"  → Dataset(type='ML', subset='train-{tag}') 로 로드하세요.")
    return train_path, test_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None,
                    help="OHLCV CSV 파일 또는 종목별 CSV가 든 폴더 (없으면 합성 데이터)")
    ap.add_argument("--split-date", default="2026-06-25", help="이 시각 이후 = test (YYYY-MM-DD)")
    ap.add_argument("--tag", default=None,
                    help="subset 태그. 파일이면 생략 시 파일명 사용, 폴더면 각 파일명(종목코드) 사용")
    ap.add_argument("--outdir", default="./data", help="출력 폴더")
    ap.add_argument("--horizon", type=int, default=1, help="타깃 예측 지평 (N봉 뒤)")
    ap.add_argument("--codes", default=None,
                    help="폴더 입력일 때 특정 종목만 (콤마 구분, 예: 005930,000660)")
    args = ap.parse_args()

    # 폴더 입력 → 안의 CSV들을 일괄 처리 (tag = 파일명 = 종목코드)
    if args.input and Path(args.input).is_dir():
        files = sorted(Path(args.input).glob("*.csv"))
        if args.codes:
            wanted = {c.strip() for c in args.codes.split(",")}
            files = [f for f in files if f.stem in wanted]
        if not files:
            raise SystemExit(f"CSV를 찾지 못했습니다: {args.input} (--codes 필터 확인)")
        print(f"[일괄 처리] {len(files)}개 종목 CSV 변환 시작")
        ok, skipped = 0, []
        for f in files:
            try:
                build(str(f), args.split_date, f.stem, args.outdir, args.horizon)
                ok += 1
            except SystemExit as e:  # 분할 실패 등은 건너뛰고 계속
                print(f"  [건너뜀] {f.name}: {e}")
                skipped.append(f.stem)
        print(f"\n[일괄 완료] 성공 {ok} / 건너뜀 {len(skipped)}"
              + (f" ({', '.join(skipped)})" if skipped else ""))
        return

    # 단일 파일 / 합성 데이터. tag 미지정 시 파일명 사용(합성은 'sample')
    tag = args.tag or (Path(args.input).stem if args.input else "sample")
    build(args.input, args.split_date, tag, args.outdir, args.horizon)


if __name__ == "__main__":
    main()
