"""
Dataset: CSV를 읽어 에이전트 타입에 맞게 변환해 들고 있는 DataFrame 래퍼.

파일 규약
---------
data_dir 안에 `{type소문자}_{subset}.csv` 이름으로 파일을 둔다.
예) Dataset(type='ML', subset='test-sample') → data/ml_test-sample.csv

ML 타입 CSV 컬럼 규약
---------------------
- datetime : 시각 (필수, 시간순 정렬 기준)
- open/high/low/close/volume : 시세 (Runner의 체결가 계산에 close 필수)
- feat_* : 피쳐 컬럼 (예: feat_ret_1, feat_ma_10). 접두사로 자동 인식
- target : 학습용 타깃 (예: 다음 봉 상승=1/하락=0). test에도 있으면 평가에 사용
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

#: Dataset이 인식하는 에이전트 타입들.
#: 나머지는 자리만 잡아둠 (추가 시 _TRANSFORMS에 함수 등록).
SUPPORTED_TYPES = ("ML", "RULE", "RL") # ML 말고 다른건 뭔지 모르겠음 ㅎㅎ

RESERVED_COLS = {"datetime", "open", "high", "low", "close", "volume", "code", "name", "target"}


class Dataset:
    """pandas DataFrame을 내부에 지닌 래퍼.

    Parameters
    ----------
    type : str
        에이전트 타입. 'ML' 등 (SUPPORTED_TYPES 참고)
    subset : str
        데이터 부분집합 식별자. 'train-sample', 'test-1234' 등.
        관례상 train-*/test-* 접두사로 학습/평가를 구분한다.
    data_dir : str | Path
        CSV가 들어있는 폴더. 기본 './data'
    df : pd.DataFrame, optional
        CSV 대신 DataFrame을 직접 주입할 수도 있다(테스트/실험용).
    """

    def __init__(
        self,
        type: str,
        subset: str,
        data_dir: str | Path = "./data",
        df: Optional[pd.DataFrame] = None,
    ):
        type = type.upper()
        if type not in SUPPORTED_TYPES:
            raise ValueError(f"지원하지 않는 type: {type!r}. 가능: {SUPPORTED_TYPES}")

        self.type = type
        self.subset = subset
        self.data_dir = Path(data_dir)

        raw = df if df is not None else self._read_csv()
        self._df = self._transform(raw)

    # ------------------------------------------------------------------ IO
    @property
    def path(self) -> Path:
        return self.data_dir / f"{self.type.lower()}_{self.subset}.csv"

    def _read_csv(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"데이터 파일이 없습니다: {self.path}\n"
                f"  → data_dir 경로를 확인하거나 scripts/make_ml_dataset.py 로 생성하세요."
            )
        return pd.read_csv(self.path)

    # ------------------------------------------------------------ transform
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """타입별 전처리. 공통: datetime 파싱 + 시간순 정렬 + 결측 제거."""
        if "datetime" not in df.columns:
            raise ValueError("CSV에 'datetime' 컬럼이 필요합니다.")
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        if self.type == "ML":
            # 피쳐/타깃에 결측이 있는 행은 학습·판단이 불가능하므로 제거
            cols = self.feature_columns_of(df) + (["target"] if "target" in df.columns else [])
            if cols:
                df = df.dropna(subset=cols).reset_index(drop=True)
        # RULE / RL 타입 변환은 필요해지면 여기에 추가
        return df

    # ------------------------------------------------------------- 조회 API
    @staticmethod
    def feature_columns_of(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c.startswith("feat_")]

    @property
    def feature_columns(self) -> list[str]:
        """feat_ 접두사가 붙은 피쳐 컬럼 이름 목록."""
        return self.feature_columns_of(self._df)

    @property
    def has_target(self) -> bool:
        return "target" in self._df.columns

    @property
    def df(self) -> pd.DataFrame:
        """내부 DataFrame의 복사본. (원본 오염 방지)"""
        return self._df.copy()

    def X(self) -> pd.DataFrame:
        """피쳐 행렬 (ML 학습용)."""
        return self._df[self.feature_columns].copy()

    def y(self) -> pd.Series:
        """타깃 벡터 (ML 학습용)."""
        if not self.has_target:
            raise ValueError("이 subset에는 'target' 컬럼이 없습니다.")
        return self._df["target"].copy()

    # ---------------------------------------------------------- Runner 통로
    def iter_steps(self) -> Iterator[dict]:
        """시간순으로 한 시점(row)씩 dict로 내보낸다. Runner가 사용."""
        for row in self._df.itertuples(index=False):
            yield row._asdict()

    # ------------------------------------------------------------- 편의기능
    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return (
            f"Dataset(type={self.type!r}, subset={self.subset!r}, "
            f"rows={len(self)}, features={len(self.feature_columns)}, "
            f"target={self.has_target})"
        )
