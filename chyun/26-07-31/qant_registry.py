"""
qant_registry — 연산자 레지스트리 (구조적 진리).

ChatGPT 피드백 ② 반영: '계약'을 두 층으로 분리한다.
  - REGISTRY (이 파일): 사람이 관리하는 구조적 진리.
      "이 연산자는 존재하고, 인자 몇 개를 받으며, 어느 축에서 동작하는가"
      위반 = E_ 하드 에러. 코퍼스에 없어도 수학적으로 유효하면 여기 추가한다.
  - corpus profile (qant_validate.build_spec 이 관측으로 생성): 통계.
      "코퍼스에서 실제로 어떻게 쓰였는가" — 위반 = W_ 경고(검토 신호)일 뿐.

이 분리로 '관측 안 됨 = 무효' 라는 과잉 차단(내 지적 ②)이 해소된다:
창의적이지만 유효한 조합은 E 없이 W_UNSEEN_* 경고만 받는다.
"""
REGISTRY_VERSION = "0.2.0"
# 0.2.0: ts_ewm 파라미터 의미 명시화(window -> span) + ts_sma_cn 신설.
#   ts_ewm(x, n)      : 지수이동평균, n = span (pandas ewm(span=n) 의미).
#                       구표기 params={"window": n} 은 normalizer 가 span 으로 이관.
#   ts_sma_cn(x, n, m): 중국식 SMA (GTJA 계열). y_t = (m/n)*x_t + (1-m/n)*y_{t-1}
#                       즉 alpha = m/n 인 재귀 평활. 일반 EMA(span)와 다르다.
#                       Wilder 평활(RSI/ATR)은 ts_sma_cn(x, n, 1) 로 표현된다.

# window: "required" | "none"  /  axes: 허용 축 집합 (None = 원소별)
def _e(axes, lo, hi, window="none", extra=None):
    d = {"axes": axes, "min_args": lo, "max_args": hi, "window": window}
    if extra:
        d["extra_params"] = extra
    return d

ELEM = {None}
REGISTRY = {
    # 원소별 산술
    "add": _e(ELEM, 2, 2), "sub": _e(ELEM, 2, 2), "mul": _e(ELEM, 2, 2),
    "div": _e(ELEM, 2, 2), "pow": _e(ELEM, 2, 2),
    "neg": _e(ELEM, 1, 1), "abs": _e(ELEM, 1, 1), "log": _e(ELEM, 1, 1),
    "log1p": _e(ELEM, 1, 1), "sqrt": _e(ELEM, 1, 1), "exp": _e(ELEM, 1, 1),
    "sign": _e(ELEM, 1, 1), "signedpower": _e(ELEM, 2, 2),
    "min": _e(ELEM, 2, 2), "max": _e(ELEM, 2, 2),
    "where": _e(ELEM, 3, 3),
    # 비교/논리
    "lt": _e(ELEM, 2, 2), "gt": _e(ELEM, 2, 2), "le": _e(ELEM, 2, 2),
    "ge": _e(ELEM, 2, 2), "eq": _e(ELEM, 2, 2), "ne": _e(ELEM, 2, 2),
    "or_": _e(ELEM, 2, 2), "and_": _e(ELEM, 2, 2),
    # 횡단면
    "cs_rank": _e({"CS"}, 1, 1), "cs_scale": _e({"CS"}, 1, 1),
    "cs_mean": _e({"CS"}, 1, 1), "cs_std": _e({"CS"}, 1, 1),
    "cs_median": _e({"CS"}, 1, 1),
    # 그룹
    "grp_demean": _e({"GRP"}, 1, 1), "grp_mean": _e({"GRP"}, 1, 1),
    "grp_rank": _e({"GRP"}, 1, 1),
    # 시계열 (단항, window 필수)
    **{op: _e({"TS"}, 1, 1, "required") for op in (
        "ts_delay", "ts_delta", "ts_returns", "ts_mean", "ts_std", "ts_sum",
        "ts_min", "ts_max", "ts_rank", "ts_argmax", "ts_argmin", "ts_product",
        "ts_decay_linear", "ts_slope", "ts_rsquare", "ts_resi", "ts_skew",
        "ts_median", "ts_var", "ts_cumsum", "ts_cumprod")},
    "ts_quantile": _e({"TS"}, 1, 1, "required", extra={"q"}),
    "ts_ewm": _e({"TS"}, 1, 1, "none", extra={"span"}),
    "ts_sma_cn": _e({"TS"}, 1, 1, "required", extra={"m"}),
    # 시계열 (이항, window 필수)
    "ts_corr": _e({"TS"}, 2, 2, "required"),
    "ts_cov": _e({"TS"}, 2, 2, "required"),
}

# 필드 레지스트리 (데이터 스키마 — 하드 진리)
BASE_FIELDS = {"open", "high", "low", "close", "volume", "vwap", "returns", "cap"}
# adv{N} 은 qant_validate.ADV_RE 로 패턴 허용
