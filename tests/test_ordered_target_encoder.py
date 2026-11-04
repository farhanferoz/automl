"""OrderedTargetEncoder: training rows see only earlier rows' targets; other rows get the full training statistic."""

import numpy as np
import pandas as pd
import pytest

from automl_package.preprocessing import OrderedTargetEncoder

PRIOR = 39 / 5  # mean of the targets below


def test_ordered_encoding_by_time_and_full_statistic_for_new_rows() -> None:
    x = pd.DataFrame({"cat": ["a", "a", "b", "a", "b"], "num": [0.0, 1.0, 2.0, 3.0, 4.0]})
    y = np.array([1.0, 3.0, 10.0, 5.0, 20.0])
    time = np.array([1, 2, 1, 2, 3])  # rows 1 and 3 share a time: neither sees the other
    enc = OrderedTargetEncoder(cols=["cat"], smoothing=1.0)
    train = enc.fit_transform(x, y, order=time)
    assert train["cat"].tolist() == pytest.approx([PRIOR, (1 + PRIOR) / 2, PRIOR, (1 + PRIOR) / 2, (10 + PRIOR) / 2])
    assert train["num"].tolist() == x["num"].tolist()  # other columns untouched
    new = enc.transform(pd.DataFrame({"cat": ["a", "b", "z", np.nan], "num": [9.0] * 4}))
    assert new["cat"].tolist() == pytest.approx([(9 + PRIOR) / 4, (30 + PRIOR) / 3, PRIOR, PRIOR])  # unseen -> prior
    # array input with a positional column; without an order a seeded permutation decides, and no row sees itself
    arr = np.column_stack([np.arange(6.0), np.ones(6)])  # every category occurs once
    out = OrderedTargetEncoder(cols=[0], random_state=0).fit_transform(arr, np.arange(6.0))
    assert out[:, 0].tolist() == pytest.approx([2.5] * 6)
    assert np.array_equal(out, OrderedTargetEncoder(cols=[0], random_state=0).fit_transform(arr, np.arange(6.0)))
    # a NaN target is encoded like any row but adds nothing: rows 0, 2 and 3 are the contributions
    gap = OrderedTargetEncoder(cols=["cat"]).fit_transform(pd.DataFrame({"cat": ["a"] * 4}), np.array([4.0, np.nan, 8.0, 0.0]), order=np.arange(4))
    assert gap["cat"].tolist() == pytest.approx([4.0, 4.0, 4.0, 16 / 3])  # prior 4 (mean of 4, 8, 0); row 1 adds nothing to row 2
    # forecasting: rows are origin days, the target of origin o is known on day o + h (h = 2); a row sees targets known strictly before it
    cat = pd.DataFrame({"cat": ["a"] * 6})
    origin, target_day = np.arange(6), np.arange(6) + 2
    series = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    prior = series.mean()
    fc = OrderedTargetEncoder(cols=["cat"]).fit(cat, series, known_at=target_day)
    seen = [[], [], [], [1.0], [1.0, 2.0], [1.0, 2.0, 4.0]]  # origin 3 sees target day 2 only (origin 0), and so on
    expected = [(sum(s) + prior) / (len(s) + 1) for s in seen]
    assert fc.transform(cat, order=origin)["cat"].tolist() == pytest.approx(expected)
    # rows the encoder was not fitted on follow the same rule, a datetime64 scale works, and an unseen category gets the prior
    new = pd.DataFrame({"cat": ["a", "a", "z"]})
    assert fc.transform(new, order=np.array([2, 100, 100]))["cat"].tolist() == pytest.approx([prior, (series.sum() + prior) / 7, prior])
    day = np.datetime64("2026-01-01")
    dated = OrderedTargetEncoder(cols=["cat"]).fit(cat, series, known_at=day + target_day.astype("timedelta64[D]"))
    assert dated.transform(cat, order=day + origin.astype("timedelta64[D]"))["cat"].tolist() == pytest.approx(expected)
    with pytest.raises(ValueError, match="known_at"):
        OrderedTargetEncoder(cols=["cat"]).fit(cat, series).transform(cat, order=origin)
    # None and NaN are the same missing category, whichever spelling was fitted and whichever is transformed
    miss = OrderedTargetEncoder(cols=["cat"]).fit(pd.DataFrame({"cat": np.array(["a", np.nan, None], dtype=object)}), np.array([1.0, 5.0, 9.0]))
    assert miss.transform(pd.DataFrame({"cat": np.array([None, None], dtype=object)}))["cat"].tolist() == pytest.approx([(14 + 5) / 3] * 2)
    assert miss.transform(pd.DataFrame({"cat": np.array([np.nan, "a"], dtype=object)}))["cat"].tolist() == pytest.approx([(14 + 5) / 3, (1 + 5) / 2])
