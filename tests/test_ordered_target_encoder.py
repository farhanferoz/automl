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
