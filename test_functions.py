"""Unit tests for the pipeline's core functions.

Expected values are taken directly from the original assignment notebook's
documented "Expected output" cells, so a passing suite here is evidence the
refactor preserves the original exercise solutions' behavior.
"""
import numpy as np
import pandas as pd
import pytest

from data_utils import check_for_leakage
from losses import compute_class_freqs, get_weighted_loss
from test_case import get_weighted_loss_test_case


def test_check_for_leakage_true():
    df1 = pd.DataFrame({"patient_id": [0, 1, 2]})
    df2 = pd.DataFrame({"patient_id": [2, 3, 4]})
    assert check_for_leakage(df1, df2, "patient_id") is True


def test_check_for_leakage_false():
    df1 = pd.DataFrame({"patient_id": [0, 1, 2]})
    df2 = pd.DataFrame({"patient_id": [3, 4, 5]})
    assert check_for_leakage(df1, df2, "patient_id") is False


def test_compute_class_freqs():
    labels = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
    ])
    pos_freqs, neg_freqs = compute_class_freqs(labels)
    np.testing.assert_array_almost_equal(pos_freqs, [0.8, 0.4, 0.8])
    np.testing.assert_array_almost_equal(neg_freqs, [0.2, 0.6, 0.2])


def test_get_weighted_loss():
    from keras import backend as K

    # TF1-style session, matching how the original notebook exercised this test.
    sess = K.get_session()
    y_true, w_p, w_n, y_pred_1, y_pred_2 = get_weighted_loss_test_case(sess)

    epsilon = 1  # matches the notebook's test configuration
    loss_fn = get_weighted_loss(w_p, w_n, epsilon)

    l1 = loss_fn(y_true, y_pred_1).eval(session=sess)
    l2 = loss_fn(y_true, y_pred_2).eval(session=sess)

    assert l1 == pytest.approx(-0.4956203, abs=1e-6)
    assert l2 == pytest.approx(-0.4956203, abs=1e-6)
    assert l1 == pytest.approx(l2, abs=1e-6)
