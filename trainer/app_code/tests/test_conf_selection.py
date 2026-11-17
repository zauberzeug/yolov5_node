import numpy as np
import pytest

from ..yolov5.utils.metrics import ap_per_class

NMS_CONF = 0.2


def picked_conf(tp_confs: list[float], fp_confs: list[float], labels: int, min_conf: float) -> float:
    conf = np.array(tp_confs + fp_confs)
    tp = np.repeat(np.array([True] * len(tp_confs) + [False] * len(fp_confs))[:, None], 10, axis=1)
    *_, reported_conf = ap_per_class(tp, conf, np.zeros(len(conf)), np.zeros(labels), names={0: 'a'},
                                     min_conf=min_conf)
    return float(reported_conf)


def test_the_conf_is_not_picked_below_the_nms_conf_when_f1_peaks_at_it():
    """F1 grows down to the lowest prediction, so the curve is flat at its maximum below the NMS confidence."""
    tp_confs = list(np.linspace(NMS_CONF, 0.29, 10))
    fp_confs = [0.9, 0.85, 0.8]

    assert picked_conf(tp_confs, fp_confs, labels=10, min_conf=0.0) == 0.0
    assert picked_conf(tp_confs, fp_confs, labels=10, min_conf=NMS_CONF) == pytest.approx(NMS_CONF, abs=1e-3)


def test_the_conf_is_not_pulled_below_the_nms_conf_by_smoothing():
    tp_confs = list(np.linspace(0.21, 0.24, 10))
    fp_confs = [*np.linspace(NMS_CONF, 0.205, 5), 0.9]

    assert picked_conf(tp_confs, fp_confs, labels=10, min_conf=0.0) < NMS_CONF
    assert picked_conf(tp_confs, fp_confs, labels=10, min_conf=NMS_CONF) >= NMS_CONF


def test_a_peak_above_the_nms_conf_is_picked_as_before():
    tp_confs = list(np.linspace(0.6, 0.7, 10))
    fp_confs = list(np.linspace(NMS_CONF, 0.3, 10))

    assert picked_conf(tp_confs, fp_confs, labels=10, min_conf=NMS_CONF) == \
        picked_conf(tp_confs, fp_confs, labels=10, min_conf=0.0)
