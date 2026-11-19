import pandas as pd
import pytest

from banksim.data import DataError, validate_records
from banksim.panel import build_daily_panel
from banksim.visibility import read_visible_window, require_continuous_observed, visible_attributes


def event(step, *, amount="10", fraud=0, age="2", gender="F"):
    return {"step": step, "customer": "T", "age": age, "gender": gender,
            "zipcodeOri": "28007", "merchant": "M", "zipMerchant": "28007",
            "category": "es_food", "amount": amount, "fraud": fraud}


@pytest.mark.acceptance("D13")
def test_empty_open_window_cannot_recover_hidden_or_future_attributes():
    events = validate_records([event(0), event(10, age="6", gender="M")])
    panel, _ = build_daily_panel(events)
    window = read_visible_window(events, panel, "T", 5, Lc=3, H=2)
    assert window.history.step.tolist() == [3, 4, 5]
    assert window.Tobs == 3
    assert window.history.target_cents.tolist() == [0, 0, 0]
    assert window.attributes == {"age": "U", "gender": "U", "attribute_source_step": None,
                                 "attribute_source_raw_row_id": None, "attribute_status": "UNKNOWN"}
    # No ordinary query may read the future conflicting record either.
    assert read_visible_window(events, panel, "T", 5, H=2).attributes["age"] == "2"


@pytest.mark.acceptance("D14")
def test_fraud_only_open_window_provides_visible_attributes():
    events = validate_records([event(0), event(4, amount="999", fraud=1)])
    panel, _ = build_daily_panel(events)
    window = read_visible_window(events, panel, "T", 5, Lc=2, H=2)
    assert window.history.step.tolist() == [4, 5]
    assert window.history.target_cents.tolist() == [0, 0]
    assert window.attributes["age"] == "2"
    assert window.attributes["gender"] == "F"
    assert window.attributes["attribute_source_step"] == 4
    assert window.attributes["attribute_source_raw_row_id"] == 2


@pytest.mark.acceptance("D15")
@pytest.mark.parametrize("kind", ["missing_row", "masked_day", "null_day", "duplicate_day"])
def test_gap_cannot_be_spliced(kind):
    events = validate_records([event(0)])
    panel, _ = build_daily_panel(events)
    if kind == "missing_row":
        panel = panel.loc[panel.step.ne(3)]
    elif kind == "masked_day":
        panel.loc[panel.step.eq(3), "observed"] = False
    elif kind == "null_day":
        panel.loc[panel.step.eq(3), "target_cents"] = pd.NA
    else:
        panel = pd.concat([panel, panel.loc[panel.step.eq(3)]], ignore_index=True)
    with pytest.raises(DataError) as caught:
        read_visible_window(events, panel, "T", 5, Lc=4, H=1)
    assert caught.value.code == "DATA_GAP"


@pytest.mark.acceptance("D15")
def test_labels_end_at_179_and_inference_needs_no_labels():
    events = validate_records([event(0)])
    panel, _ = build_daily_panel(events)
    assert read_visible_window(events, panel, "T", 172, H=7).labels.step.tolist() == list(range(173, 180))
    for origin in (173, 179):
        with pytest.raises(DataError) as caught:
            read_visible_window(events, panel, "T", origin, H=7)
        assert caught.value.code == "DATA_GAP"
    inference = read_visible_window(events, panel, "T", 179, Lc=14, H=7, require_labels=False)
    assert inference.labels is None
    assert inference.history.step.max() == 179
    panel.loc[panel.step.eq(6), "observed"] = False
    with pytest.raises(DataError) as caught:
        read_visible_window(events, panel, "T", 5, H=2)
    assert caught.value.code == "DATA_GAP"


@pytest.mark.acceptance("D15")
def test_preappearance_not_admitted_to_exact_cold_length():
    events = validate_records([event(3)])
    panel, _ = build_daily_panel(events)
    with pytest.raises(DataError) as caught:
        read_visible_window(events, panel, "T", 5, Lc=4, H=1)
    assert caught.value.code == "DATA_GAP"
    with pytest.raises(DataError) as caught:
        read_visible_window(events, panel, "T", 2, require_labels=False)
    assert caught.value.code == "NO_HISTORY"
    assert read_visible_window(events, panel, "T", 5, Lc=3, H=1).Tobs == 3


@pytest.mark.acceptance("D13", "D14")
def test_exact_open_interval_endpoints_and_zero_amount_record():
    events = validate_records([event(0), event(3, amount="0"), event(5, amount="0"), event(6)])
    panel, _ = build_daily_panel(events)
    window = read_visible_window(events, panel, "T", 5, Lc=3, H=1)
    assert window.history.step.tolist() == [3, 4, 5]
    assert window.Tobs == 3
    assert window.history.normal_tx_count.tolist() == [1, 0, 1]
    assert window.history.target_cents.tolist() == [0, 0, 0]
    assert window.attributes["attribute_source_step"] == 3
    assert visible_attributes(events, "T", 5, 5)["attribute_source_step"] == 5
