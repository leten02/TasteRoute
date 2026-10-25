from tasteroute.item_extras import attribute_features, is_open_at, parse_hours


def test_attribute_features_parse_yelp_strings():
    f = attribute_features({
        "OutdoorSeating": "True", "HasTV": "False", "WiFi": "u'free'", "Alcohol": "'full_bar'",
        "Ambience": "{'romantic': True, 'casual': False}", "GoodForMeal": "None", "NoiseLevel": "u'quiet'",
    })
    assert f["attr_OutdoorSeating"] == 1 and f["attr_HasTV"] == 0
    assert f["attr_WiFi_free"] == 1 and f["attr_Alcohol_full_bar"] == 1 and f["attr_Alcohol_beer_and_wine"] == 0
    assert f["attr_Ambience_romantic"] == 1 and f["attr_Ambience_casual"] == 0
    assert f["attr_GoodForMeal_brunch"] == 0 and f["attr_NoiseLevel_quiet"] == 1
    assert attribute_features(None)["attr_OutdoorSeating"] == 0


def test_hours_past_midnight_and_all_day():
    h = parse_hours({"Friday": "17:0-2:0", "Monday": "0:0-0:0", "Tuesday": "bad"})
    assert h["Friday"] == (17 * 60, 26 * 60)
    assert h["Monday"] == (0, 24 * 60) and "Tuesday" not in h
    assert is_open_at(h, "Friday", 20 * 60)
    assert is_open_at(h, "Saturday", 60)  # Friday's opening runs to 2 am
    assert not is_open_at(h, "Saturday", 3 * 60)
    assert not is_open_at(h, "Friday", 10 * 60)


def test_unknown_hours_count_as_open():
    assert is_open_at({}, "Sunday", 9 * 60)


def test_attribute_matrix_idf_and_combine():
    import numpy as np
    import pandas as pd
    from scipy import sparse

    from tasteroute.item_extras import attribute_matrix, combine_features

    extras = pd.DataFrame({"attr_A": [1.0, 1.0, 0.0], "attr_B": [1.0, 0.0, 0.0], "hours": [{}, {}, {}]}, index=[0, 1, 2])
    A, cols = attribute_matrix(extras, 4)
    assert cols == ["attr_A", "attr_B"] and A.shape == (4, 2)
    assert A[0, 1] > A[0, 0] > 0  # the rarer attribute B weighs more
    assert A[3].nnz == 0  # item without a record
    C = combine_features(sparse.csr_matrix(np.ones((4, 3))), A, 0.5)
    assert C.shape == (4, 5) and np.isclose(C[0, 4], 0.5 * A[0, 1])


def test_open_mask_and_hours_text():
    import pandas as pd

    from tasteroute.item_extras import hours_today, open_mask

    extras = pd.DataFrame({"hours": [{"Monday": (600, 1200)}, {}, {"Monday": (0, 1440)}]}, index=[0, 1, 2])
    m = open_mask(extras, 4, "Monday", 700, stay_minutes=60)
    assert list(m) == [True, True, True, True]
    m = open_mask(extras, 4, "Monday", 1170, stay_minutes=60)  # 19:30 + 1 h passes 20:00
    assert list(m) == [False, True, True, True]
    assert hours_today({"Monday": (600, 1200)}, "Monday") == "10:00-20:00"
    assert hours_today({"Monday": (0, 1440)}, "Monday") == "24시간"
    assert hours_today({}, "Monday") is None


def test_stay_past_midnight_rolls_over_to_next_day():
    from tasteroute.item_extras import is_open_at

    both = {"Monday": (0, 1440), "Tuesday": (0, 1440)}
    monday_only = {"Monday": (0, 1440)}
    assert is_open_at(both, "Monday", 23 * 60 + 120)  # Monday 23:00 + 2 h = Tuesday 01:00
    assert not is_open_at(monday_only, "Monday", 23 * 60 + 120)
    assert is_open_at({"Sunday": (600, 1200)}, "Saturday", 24 * 60 + 700)  # rolls Saturday -> Sunday
