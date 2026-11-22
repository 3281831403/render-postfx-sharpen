"""Contract tests for :mod:`sharpen.cas`.

The image buffers below are plain ``list[list[float]]`` grids whose values are
display-referred (0.0 .. 1.0) unless a function contract says otherwise.  All
numeric comparisons use an absolute tolerance of 1e-9.
"""

import copy

import pytest

from sharpen.cas import (
    clamp_overshoot,
    contrast_adaptive_weight,
    local_min_max,
    noise_gain,
    sharpen,
    sharpen_batch,
    unsharp,
    validate_params,
)

TOL = 1e-9

# A soft bright edge: flat dark plateau, convex ramp, inflection, concave ramp,
# flat bright plateau.  Column 6 is still on the dark side of the edge and has
# negative high-boost detail, column 7 is the steep part.
EDGE_ROW = [0.05, 0.05, 0.05, 0.05, 0.08, 0.15, 0.32, 0.55, 0.75, 0.88, 0.94, 0.94]


def _edge_frame(height=9):
    return [list(EDGE_ROW) for _ in range(height)]


def _assert_close(actual, expected, tol=TOL):
    assert isinstance(actual, list)
    assert len(actual) == len(expected)
    for row_a, row_e in zip(actual, expected):
        assert isinstance(row_a, list)
        assert len(row_a) == len(row_e)
        for a, e in zip(row_a, row_e):
            assert a == pytest.approx(e, abs=tol)


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _noise_and_edge_frame():
    """Left third: fine low-contrast checkerboard, right third: soft edge."""
    width, height = 40, 10
    frame = []
    for y in range(height):
        row = []
        for x in range(width):
            if x < 16:
                row.append(0.46 if (x + y) % 2 == 0 else 0.54)
            elif x < 24:
                row.append(0.5)
            else:
                row.append(0.10 + 0.82 * _smoothstep((x - 24) / 15.0))
        frame.append(row)
    return frame


def test_local_min_max_includes_diagonal_neighbours():
    img = [[0.5] * 5 for _ in range(5)]
    img[1][1] = 0.05
    img[3][3] = 0.95

    result = local_min_max(img, 2, 2)
    assert isinstance(result, tuple)
    lo, hi = result
    assert lo == pytest.approx(0.05, abs=TOL)
    assert hi == pytest.approx(0.95, abs=TOL)

    border = [[0.10, 0.20], [0.30, 0.40]]
    assert local_min_max(border, 0, 0) == pytest.approx((0.10, 0.40), abs=TOL)


def test_contrast_adaptive_weight_is_smooth_not_linear():
    assert contrast_adaptive_weight(0.0) == pytest.approx(0.0, abs=TOL)
    assert contrast_adaptive_weight(0.05) <= 0.001
    assert contrast_adaptive_weight(0.20) == pytest.approx(0.15625, abs=TOL)
    assert contrast_adaptive_weight(0.30) == pytest.approx(0.5, abs=TOL)
    assert contrast_adaptive_weight(0.50) == pytest.approx(1.0, abs=TOL)
    assert contrast_adaptive_weight(1.0) == pytest.approx(1.0, abs=TOL)

    samples = [contrast_adaptive_weight(i / 50.0) for i in range(51)]
    assert all(a <= b + 1e-12 for a, b in zip(samples, samples[1:]))


def test_unsharp_blur_scales_with_radius():
    step = [[0.2 if x < 12 else 0.6 for x in range(24)] for _ in range(24)]
    wide = unsharp(step, 4.0, 1.0)
    narrow = unsharp(step, 0.5, 1.0)

    # Three pixels away from the step only a wide blur has any influence.
    assert abs(wide[12][15] - 0.6) > 0.02
    assert abs(narrow[12][15] - 0.6) <= TOL
    assert abs(wide[12][15] - narrow[12][15]) > 0.02


def test_unsharp_uses_high_boost_sign():
    step = [[0.2 if x < 12 else 0.6 for x in range(24)] for _ in range(24)]
    out = unsharp(step, 1.0, 1.0)

    assert out[12][11] < 0.2 - 0.01
    assert out[12][12] > 0.6 + 0.01


def test_clamp_overshoot_limits_both_directions():
    assert clamp_overshoot(0.20, 0.10, 0.50) == pytest.approx(0.20, abs=TOL)
    assert clamp_overshoot(-0.40, 0.10, 0.50) == pytest.approx(0.10, abs=TOL)
    assert clamp_overshoot(0.90, 0.10, 0.50) == pytest.approx(0.50, abs=TOL)
    assert clamp_overshoot(0.10, 0.10, 0.50) == pytest.approx(0.10, abs=TOL)
    assert clamp_overshoot(0.50, 0.10, 0.50) == pytest.approx(0.50, abs=TOL)

    with pytest.raises(ValueError):
        clamp_overshoot(0.30, 0.50, 0.10)


def test_sharpen_keeps_negative_edges():
    img = _edge_frame()
    out = sharpen(img, 1.0, 1.0, 0.05)

    # Column 6 sits on the dark side; its high-boost detail is negative.
    assert out[4][6] < img[4][6] - 0.03
    assert out[4][6] < out[4][7]


def test_noise_gain_uses_boosted_amplitude():
    assert noise_gain(0.0, 0.0) == pytest.approx(1.0, abs=TOL)
    assert noise_gain(0.5, 0.0) == pytest.approx(2.25, abs=TOL)
    assert noise_gain(1.0, 0.0) == pytest.approx(4.0, abs=TOL)
    assert noise_gain(1.0, 0.5) == pytest.approx(2.0, abs=TOL)
    assert noise_gain(2.0, 0.25) == pytest.approx(6.75, abs=TOL)


def test_sharpen_processes_linear_light_not_srgb():
    # A linear sRGB ramp is exactly flat for an sRGB-space blur, so an
    # sRGB-space high-boost would leave it untouched.  Decoded to linear light
    # the same ramp is convex and must be changed.
    ramp = [[0.05 + 0.075 * x for x in range(13)] for _ in range(13)]
    out = sharpen(ramp, 2.0, 2.0, 0.05)

    assert abs(out[6][6] - ramp[6][6]) > 5e-4


def test_validate_params_rejects_amount_above_upper_bound():
    assert validate_params(1.0, 2.0, 0.1) == []
    assert validate_params(2.01, 2.0, 0.1) == ["amount must be in [0.0, 2.0]"]
    assert validate_params(-0.01, 2.0, 0.1) == ["amount must be in [0.0, 2.0]"]
    assert validate_params(1.0, 0.49, 0.1) == ["radius must be in [0.5, 8.0]"]
    assert validate_params(1.0, 2.0, 1.01) == ["threshold must be in [0.0, 1.0]"]
    assert validate_params(3.0, 0.1, 2.0) == [
        "amount must be in [0.0, 2.0]",
        "radius must be in [0.5, 8.0]",
        "threshold must be in [0.0, 1.0]",
    ]


def test_batch_does_not_reuse_previous_frame_buffer():
    first = _edge_frame()
    second = [[0.85 - 0.05 * x + 0.005 * y for x in range(12)] for y in range(9)]
    config = {"amount": 1.0, "radius": 1.5, "threshold": 0.05}

    result = sharpen_batch([first, second], config)
    assert result["count"] == 2

    expected_second = sharpen(second, 1.0, 1.5, 0.05)
    _assert_close(result["frames"][1], expected_second)
    assert result["frames"][0] != result["frames"][1]


def test_zero_amount_is_identity():
    img = _edge_frame()
    out = sharpen(img, 0.0, 1.5, 0.10)

    _assert_close(out, img)


def test_strong_edge_gradient_increases():
    img = _edge_frame()
    out = sharpen(img, 1.0, 1.0, 0.05)

    before = img[4][7] - img[4][6]
    after = out[4][7] - out[4][6]
    assert after > before + 0.01


def test_sharpen_stays_within_local_range():
    img = _edge_frame()
    out = sharpen(img, 2.0, 1.0, 0.0)

    for y in range(len(img)):
        for x in range(len(img[0])):
            lo, hi = local_min_max(img, x, y)
            assert lo - TOL <= out[y][x] <= hi + TOL


def test_noise_increment_below_edge_increment():
    img = _noise_and_edge_frame()
    out = sharpen(img, 1.0, 1.0, 0.10)

    noise_increment = sum(
        abs(out[y][x] - img[y][x])
        for y in range(1, len(img) - 1)
        for x in range(1, 15)
    )
    edge_increment = sum(
        abs(out[y][x] - img[y][x])
        for y in range(1, len(img) - 1)
        for x in range(26, 38)
    )

    assert noise_increment <= 1e-6
    assert noise_increment < edge_increment


def test_batch_matches_separate_sharpening():
    first = _edge_frame()
    second = [[0.2 + 0.01 * (x + y) for x in range(12)] for y in range(9)]
    first_before = copy.deepcopy(first)
    second_before = copy.deepcopy(second)
    config = {"amount": 1.0, "radius": 1.0, "threshold": 0.05}

    result = sharpen_batch([first, second], config)
    separate_first = sharpen(first, 1.0, 1.0, 0.05)
    separate_second = sharpen(second, 1.0, 1.0, 0.05)

    assert result["count"] == 2
    assert result["noise_gain"] == pytest.approx(noise_gain(1.0, 0.05), abs=TOL)
    _assert_close(result["frames"][0], separate_first)
    _assert_close(result["frames"][1], separate_second)
    assert result["frames"][0] is not first
    assert result["frames"][1] is not second
    assert first == first_before
    assert second == second_before
