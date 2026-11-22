"""Contrast-adaptive sharpening (CAS) primitives.

Conventions used by every function in this module
-------------------------------------------------
* An image is a non-empty rectangular ``list[list[float]]``.  ``img[y][x]`` is
  the pixel in row ``y`` and column ``x``; values are display-referred sRGB
  numbers in ``0.0 .. 1.0`` unless a contract explicitly says "linear light".
* Linear-light values are also in ``0.0 .. 1.0``.  The conversion used by the
  pipeline is ``linear = srgb ** 2.2`` and ``srgb = linear ** (1 / 2.2)``.
* Sharpening weights are in ``0.0 .. 1.0``; ``amount`` is in ``0.0 .. 2.0``.
* All numeric contracts are checked with an absolute tolerance of ``1e-9``.
"""

import math

_LOW_KNEE = 0.10
_HIGH_KNEE = 0.50


def _kernel(radius):
    """Return a normalized 1-D Gaussian kernel for ``sigma = radius``.

    The half-width is ``max(1, ceil(3 * radius))`` so the support grows with
    the radius instead of being pinned to a 3x3 neighbourhood.
    """
    half = max(1, int(math.ceil(3.0 * radius)))
    sigma = float(radius)
    two_sigma_sq = 2.0 * sigma * sigma
    weights = [math.exp(-(i * i) / two_sigma_sq) for i in range(-half, half + 1)]
    total = sum(weights)
    return [w / total for w in weights]


def _blur(img, radius):
    """Separable Gaussian blur with border-replicate sampling."""
    height = len(img)
    width = len(img[0])
    kernel = _kernel(radius)
    half = len(kernel) // 2

    # Horizontal pass: rows are contiguous, so this is the cheap direction.
    horizontal = [[0.0] * width for _ in range(height)]
    for y in range(height):
        row = img[y]
        out_row = horizontal[y]
        for x in range(width):
            acc = 0.0
            for k, weight in enumerate(kernel):
                sx = x + k - half
                if sx < 0:
                    sx = 0
                elif sx >= width:
                    sx = width - 1
                acc += weight * row[sx]
            out_row[x] = acc

    # Vertical pass.
    blurred = [[0.0] * width for _ in range(height)]
    for y in range(height):
        out_row = blurred[y]
        for x in range(width):
            acc = 0.0
            for k, weight in enumerate(kernel):
                sy = y + k - half
                if sy < 0:
                    sy = 0
                elif sy >= height:
                    sy = height - 1
                acc += weight * horizontal[sy][x]
            out_row[x] = acc
    return blurred


def local_min_max(img, x, y):
    """Return the extrema of the clipped 3x3 neighbourhood around ``(x, y)``.

    The neighbourhood contains the centre cell and all eight neighbours
    (including the four diagonal cells).  At image borders it is clipped to the
    part that lies inside the image, so a corner uses a 2x2 block.

    Args:
        img: rectangular image, values in ``0.0 .. 1.0``.
        x: column index, ``0 <= x < len(img[0])``.
        y: row index, ``0 <= y < len(img)``.

    Returns:
        ``(lo, hi)`` as a tuple of floats with ``lo <= hi``.
    """
    height = len(img)
    width = len(img[0])
    y0 = max(0, y - 1)
    y1 = min(height - 1, y + 1)
    x0 = max(0, x - 1)
    x1 = min(width - 1, x + 1)

    lo = img[y0][x0]
    hi = img[y0][x0]
    for ny in range(y0, y1 + 1):
        row = img[ny]
        for nx in range(x0, x1 + 1):
            value = row[nx]
            if value < lo:
                lo = value
            if value > hi:
                hi = value
    return (lo, hi)


def contrast_adaptive_weight(local_contrast):
    """Map a local contrast value (``hi - lo``) to a sharpening weight.

    The mapping is a smoothstep with knees at ``0.10`` and ``0.50``::

        c <= 0.10                  -> 0.0
        c >= 0.50                  -> 1.0
        0.10 < c < 0.50            -> t = (c - 0.10) / 0.40
                                      w = t * t * (3.0 - 2.0 * t)

    The result is monotonically non-decreasing in ``c`` and lies in
    ``0.0 .. 1.0``; low-contrast regions receive (almost) no sharpening.

    Args:
        local_contrast: ``hi - lo`` of a neighbourhood, normally ``0.0 .. 1.0``.

    Returns:
        The adaptive weight in ``0.0 .. 1.0``.
    """
    c = float(local_contrast)
    if c <= _LOW_KNEE:
        return 0.0
    if c >= _HIGH_KNEE:
        return 1.0
    t = (c - _LOW_KNEE) / (_HIGH_KNEE - _LOW_KNEE)
    return t * t * (3.0 - 2.0 * t)


def unsharp(img, radius, amount):
    """Apply a high-boost (unsharp mask) filter directly in the value domain.

    A normalized separable Gaussian blur is built with ``sigma = radius`` and
    kernel half-width ``max(1, ceil(3 * radius))``; samples outside the image
    are replaced by the nearest border pixel.  Then, for every pixel::

        out = clamp(img + amount * (img - blur), 0.0, 1.0)

    No gamma conversion happens here: the blur and the boost use the same
    display-referred values that were passed in.  ``radius`` must be positive
    and ``amount`` non-negative, otherwise ``ValueError`` is raised.

    Args:
        img: rectangular image, values in ``0.0 .. 1.0``.
        radius: Gaussian sigma in pixels, ``> 0``.
        amount: boost strength, ``>= 0``.

    Returns:
        A new image with values clamped to ``0.0 .. 1.0``.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if amount < 0:
        raise ValueError("amount must be non-negative")

    blur = _blur(img, float(radius))
    height = len(img)
    width = len(img[0])
    out = []
    for y in range(height):
        src_row = img[y]
        blur_row = blur[y]
        out_row = []
        for x in range(width):
            value = src_row[x] + amount * (src_row[x] - blur_row[x])
            if value < 0.0:
                value = 0.0
            elif value > 1.0:
                value = 1.0
            out_row.append(value)
        out.append(out_row)
    return out


def clamp_overshoot(value, lo, hi):
    """Clamp ``value`` to the closed interval ``[lo, hi]``.

    Both ends are enforced: values below ``lo`` become ``lo`` and values above
    ``hi`` become ``hi``.  If ``lo > hi`` a ``ValueError`` is raised.
    """
    if lo > hi:
        raise ValueError("lo must not exceed hi")
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def sharpen(img, amount, radius, threshold):
    """Run the complete contrast-adaptive sharpening pipeline in linear light.

    Contract:

    1. Call :func:`validate_params`; if it returns any message, raise
       ``ValueError("; ".join(messages))``.
    2. Decode the sRGB input to linear light with ``v ** 2.2``.
    3. Blur the *linear* image with the Gaussian from :func:`unsharp`
       (``sigma = radius``, half-width ``max(1, ceil(3 * radius))``, border
       replicate).
    4. For each pixel compute ``lo, hi = local_min_max(img, x, y)`` on the
       original sRGB image and ``c = hi - lo``.  If ``c < threshold`` the
       pixel is copied through unchanged (low-contrast pixels are not
       sharpened; the comparison uses local contrast, not signed detail).
    5. Otherwise ``w = contrast_adaptive_weight(c)``,
       ``candidate = lin + amount * w * (lin - blur)``.
    6. Clamp ``candidate`` into ``[lo ** 2.2, hi ** 2.2]`` so no white or
       black halo can leave the local neighbourhood range, then encode with
       ``candidate ** (1 / 2.2)``.

    ``amount`` belongs to ``0.0 .. 2.0``; ``radius`` to ``0.5 .. 8.0``;
    ``threshold`` to ``0.0 .. 1.0``.  ``amount == 0`` must reproduce the
    input exactly (within ``1e-9``).  The input image is never modified.

    Args:
        img: rectangular sRGB image, values in ``0.0 .. 1.0``.
        amount: sharpening amount, ``0.0 .. 2.0``.
        radius: Gaussian sigma in pixels, ``0.5 .. 8.0``.
        threshold: minimum local contrast required to sharpen, ``0.0 .. 1.0``.

    Returns:
        A new sRGB image with values in ``0.0 .. 1.0``.
    """
    messages = validate_params(amount, radius, threshold)
    if messages:
        raise ValueError("; ".join(messages))

    height = len(img)
    width = len(img[0])
    gamma = 2.2
    inv_gamma = 1.0 / gamma

    linear = [[value ** gamma for value in row] for row in img]
    blur = _blur(linear, float(radius))

    out = []
    for y in range(height):
        src_row = img[y]
        lin_row = linear[y]
        blur_row = blur[y]
        out_row = []
        for x in range(width):
            lo, hi = local_min_max(img, x, y)
            contrast = hi - lo
            if contrast < threshold:
                out_row.append(src_row[x])
                continue
            weight = contrast_adaptive_weight(contrast)
            candidate = lin_row[x] + amount * weight * (lin_row[x] - blur_row[x])
            candidate = clamp_overshoot(candidate, lo ** gamma, hi ** gamma)
            out_row.append(candidate ** inv_gamma)
        out.append(out_row)
    return out


def noise_gain(amount, threshold):
    """Return the reported noise amplification factor of the sharpening stage.

    ``threshold`` is clipped to ``0.0 .. 1.0`` and the factor is::

        (1.0 + amount) ** 2 * (1.0 - threshold)

    so zero threshold and ``amount == 1`` report a factor of ``4.0`` (noise
    power grows with the boosted amplitude, not with ``amount ** 2``).

    Args:
        amount: sharpening amount, ``0.0 .. 2.0``.
        threshold: local-contrast threshold, ``0.0 .. 1.0``.

    Returns:
        A non-negative float.
    """
    t = float(threshold)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return (1.0 + amount) ** 2 * (1.0 - t)


def validate_params(amount, radius, threshold):
    """Collect range errors for the three sharpening parameters.

    Return an empty list when everything is valid.  Errors are appended in the
    fixed order ``amount``, ``radius``, ``threshold`` and use exactly these
    messages::

        "amount must be in [0.0, 2.0]"       # 0.0 <= amount <= 2.0
        "radius must be in [0.5, 8.0]"       # 0.5 <= radius <= 8.0
        "threshold must be in [0.0, 1.0]"    # 0.0 <= threshold <= 1.0

    The upper bound of ``amount`` is part of the check so over-sharpening
    parameters are rejected instead of silently accepted.

    Args:
        amount: sharpening amount.
        radius: Gaussian sigma in pixels.
        threshold: local-contrast threshold.

    Returns:
        ``list[str]`` of messages, empty when all parameters are valid.
    """
    messages = []
    if not (0.0 <= amount <= 2.0):
        messages.append("amount must be in [0.0, 2.0]")
    if not (0.5 <= radius <= 8.0):
        messages.append("radius must be in [0.5, 8.0]")
    if not (0.0 <= threshold <= 1.0):
        messages.append("threshold must be in [0.0, 1.0]")
    return messages


def sharpen_batch(frames, config):
    """Sharpen a batch of frames, each one independently.

    ``config`` must contain ``"amount"``, ``"radius"`` and ``"threshold"``.
    The parameters are validated once with :func:`validate_params`; an invalid
    configuration raises ``ValueError("; ".join(messages))``.

    Every frame is processed from its own pixels only: no blur buffer or
    intermediate result may be carried over from one frame to the next.  The
    result is::

        {
            "frames": [sharpen(frame, amount, radius, threshold), ...],
            "count": len(frames),
            "noise_gain": noise_gain(amount, threshold),
        }

    Args:
        frames: list of rectangular sRGB images.
        config: dict with the three numeric sharpening parameters.

    Returns:
        ``dict`` with the keys ``"frames"``, ``"count"`` and ``"noise_gain"``.
    """
    amount = config["amount"]
    radius = config["radius"]
    threshold = config["threshold"]

    messages = validate_params(amount, radius, threshold)
    if messages:
        raise ValueError("; ".join(messages))

    return {
        "frames": [sharpen(frame, amount, radius, threshold) for frame in frames],
        "count": len(frames),
        "noise_gain": noise_gain(amount, threshold),
    }
