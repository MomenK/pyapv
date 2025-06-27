import numpy as np
import matplotlib.pyplot as plt


def restore_mb_tu_image(flat: np.ndarray, height: int, width: int) -> np.ndarray:
    """Recreate a 2-D image from a 1-D stream ordered TU-by-TU inside 16×16 MBs."""
    MB, TU = 16, 8
    PIXELS_PER_TU = TU * TU
    PIXELS_PER_MB = MB * MB

    if flat.size != height * width:
        raise ValueError("Flat array length does not match height × width.")
    if height % MB or width % MB:
        raise ValueError("Image dimensions must be multiples of 16.")

    mbs_per_row = width  // MB
    mbs_per_col = height // MB
    img = np.empty((height, width), dtype=flat.dtype)

    for mb_idx in range(mbs_per_row * mbs_per_col):
        mb_row, mb_col = divmod(mb_idx, mbs_per_row)
        mb_row_px = mb_row * MB
        mb_col_px = mb_col * MB
        mb_offset = mb_idx * PIXELS_PER_MB

        for tu_idx in range(4):                      # TL, TR, BL, BR
            tu_row_off = (tu_idx // 2) * TU
            tu_col_off = (tu_idx %  2) * TU
            tu_offset  = mb_offset + tu_idx * PIXELS_PER_TU
            tu_block   = flat[tu_offset : tu_offset + PIXELS_PER_TU].reshape(TU, TU)

            img[
                mb_row_px + tu_row_off : mb_row_px + tu_row_off + TU,
                mb_col_px + tu_col_off : mb_col_px + tu_col_off + TU
            ] = tu_block

    return img


# ----------------------------------------------------------------------
# TEST PATTERN
# ----------------------------------------------------------------------
H, W = 320, 320                       # any multiples of 16 work
flat_pixels = np.arange(H * W)        # simple ramp: 0, 1, 2, …

image = restore_mb_tu_image(flat_pixels, height=H, width=W)

# Quick numeric sanity-check: look at the top-left 10×10 region
print("Top-left corner:")
print(image[:16, :16])

# Visual inspection
plt.figure(figsize=(4, 4))
plt.imshow(image[:16, :16], cmap='viridis')
plt.title(f"Reconstructed {H}×{W} image")
# plt.axis("off")
# plt.show()

plt.figure(figsize=(4, 4))
plt.imshow(image, cmap='viridis')
plt.title(f"Reconstructed {H}×{W} image")
plt.axis("off")
plt.show()