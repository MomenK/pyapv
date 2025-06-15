import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

class FrameViewer:
    def __init__(self, filename, SubWidthC, SubHeight, BitDepth, rec_samples):
        self.rec_samples = rec_samples
        self.filename = filename
        self.sub_w = SubWidthC
        self.sub_h = SubHeight
        self.BitDepth = BitDepth
        self.make_cb_cr_cmaps()

    def save_frame_as_image(self):
        """
        Upsample chroma, map bit-depth → 8-bit, convert YCbCr→RGB and save.

        Parameters
        ----------
        sub_w, sub_h : int
            Horizontal / vertical chroma subsampling factors (1 or 2).
        """
        if len(self.rec_samples) < 3:               # need at least Y,Cb,Cr
            return

        y, cb, cr = self.rec_samples[:3]

        # --- 1)  rebuild full-size chroma planes -----------------
        if self.sub_h == 2:
            cb = np.repeat(cb, 2, axis=0)
            cr = np.repeat(cr, 2, axis=0)
        if self.sub_w == 2:
            cb = np.repeat(cb, 2, axis=1)
            cr = np.repeat(cr, 2, axis=1)

        # --- 2)  bit-depth → 8-bit + add offsets ----------------
        y8, cb8, cr8 = self._prepare_planes_for_display(
            y, cb, cr,
            bit_depth = self.BitDepth,
            full_range = True
        )

        # --- 3)  YCbCr → RGB  (BT.601 full-swing) ---------------
        print(y.shape[0], y.shape[1])
        rgb = self.ycbcr_to_rgb(y8, cb8, cr8).reshape(y.shape[0],y.shape[1],3) 
        print(rgb.shape)

        # --- 4)  plot and save ---------------
        fig, axes = plt.subplots(2, 2, figsize=(12, 4))   # 1 row × 3 cols
        mid_val = (1 << (self.BitDepth - 1))

        planes = [
            (rgb / 256, "Reconstructed RGB", None),
            (y8  / 256, "Reconstructed Luma (Y)", 'gray'),
            (cb8 / 256, "Reconstructed Chroma (Cb)", 'cb_map'),
            (cr8 / 256, "Reconstructed Chroma (Cr)", 'cr_map'),
        ]

        for ax, (img, title, color) in zip(axes.flat, planes):
            ax.imshow(img, cmap=color,vmin=0,vmax=1)
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.filename+"_decomposition.png")

        Image.fromarray(rgb, 'RGB').save(self.filename+".png")
        print(f"saved reconstructed frame → {self.filename}")

        yuv_name = self.filename+".yuv"
        out_dtype = np.dtype('<u2')
        with open(yuv_name, "wb") as fp:
            # # Y plane (original resolution, original bit-depth → little-endian)
            fp.write(y.astype(out_dtype if self.BitDepth > 8 else np.uint8).tobytes())
            # # Re-sub-sample Cb/Cr back to native resolution before dumping
            cb_native = cb[::self.sub_h, ::self.sub_w]
            cr_native = cr[::self.sub_h, ::self.sub_w]
            fp.write(cb_native.astype(out_dtype if self.BitDepth > 8 else np.uint8).tobytes())
            fp.write(cr_native.astype(out_dtype if self.BitDepth > 8 else np.uint8).tobytes())

        print(f"Raw YUV ({self.BitDepth}-bit planar) saved → {yuv_name}")
    # ------------------------------------------------------------
    def _prepare_planes_for_display(self,
                                    y, cb, cr,
                                    *,
                                    bit_depth: int,
                                    full_range: bool = False):
        """
        • converts any bit-depth (8/10/12/…) luma & chroma planes to 8-bit
        • adds / removes video-range offsets when *full_range == False*

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]  ––  uint8 Y, Cb, Cr
        """
        # scale factor between source bit-depth and 8-bit
        shift = bit_depth - 8
        scale = 1 << shift                   # e.g. 10-bit → 4, 12-bit → 16

        # --- luma ------------------------------------------------
        if full_range:
            y8 = (y / scale)
        else:
            # video-range (16-235); undo offset & rescale
            y_offset = 16  * scale
            y_range  = 219 * scale
            y8 = (y - y_offset) * 255.0 / y_range

        # --- chroma ----------------------------------------------
        if full_range:
            cb8 = (cb / scale)
            cr8 = (cr / scale)
        else:
            # video-range (16-240); undo offset & rescale
            c_offset = 128 * scale
            c_range  = 224 * scale
            cb8 = (cb - c_offset) * 255.0 / c_range + 128
            cr8 = (cr - c_offset) * 255.0 / c_range + 128

        # clip & uint8
        return (np.clip(y8,  0, 255).astype(np.uint8),
                np.clip(cb8, 0, 255).astype(np.uint8),
                np.clip(cr8, 0, 255).astype(np.uint8))
    # ------------------------------------------------------------
    def ycbcr_to_rgb(self, y, cb, cr):
        """
        BT.601 full-range YCbCr→RGB conversion.
        *ycbcr_8bit* must already be uint8 / 0-255.
        """
        y  = y.astype(np.float32)
        cb = cb.astype(np.float32) - 128.0
        cr = cr.astype(np.float32) - 128.0

        r = np.clip(y + 1.40200 * cr, 0, 255).astype(np.uint8)
        g = np.clip(y - 0.34414 * cb - 0.71414 * cr, 0, 255).astype(np.uint8)
        b = np.clip(y + 1.77200 * cb, 0, 255).astype(np.uint8)

        return np.stack((r,g,b),axis=-1)
    
    def make_cb_cr_cmaps(self, N=256):

        print(f"Checking for cmap...")
        if "cb_map" in list(mpl.colormaps) and "cr_map" in list(mpl.colormaps):
            return True
        
        full_scale = 255
        mid_val = 128
        N = 256
        y   = np.zeros(N) + mid_val
        
        cb = np.linspace(-mid_val, mid_val, N) + mid_val
        cr = np.zeros(N) + mid_val
        rgb = self.ycbcr_to_rgb(y,cb,cr) / full_scale
        Cb_cmap = mpl.colors.ListedColormap(rgb, name="cb_map")
        

        cb  = np.zeros(N) + mid_val
        cr  = np.linspace(-mid_val, mid_val, N) + mid_val
        rgb = self.ycbcr_to_rgb(y,cb,cr) / full_scale
        Cr_cmap = mpl.colors.ListedColormap(rgb, name="cr_map")

        # if mpl.colormaps.get_cmap(Cb_cmap):
        #     pass
        # else:
        mpl.colormaps.register(Cb_cmap)

        # if mpl.colormaps.get_cmap(Cr_cmap):
        #     pass
        # else:
        mpl.colormaps.register(Cr_cmap)

