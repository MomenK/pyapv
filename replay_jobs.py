#!/usr/bin/env python3
"""
replay_jobs.py – replay a saved tile-job queue and visualise one tile
--------------------------------------------------------------------
Run:
    python replay_jobs.py jobs1.pkl
"""

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

from apv_parser import APVDecoder


def _infer_dims(job_queue):
    """Return dims from the keys inside *job_queue*."""
    comp_ids  = set()
    subw = set()
    subh = set()
    for cfg, _ in job_queue.values():
        comp_ids.add(cfg["cIdx"])
        subw.add(cfg["subW"])
        subh.add(cfg["subH"])
    return len(comp_ids), max(subw), max(subh)


# ------------------------------------------------------------------
#  main
# ------------------------------------------------------------------
def main(pkl_path: str) -> None:
    # 1. load the pickled job queue --------------------------------
    with open(pkl_path, "rb") as fp:
        job_queue: dict[int, tuple] = pickle.load(fp)
    print(f"[loaded] {len(job_queue)} jobs from {pkl_path}")

    # 2. create a *blank* decoder just to reuse its helpers ---------
    dec = APVDecoder.__new__(APVDecoder)      # bypass __init__
    dec.job_queue = job_queue                # attach the jobs

    # frame_buffer will be filled by decode_tiles()
    num_comps , SubWidthC, SubHeightC = _infer_dims(job_queue)
    dec.setup_frame_buffer(3840, 2160, num_comps , SubWidthC, SubHeightC)

    # 3. decode tiles --------------------------
    dec.decode_tiles(job_queue)

    # 4. visualise the very first reconstructed luma tile ----------
    if dec.frame_buffer is None:
        print("No frame buffer was produced!")
        return

    y_plane = dec.frame_buffer[0]            # luma component
    plt.imshow(y_plane, cmap="gray")
    plt.title("First reconstructed tile – Y component")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python replay_jobs.py <jobs.pkl>")
        sys.exit(1)
    main(sys.argv[1])
