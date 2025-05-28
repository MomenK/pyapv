# APV Frame Parser

This project provides a Python implementation for parsing APV (Advanced Pixel Video) bitstreams. It supports inspection of all bitstream elements from Access Units (AUs) down to individual macroblocks (MBs).

---

## 🔹 Bitstream Hierarchy

Below is the structural hierarchy of an APV bitstream:

```
Access Unit (AU)
├── Signature (4 bytes)
├── Repeated Packetized Bitstream Units (PBUs)
    ├── PBU Header
    │   ├── pbu_type
    │   ├── group_id
    │   └── reserved_zero_8bits
    └── PBU Payload
        ├── frame() [if pbu_type in {1,2,25–27}]
        │   ├── frame_header()
        │   │   ├── frame_info()
        │   │   ├── color_description (optional)
        │   │   ├── quantization_matrix (optional)
        │   │   └── tile_info()
        │   └── Repeated Tiles
        │       ├── tile_header()
        │       └── tile_data()
        │           ├── For each component
        │           │   └── Repeated Macroblocks
        │           │       └── macroblock_layer()
        ├── au_info()        [if pbu_type == 65]
        ├── metadata()        [if pbu_type == 66]
        └── filler()          [if pbu_type == 67]
```

---

## 📊 Key Components

* **Access Unit (AU):** Top-level container for PBUs.
* **PBU (Packetized Bitstream Unit):** Contains logical bitstream data like frames, metadata, etc.
* **frame():** Represents an image frame, includes headers and tile layout.
* **tile():** Frame subdivision, used for decoding and parallelism.
* **macroblock\_layer():** Basic unit of encoded pixel data (typically 16x16).

---

## 🚀 Usage

To parse an `.apv` file, simply run:

```bash
python3 parser.py ../openapv/test/bitstream/tile_A.apv 
```

The output will describe the structure and contents of each element in the bitstream.

---

## ✅ Output Example

```
Access Unit 0:
  Signature: aPv1
  Total size: 3109535 bytes
    PBU 0:
      Size     : 3109449 bytes
      Type     : 1 (0x01)
      Group ID : 1
      Reserved : 0 (should be 0)
      Payload Preview (first 16 bytes after header): 21 7b 40 00 0f 00 00 08 70 22 00 00 00 00 03 c0...
      -> Contains frame() data
        Parsing frame_header()
          frame_info():
            profile_idc: 33
            level_idc: 123
            band_idc: 2
            frame_width: 3840
            frame_height: 2160
            chroma_format_idc: 2
            bit_depth: 10
            capture_time_distance: 0
          reserved_zero_8bits: 0
          color_description_present_flag: 0
          use_q_matrix: 0
          tile_info():
            tile_width_in_mbs: 240
            tile_height_in_mbs: 135
            TileCols: 1, TileRows: 1, NumTiles: 1
            tile_size_present_in_fh_flag: 0
          reserved_zero_8bits: 0
          byte_alignment(): aligned to next byte boundary
            Parsing tile_header() for tile 0:
              tile_size: 3109421
              tile_header_size: 20
              tile_index: 0
              tile_data_sizes: [2513604, 280304, 315493]
              tile_qp: [30, 30, 30]
              reserved_zero_8bits: 0
              tile_data for component 0:
                macroblock_layer at xMb=0, yMb=0, cIdx=0
                  Preview: 71 52 67 2a a8 b7 c6 7a af 89 ff 24 de ca 17 12...
              tile_data for component 1:
                macroblock_layer at xMb=0, yMb=0, cIdx=1
                  Preview: a3 45 f1 6f a3 2e 82 0c 23 83 ed 9b 9a 59 0b d1...
              tile_data for component 2:
                macroblock_layer at xMb=0, yMb=0, cIdx=2
                  Preview: 89 4a 54 ff a0 d8 25 1d 19 07 2b 7d 07 b2 69 0a...
```

---

## 💡 Notes

* This parser does not decode image data; it only parses bitstream metadata.
* Support for full macroblock decoding can be added later if needed.

---
