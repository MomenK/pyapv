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
python apv_parser.py path/to/bitstream.apv
```

The output will describe the structure and contents of each element in the bitstream.

---

## ✅ Output Example

```
Access Unit 0:
  Signature: APV1
  Total size: 10240 bytes
    PBU 0:
      Size     : 9000 bytes
      Type     : 1 (0x01)
      Group ID : 10
      Payload Preview: ab cd ef 01 ...
      -> Contains frame() data
        Parsing frame_header()
          frame_info():
            frame_width: 1920
            frame_height: 1080
            chroma_format_idc: 3
          tile_info():
            TileCols: 6, TileRows: 4, NumTiles: 24
            tile_size_present_in_fh_flag: 1
              tile_size_in_fh: [...]
            Parsing tile_header() for tile 0:
              macroblock_layer at xMb=0, yMb=0, cIdx=0
```

---

## 💡 Notes

* This parser does not decode image data; it only parses bitstream metadata.
* Support for full macroblock decoding can be added later if needed.

---
