import struct
import argparse

class BitstreamReader:
    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0

    def read_bit(self):
        if self.byte_pos >= len(self.data):
            raise EOFError("End of stream")
        byte = self.data[self.byte_pos]
        bit = (byte >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return bit

    def read_bits(self, n):
        val = 0
        for _ in range(n):
            val = (val << 1) | self.read_bit()
        return val

    def read_bytes(self, n):
        if self.bit_pos != 0:
            raise RuntimeError("Not byte-aligned")
        result = self.data[self.byte_pos:self.byte_pos+n]
        self.byte_pos += n
        return result

    def more_data(self):
        return self.byte_pos < len(self.data)

    def remaining_bytes(self):
        return len(self.data) - self.byte_pos

    def byte_align(self):
        if self.bit_pos != 0:
            remaining = 8 - self.bit_pos
            _ = self.read_bits(remaining)  # Skip to next byte

class APVDecoder:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            self.data = f.read()
        self.reader = BitstreamReader(self.data)
        self.frames = []

    def get_num_components(self, chroma_format_idc):
        if chroma_format_idc == 0:
            return 1
        elif chroma_format_idc == 2 or chroma_format_idc == 3:
            return 3
        elif chroma_format_idc == 4:
            return 4
        else:
            raise ValueError(f"Unsupported or reserved chroma_format_idc: {chroma_format_idc}")

    def parse_access_units(self):
        r = self.reader
        index = 0
        while r.more_data():
            try:
                au_size = int.from_bytes(r.read_bytes(4), 'big')
                au_payload = r.read_bytes(au_size)
                sub_reader = BitstreamReader(au_payload)

                signature = sub_reader.read_bytes(4).decode('ascii', errors='replace')
                print(f"\nAccess Unit {index}:")
                print(f"  Signature: {signature}")
                print(f"  Total size: {au_size} bytes")

                curr_read_size = 4
                pbu_index = 0

                while curr_read_size < au_size:
                    if sub_reader.remaining_bytes() < 4:
                        raise EOFError("Unexpected end of AU while reading pbu_size")

                    pbu_size = int.from_bytes(sub_reader.read_bytes(4), 'big')
                    curr_read_size += 4

                    if sub_reader.remaining_bytes() < pbu_size:
                        raise EOFError("Unexpected end of AU while reading PBU")

                    pbu_data = sub_reader.read_bytes(pbu_size)
                    curr_read_size += pbu_size

                    pbu_reader = BitstreamReader(pbu_data)
                    pbu_type = int.from_bytes(pbu_reader.read_bytes(1), 'big')
                    group_id = int.from_bytes(pbu_reader.read_bytes(2), 'big')
                    reserved = int.from_bytes(pbu_reader.read_bytes(1), 'big')

                    print(f"    PBU {pbu_index}:")
                    print(f"      Size     : {pbu_size} bytes")
                    print(f"      Type     : {pbu_type} (0x{pbu_type:02X})")
                    print(f"      Group ID : {group_id}")
                    print(f"      Reserved : {reserved} (should be 0)")

                    payload_after_header = pbu_data[4:]
                    preview = payload_after_header[:16].hex(' ', 1)
                    print(f"      Payload Preview (first 16 bytes after header): {preview}...")

                    if 1 <= pbu_type <= 2 or 25 <= pbu_type <= 27:
                        print(f"      -> Contains frame() data")
                        self.parse_frame(pbu_reader)
                    elif pbu_type == 65:
                        print(f"      -> Contains au_info()")
                    elif pbu_type == 66:
                        print(f"      -> Contains metadata()")
                    elif pbu_type == 67:
                        print(f"      -> Contains filler()")
                    else:
                        print(f"      -> Unknown or unhandled pbu_type")

                    pbu_index += 1

                self.frames.append(au_payload)
                index += 1

            except Exception as e:
                print(f"Error parsing AU {index}: {e}")
                break

    def parse_tile(self, reader: BitstreamReader, tile_idx: int, num_comps: int):

        tile_header_size = int.from_bytes(reader.read_bytes(2), 'big')
        tile_index = int.from_bytes(reader.read_bytes(2), 'big')
        tile_data_sizes = [int.from_bytes(reader.read_bytes(4), 'big') for _ in range(num_comps)]
        tile_qps = [int.from_bytes(reader.read_bytes(1), 'big') for _ in range(num_comps)]
        reserved = int.from_bytes(reader.read_bytes(1), 'big')
        reader.byte_align()

        print(f"              tile_header_size: {tile_header_size}")
        print(f"              tile_index: {tile_index}")
        print(f"              tile_data_sizes: {tile_data_sizes}")
        print(f"              tile_qp: {tile_qps}")
        print(f"              reserved_zero_8bits: {reserved}")

        MbWidth = 16
        MbHeight = 16
        ColStarts = self.ColStarts
        RowStarts = self.RowStarts
        TileCols = self.TileCols

        x0 = ColStarts[tile_idx % TileCols]
        y0 = RowStarts[tile_idx // TileCols]
        numMbColsInTile = (ColStarts[(tile_idx % TileCols) + 1] - ColStarts[tile_idx % TileCols]) // MbWidth
        numMbRowsInTile = (RowStarts[(tile_idx // TileCols) + 1] - RowStarts[tile_idx // TileCols]) // MbHeight
        numMbsInTile = numMbColsInTile * numMbRowsInTile

        for cIdx in range(num_comps):
            print(f"              tile_data for component {cIdx}:")
            mb_data = reader.read_bytes(tile_data_sizes[cIdx])
            subreader = BitstreamReader(mb_data)
            for i in range(numMbsInTile):
                xMb = x0 + ((i % numMbColsInTile) * MbWidth)
                yMb = y0 + ((i // numMbColsInTile) * MbHeight)
                print(f"                macroblock_layer at xMb={xMb}, yMb={yMb}, cIdx={cIdx}")
                preview = mb_data[subreader.byte_pos:subreader.byte_pos + 16].hex(' ', 1)
                print(f"                  Preview: {preview}...")
                # Skipping full macroblock_layer parsing
                break
            subreader.byte_align()

        dummy_count = 0
        while reader.more_data():
            _ = reader.read_bytes(1)
            dummy_count += 1
        if dummy_count > 0:
            print(f"              tile_dummy_byte: {dummy_count} trailing bytes skipped")

    def parse_frame(self, reader: BitstreamReader):
        print("        Parsing frame_header()")

        profile_idc = int.from_bytes(reader.read_bytes(1), 'big')
        level_idc = int.from_bytes(reader.read_bytes(1), 'big')

        byte = int.from_bytes(reader.read_bytes(1), 'big')
        band_idc = (byte >> 5) & 0x07
        reserved5 = byte & 0x1F

        frame_width = int.from_bytes(reader.read_bytes(3), 'big')
        frame_height = int.from_bytes(reader.read_bytes(3), 'big')

        byte = int.from_bytes(reader.read_bytes(1), 'big')
        chroma_format_idc = (byte >> 4) & 0x0F
        bit_depth_minus8 = byte & 0x0F

        num_comps = self.get_num_components(chroma_format_idc)

        capture_time_distance = int.from_bytes(reader.read_bytes(1), 'big')
        reserved_zero_8bits = int.from_bytes(reader.read_bytes(1), 'big')

        print(f"          frame_info():")
        print(f"            profile_idc: {profile_idc}")
        print(f"            level_idc: {level_idc}")
        print(f"            band_idc: {band_idc}")
        print(f"            frame_width: {frame_width}")
        print(f"            frame_height: {frame_height}")
        print(f"            chroma_format_idc: {chroma_format_idc}")
        print(f"            bit_depth: {bit_depth_minus8 + 8}")
        print(f"            capture_time_distance: {capture_time_distance}")

        reserved0 = int.from_bytes(reader.read_bytes(1), 'big')
        print(f"          reserved_zero_8bits: {reserved0}")

        color_description_present_flag = reader.read_bits(1)
        print(f"          color_description_present_flag: {color_description_present_flag}")

        if color_description_present_flag:
            color_primaries = int.from_bytes(reader.read_bytes(1), 'big')
            transfer_characteristics = int.from_bytes(reader.read_bytes(1), 'big')
            matrix_coefficients = int.from_bytes(reader.read_bytes(1), 'big')
            full_range_flag = int.from_bytes(reader.read_bytes(1), 'big') >> 7
            print(f"            color_primaries: {color_primaries}")
            print(f"            transfer_characteristics: {transfer_characteristics}")
            print(f"            matrix_coefficients: {matrix_coefficients}")
            print(f"            full_range_flag: {full_range_flag}")

        use_q_matrix = reader.read_bits(1)
        print(f"          use_q_matrix: {use_q_matrix}")

        if use_q_matrix:
            print(f"            quantization_matrix():")
            for comp in range(3):
                print(f"              Component {comp}:")
                for y in range(8):
                    row = [int.from_bytes(reader.read_bytes(1), 'big') for x in range(8)]
                    print(f"                {row}")

        print("          tile_info():")
        tile_width_in_mbs = reader.read_bits(20)
        tile_height_in_mbs = reader.read_bits(20)
        print(f"            tile_width_in_mbs: {tile_width_in_mbs}")
        print(f"            tile_height_in_mbs: {tile_height_in_mbs}")

        MbWidth = 16
        MbHeight = 16
        FrameWidthInMbsY = (frame_width // 16)
        FrameHeightInMbsY = (frame_height // 16)

        ColStarts = []
        startMb = 0
        while startMb < FrameWidthInMbsY:
            ColStarts.append(startMb * MbWidth)
            startMb += tile_width_in_mbs
        ColStarts.append(FrameWidthInMbsY * MbWidth)
        TileCols = len(ColStarts) - 1

        RowStarts = []
        startMb = 0
        while startMb < FrameHeightInMbsY:
            RowStarts.append(startMb * MbHeight)
            startMb += tile_height_in_mbs
        RowStarts.append(FrameHeightInMbsY * MbHeight)
        TileRows = len(RowStarts) - 1

        NumTiles = TileCols * TileRows
        print(f"            TileCols: {TileCols}, TileRows: {TileRows}, NumTiles: {NumTiles}")

        self.ColStarts = ColStarts
        self.RowStarts = RowStarts
        self.TileCols = TileCols

        tile_size_present_in_fh_flag = reader.read_bits(1)
        if tile_size_present_in_fh_flag:
            print("            tile_size_present_in_fh_flag: 1")
            tile_size_in_fh = [int.from_bytes(reader.read_bytes(4), 'big') for _ in range(NumTiles)]
            print(f"              tile_size_in_fh: {tile_size_in_fh}")
        else:
            print("            tile_size_present_in_fh_flag: 0")

        reserved1 = reader.read_bits(8)
        print(f"          reserved_zero_8bits: {reserved1}")
        reader.byte_align()
        print("          byte_alignment(): aligned to next byte boundary")

        for i in range(NumTiles):
            print(f"            Parsing tile_header() for tile {i}:")
            tile_size = int.from_bytes(reader.read_bytes(4), 'big')
            tile_data = reader.read_bytes(tile_size)
            subreader = BitstreamReader(tile_data)
            print(f"              tile_size: {tile_size}")
            self.parse_tile(subreader, i, num_comps)

# Example usage
decoder = APVDecoder("../test/bitstream/tile_B.apv")
decoder.parse_access_units()