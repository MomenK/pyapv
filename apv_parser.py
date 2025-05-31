import struct
import numpy as np
import argparse
import matplotlib.pyplot as plt

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
    
    def parse_exp_golomb(self, k_param: int):
        """
        Implements Fig-25 ‘parsing process of symbolValue’ from the spec.
        k_param is the context-adaptation parameter (kParam in the text).
        """
        symbol_value = 0
        parse_exp_golomb = True
        k = max(0, k_param)          # k is allowed to be zero
        stopLoop = False

        first_bit = self.read_bit()

        if first_bit == 1:           # 1 ––> symbolValue = 0, done
            parse_exp_golomb = False
        else:
            second_bit = self.read_bit()
            if second_bit == 0:      # 01 ––> add 1<<k
                symbol_value += 1 << k
                parse_exp_golomb = False
            else:                    # 00 ––> add 2<<k  and continue Exp-Golomb
                symbol_value += 2 << k
                parse_exp_golomb = True

        if parse_exp_golomb:
            # unary ’0…01’ prefix
            while True:
                if self.read_bit() == 1:
                    stopLoop = True
                else:
                    symbol_value += 1 << k
                    k += 1

                if stopLoop == True:
                    break
        # now k ≥ 0   → read k LSBs
        if k > 0:
            symbol_value += self.read_bits(k)

        return symbol_value

class APVDecoder:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            self.data = f.read()
        self.reader = BitstreamReader(self.data)
        self.frames = []

    def setup_frame_buffers(self, width, height, num_comps, SubWidthC, SubHeightC):
        self.rec_samples = []
        for cIdx in range(num_comps):
            w = width if cIdx == 0 else width // SubWidthC
            h = height if cIdx == 0 else height // SubHeightC
            self.rec_samples.append(np.zeros((h, w), dtype=np.uint8))
    
    

    def clip(self, min_val, max_val, val):
        return max(min_val, min(max_val, val))

    def scale_transform_coefficients(self, coeff_block, cIdx, qp):
        levelScale = [40, 45, 51, 57, 64, 71]
        QMatrix = np.ones((8, 8), dtype=np.int32)
        qP = qp
        bdShift = 8 + ((3 + 3) // 2) - 5

        d = np.zeros((8, 8), dtype=np.int32)
        for y in range(8):
            for x in range(8):
                val = coeff_block[y][x] * QMatrix[y][x] * levelScale[qP % 6]
                val = val << (qP // 6)
                val = (val + (1 << (bdShift - 1))) >> bdShift
                d[y][x] = self.clip(-32768, 32767, val)

        return d

    def inverse_transform(self, block):
        transMatrix = np.array([
            [64, 64, 64, 64, 64, 64, 64, 64],
            [89, 75, 50, 18, -18, -50, -75, -89],
            [84, 35, -35, -84, -84, -35, 35, 84],
            [75, -18, -89, -50, 50, 89, 18, -75],
            [64, -64, -64, 64, 64, -64, -64, 64],
            [50, -89, 18, 75, -75, -18, 89, -50],
            [35, -84, 84, -35, -35, 84, -84, 35],
            [18, -50, 75, -89, 89, -75, 50, -18]
        ])
        temp = np.dot(transMatrix, block)
        temp = (temp + 64) >> 7
        result = np.dot(temp, transMatrix.T)
        result = (result + 64) >> 7
        return result

    def parse_macroblock_layer(self, xMb, yMb, cIdx, SubWidthC, SubHeightC, qp):
        subW = 1 if cIdx == 0 else SubWidthC
        subH = 1 if cIdx == 0 else SubHeightC
        MbWidth = 16
        MbHeight = 16
        blkWidth = MbWidth if cIdx == 0 else MbWidth // SubWidthC
        blkHeight = MbHeight if cIdx == 0 else MbHeight // SubHeightC
        TrSize = 8

        self.TransCoeff = np.zeros((TrSize, TrSize), dtype=np.int32)

        for y in range(0, blkHeight, TrSize):
            for x in range(0, blkWidth, TrSize):
                # print(f" start dc_coeff_coding")
                kParam = self.clip(0, 5, self.PrevDCDiff >> 1)
                abs_dc_coeff_diff = self.mb_reader.parse_exp_golomb(kParam)
                
                if abs_dc_coeff_diff:
                    sign_dc_coeff_diff = self.mb_reader.read_bit()
                else:
                    sign_dc_coeff_diff = 0
                self.TransCoeff[0][0] = self.PrevDC + abs_dc_coeff_diff * (1 - (2 * sign_dc_coeff_diff))
                self.PrevDC = self.TransCoeff[0][0]
                self.PrevDCDiff = abs_dc_coeff_diff
                # print (f"remaining bytes {self.mb_reader.remaining_bytes()}")
                # print(f" DC coeff {self.PrevDC}")
                # self.Prev1stAcLevel = self.ac_coeff_coding(
                #     reader, xMb // subW + x, yMb // subH + y, 3, 3, cIdx, self.Prev1stAcLevel
                # )

                log2BlkWidth = 3
                log2BlkHeight = 3

                # print(f" start ac_coeff_coding")
                scanPos = 1
                firstAC = 1
                PrevLevel = self.Prev1stAcLevel
                PrevRun = 0

                blk_size = 1 << (log2BlkWidth + log2BlkHeight)
                width = 1 << log2BlkWidth
                height = 1 << log2BlkHeight

                # Use zig-zag scan order for 8x8 block
                # ScanOrder = self.build_zigzag_scan(width, height)
                # print(ScanOrder)

                ScanOrder = [
                    0,  1,  8, 16,  9,  2,  3, 10,
                    17, 24, 32, 25, 18, 11,  4,  5,
                    12, 19, 26, 33, 40, 48, 41, 34,
                    27, 20, 13,  6,  7, 14, 21, 28,
                    35, 42, 49, 56, 57, 50, 43, 36,
                    29, 22, 15, 23, 30, 37, 44, 51,
                    58, 59, 52, 45, 38, 31, 39, 46,
                    53, 60, 61, 54, 47, 55, 62, 63
                ]


                while scanPos < blk_size:
                    kRun = self.clip(0, 2, PrevRun >> 2)
                    coeff_zero_run = self.mb_reader.parse_exp_golomb(kRun)
                    # print(f" coeff_zero_run {coeff_zero_run} - scanPos {scanPos}" )

                    for _ in range(coeff_zero_run):
                        blkPos = ScanOrder[scanPos]
                        xC = blkPos & (width - 1)
                        yC = blkPos >> log2BlkWidth
                        self.TransCoeff[yC][xC] = 0
                        scanPos += 1

                    PrevRun = coeff_zero_run

                    if scanPos < blk_size:
                        kLevel = self.clip(0, 4, PrevLevel >> 2)
                        abs_ac_coeff_minus1 = self.mb_reader.parse_exp_golomb(kLevel)
                        sign_ac_coeff = self.mb_reader.read_bits(1)
                        level = (abs_ac_coeff_minus1 + 1) * (1 - (2 * sign_ac_coeff))

                        blkPos = ScanOrder[scanPos]
                        xC = blkPos & (width - 1)
                        yC = blkPos >> log2BlkWidth
                        self.TransCoeff[yC][xC] = level
                        scanPos += 1

                        PrevLevel = abs_ac_coeff_minus1 + 1
                        if firstAC:
                            firstAC = 0
                            self.Prev1stAcLevel = PrevLevel
                    if (scanPos >= blk_size): 
                        break
            

                scaled_block = self.scale_transform_coefficients(self.TransCoeff, cIdx, qp)
                rec_block = self.inverse_transform(scaled_block)

                # print(f"                    Block @ ({xMb + x}, {yMb + y}), cIdx={cIdx}: coeff ->")
                # for row in ac_block:
                #     print(f"                      {list(row)}")
                # print(f"                    Reconstructed samples:")
                # for row in rec_block:
                #     print(f"                      {list(row)}")

                if hasattr(self, 'rec_samples') and self.rec_samples[cIdx] is not None:
                    for i in range(TrSize):
                        for j in range(TrSize):
                            yy = yMb // subH + y + i
                            xx = xMb // subW + x + j
                            if 0 <= yy < self.rec_samples[cIdx].shape[0] and 0 <= xx < self.rec_samples[cIdx].shape[1]:
                                # print(f"                    Writing into frame buffer component {cIdx} value {self.clip(0, 255, rec_block[i, j])}")
                                self.rec_samples[cIdx][yy, xx] = self.clip(0, 255, rec_block[i, j])
                    # self.save_frame_as_image("frame_output.png")

    def build_zigzag_scan(self, blk_w: int, blk_h: int):
        """
        Returns a list of length blk_w*blk_h with the classical zig-zag
        ordering used by the APV spec. Implements the pseudocode in the
        user’s note (top-left DC first, then anti-diagonals that alternate
        R↘︎L and L↗︎R).
        """
        scan = [0] * (blk_w * blk_h)
        pos  = 0
        scan[pos] = 0          # DC at (0,0)
        pos += 1

        for line in range(1, blk_w + blk_h - 1):
            if line & 1:       # odd → ↙︎ direction (x dec, y inc)
                x = min(line, blk_w  - 1)
                y = max(0,   line - (blk_w  - 1))
                while x >= 0 and y < blk_h:
                    scan[pos] = y * blk_w + x
                    pos += 1
                    x  -= 1
                    y  += 1
            else:              # even → ↗︎ direction (y dec, x inc)
                y = min(line, blk_h - 1)
                x = max(0,   line - (blk_h - 1))
                while y >= 0 and x < blk_w:
                    scan[pos] = y * blk_w + x
                    pos += 1
                    x  += 1
                    y  -= 1
        return scan

    # def ac_coeff_coding(self, reader, x0, y0, log2BlkWidth, log2BlkHeight, cIdx, self.Prev1stAcLevel):
    #     print(f" start ac_coeff_coding")
    #     scanPos = 1
    #     firstAC = 1
    #     PrevLevel = self.Prev1stAcLevel
    #     PrevRun = 0

    #     blk_size = 1 << (log2BlkWidth + log2BlkHeight)
    #     width = 1 << log2BlkWidth
    #     height = 1 << log2BlkHeight

    #     # Use zig-zag scan order for 8x8 block
    #     # ScanOrder = self.build_zigzag_scan(width, height)
    #     # print(ScanOrder)

    #     ScanOrder = [
    #         0,  1,  8, 16,  9,  2,  3, 10,
    #         17, 24, 32, 25, 18, 11,  4,  5,
    #         12, 19, 26, 33, 40, 48, 41, 34,
    #         27, 20, 13,  6,  7, 14, 21, 28,
    #         35, 42, 49, 56, 57, 50, 43, 36,
    #         29, 22, 15, 23, 30, 37, 44, 51,
    #         58, 59, 52, 45, 38, 31, 39, 46,
    #         53, 60, 61, 54, 47, 55, 62, 63
    #     ]


    #     while True:
    #         kRun = self.clip(0, 2, PrevRun >> 2)
    #         coeff_zero_run = reader.parse_exp_golomb(kRun)
    #         print(f" coeff_zero_run {coeff_zero_run} - scanPos {scanPos}" )

    #         for _ in range(coeff_zero_run):
    #             # if scanPos >= blk_size:
    #             #     break
    #             blkPos = ScanOrder[scanPos]
    #             xC = blkPos & (width - 1)
    #             yC = blkPos >> log2BlkWidth
    #             self.TransCoeff[yC][xC] = 0
    #             scanPos += 1

    #         PrevRun = coeff_zero_run

    #         if scanPos < blk_size:
    #             kLevel = self.clip(0, 4, PrevLevel >> 2)
    #             abs_ac_coeff_minus1 = reader.parse_exp_golomb(kLevel)
    #             sign_ac_coeff = reader.read_bits(1)
    #             level = (abs_ac_coeff_minus1 + 1) * (1 - (2 * sign_ac_coeff))

    #             blkPos = ScanOrder[scanPos]
    #             xC = blkPos & (width - 1)
    #             yC = blkPos >> log2BlkWidth
    #             self.TransCoeff[yC][xC] = level
    #             scanPos += 1

    #             PrevLevel = abs_ac_coeff_minus1 + 1
    #             if firstAC:
    #                 firstAC = 0
    #                 self.Prev1stAcLevel = PrevLevel
    #         if (scanPos >= blk_size): 
    #             break
    #     return self.Prev1stAcLevel

    def get_chroma_subsampling_factors(self,chroma_format_idc):
        if chroma_format_idc == 0:  # 4:0:0
            return 1, 1
        elif chroma_format_idc == 2:  # 4:2:2
            return 2, 1
        elif chroma_format_idc == 3:  # 4:4:4
            return 1, 1
        elif chroma_format_idc == 4:  # 4:4:4:4
            return 1, 1
        else:
            raise ValueError(f"Unsupported or reserved chroma_format_idc: {chroma_format_idc}")

        
    def get_num_components(self, chroma_format_idc: int) -> int:
        if chroma_format_idc == 0:
            return 1          # 4:0:0 (monochrome)
        elif chroma_format_idc in (2, 3):
            return 3          # 4:2:2 or 4:4:4
        elif chroma_format_idc == 4:
            return 4          # 4:4:4:4
        else:
            raise ValueError(f"Unsupported / reserved chroma_format_idc {chroma_format_idc}")

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

        x0 = self.ColStarts[tile_idx % self.TileCols]
        y0 = self.RowStarts[tile_idx // self.TileCols]
        numMbColsInTile  = (self.ColStarts[(tile_idx % self.TileCols)+1] - x0) // MbWidth
        numMbRowsInTile  = (self.RowStarts[(tile_idx // self.TileCols)+1] - y0) // MbHeight
        max_mbs_in_tile  = numMbColsInTile * numMbRowsInTile
        SubWidthC, SubHeightC = self.get_chroma_subsampling_factors(self.chroma_format_idc)

        for cIdx in range(num_comps):
            print(f"              tile_data for component {cIdx}:")
            mb_data = reader.read_bytes(tile_data_sizes[cIdx])
            self.mb_reader = BitstreamReader(mb_data)
            print (f"remaining bytes {self.mb_reader.remaining_bytes()}")
            self.PrevDC = 0
            self.PrevDCDiff = 20
            self.Prev1stAcLevel = 0

            for i in range(max_mbs_in_tile):
                xMb = x0 + ((i % numMbColsInTile) * MbWidth)
                yMb = y0 + ((i // numMbColsInTile) * MbHeight)
                print(f"                macroblock_layer at xMb={xMb}, yMb={yMb} {i}/{max_mbs_in_tile}, cIdx={cIdx}")
                # preview = mb_data[subreader.byte_pos:subreader.byte_pos + 16].hex(' ', 1)
                # print(f"                  Preview: {preview}...")
                
                self.parse_macroblock_layer(xMb, yMb, cIdx, SubWidthC, SubHeightC,tile_qps[cIdx])
            # self.save_frame_as_image("frame_output.png")
            reader.byte_align()

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
        self.chroma_format_idc = (byte >> 4) & 0x0F
        bit_depth_minus8 = byte & 0x0F

        num_comps = self.get_num_components(self.chroma_format_idc)
        SubWidthC, SubHeightC = self.get_chroma_subsampling_factors(self.chroma_format_idc)
        self.setup_frame_buffers(frame_width, frame_height, num_comps, SubWidthC, SubHeightC)

        capture_time_distance = int.from_bytes(reader.read_bytes(1), 'big')
        reserved_zero_8bits = int.from_bytes(reader.read_bytes(1), 'big')

        print(f"          frame_info():")
        print(f"            profile_idc: {profile_idc}")
        print(f"            level_idc: {level_idc}")
        print(f"            band_idc: {band_idc}")
        print(f"            frame_width: {frame_width}")
        print(f"            frame_height: {frame_height}")
        print(f"            chroma_format_idc: {self.chroma_format_idc}")
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
        
        self.save_frame_as_image("frame_output.png")

    def save_frame_as_image(self, filename):
        if len(self.rec_samples) >= 3:
            y, cb, cr = self.rec_samples[:3]
            # cb_resized = np.repeat(np.repeat(cb, 2, axis=0), 2, axis=1)
            # cr_resized = np.repeat(np.repeat(cr, 2, axis=0), 2, axis=1)
            # ycbcr = np.stack((y, cb_resized, cr_resized), axis=2).astype(np.uint8)
            # rgb = self.ycbcr_to_rgb(ycbcr)
            plt.imshow(y, cmap='gray')
            plt.title("Reconstructed Luma (Y)")
            plt.axis("off")
            plt.show()

            # img = Image.fromarray(y, 'RGB')
            # img.save(filename)
            # print(f"Saved reconstructed frame to {filename}")

    def ycbcr_to_rgb(self, ycbcr):
        m = np.array([
            [1.0,  0.0,       1.402],
            [1.0, -0.344136, -0.714136],
            [1.0,  1.772,     0.0]
        ])
        ycbcr = ycbcr.astype(np.float32)
        ycbcr[..., 1:] -= 128.0
        rgb = ycbcr @ m.T
        return np.clip(rgb, 0, 255).astype(np.uint8)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APV Bitstream Parser")
    parser.add_argument("filepath", help="Path to the .apv bitstream file")
    args = parser.parse_args()

    decoder = APVDecoder(args.filepath)
    decoder.parse_access_units()