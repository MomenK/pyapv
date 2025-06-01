import struct
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image

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
    
class TileComp:
    def __init__(self, cIdx, reader : BitstreamReader, configs):
        self.reader = reader
        self.configs = configs

        self.x0 = 0 #Time coordinates
        self.y0 = 0 #Not coordinates
    
        self.MbWidth = 16
        self.MbHeight = 16
        self.TrSize = 8
        self.subW = 1 if cIdx == 0 else configs["SubWidthC"]
        self.subH = 1 if cIdx == 0 else configs["SubHeightC"]
        self.blkWidth    = self.MbWidth  if cIdx == 0 else self.MbWidth  // configs["SubWidthC"]
        self.blkHeight   = self.MbHeight if cIdx == 0 else self.MbHeight // configs["SubHeightC"]
        self.qp = configs["QP"]


        self.PrevDC = 0
        self.PrevDCDiff = 20
        self.Prev1stAcLevel = 0

        # tile_height =  (self.configs["numMbsInTile"] // self.configs["numMbColsInTile"]) * self.blkHeight
        # tile_width  =  self.configs["numMbColsInTile"] * self.blkWidth
        tile_height =  (self.configs["numMbsInTile"] // self.configs["numMbColsInTile"]) * self.MbHeight
        tile_width  =  self.configs["numMbColsInTile"] * self.MbWidth 

        print(f"Creating image buffer of size {tile_height} X {tile_width} for component {cIdx}")
        self.rec_samples = np.zeros((tile_height,tile_width), dtype=np.int16) #TODO: Correct datatype?


    def decode(self):
        # macroblock_layer iteration.
        for i in range(self.configs["numMbsInTile"]):
            xMb = self.x0 + ((i % self.configs["numMbColsInTile"]) * self.MbWidth)
            yMb = self.y0 + ((i // self.configs["numMbColsInTile"]) * self.MbHeight)
            # print(f"                macroblock_layer at xMb={xMb}, yMb={yMb} {i}/{self.configs['numMbsInTile']} - MB={self.blkHeight} x {self.blkWidth} ")


            self.TransCoeff = np.zeros((self.TrSize, self.TrSize), dtype=np.int32)

            for y in range(0, self.blkHeight, self.TrSize):
                for x in range(0, self.blkWidth, self.TrSize):
                    #Entropy

                    # print(f" start dc_coeff_coding")
                    kParam = self.clip(0, 5, self.PrevDCDiff >> 1)
                    abs_dc_coeff_diff = self.reader.parse_exp_golomb(kParam)
                    
                    if abs_dc_coeff_diff:
                        sign_dc_coeff_diff = self.reader.read_bit()
                    else:
                        sign_dc_coeff_diff = 0
                    self.TransCoeff[0][0] = self.PrevDC + abs_dc_coeff_diff * (1 - (2 * sign_dc_coeff_diff))
                    self.PrevDC = self.TransCoeff[0][0]
                    self.PrevDCDiff = abs_dc_coeff_diff


                    # print(f" start ac_coeff_coding")
                    log2BlkWidth  = 3 # log2(self.TrSize)
                    log2BlkHeight = 3 # log2(self.TrSize)

                    scanPos = 1
                    firstAC = 1
                    PrevLevel = self.Prev1stAcLevel
                    PrevRun = 0

                    blk_size = 1 << (log2BlkWidth + log2BlkHeight)
                    width = 1 << log2BlkWidth
                    height = 1 << log2BlkHeight

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
                        coeff_zero_run = self.reader.parse_exp_golomb(kRun)

                        for _ in range(coeff_zero_run):
                            blkPos = ScanOrder[scanPos]
                            xC = blkPos & (width - 1)
                            yC = blkPos >> log2BlkWidth
                            self.TransCoeff[yC][xC] = 0
                            scanPos += 1

                        PrevRun = coeff_zero_run

                        if scanPos < blk_size:
                            kLevel = self.clip(0, 4, PrevLevel >> 2)
                            abs_ac_coeff_minus1 = self.reader.parse_exp_golomb(kLevel)
                            sign_ac_coeff = self.reader.read_bits(1)
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
                
                    #Compute
                    scaled_block = self.scale_transform_coefficients(self.TransCoeff)
                    rec_block = self.inverse_transform(scaled_block)

                    #Store data in local memory
                    if hasattr(self, 'rec_samples') and self.rec_samples is not None:
                        for i in range(self.TrSize):
                            for j in range(self.TrSize):
                                yy = yMb // self.subH + y + i
                                xx = xMb // self.subW + x + j
                                if 0 <= yy < self.rec_samples.shape[0] and 0 <= xx < self.rec_samples.shape[1]:
                                    # print(f"                    Writing into frame buffer component value {self.clip(0, 255, rec_block[i, j])}")
                                    self.rec_samples[yy, xx] = rec_block[i, j]

    def dump_data(self):
        return self.rec_samples


    def clip(self, min_val, max_val, val):
        return max(min_val, min(max_val, val))

    def scale_transform_coefficients(self, coeff_block):
        levelScale = [40, 45, 51, 57, 64, 71]
        QMatrix = np.ones((8, 8), dtype=np.int32)
        qP = self.qp
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
            self.rec_samples.append(np.zeros((h, w), dtype=np.int16))
    
    
    def clip(self, min_val, max_val, val):
        return max(min_val, min(max_val, val))
        
    def parse_chroma_format_idc (self,chroma_format_idc):
        if chroma_format_idc == 0:  # 4:0:0
            return 1, 1, 1
        elif chroma_format_idc == 2:  # 4:2:2
            return 3, 2, 1
        elif chroma_format_idc == 3:  # 4:4:4
            return 3, 1, 1
        elif chroma_format_idc == 4:  # 4:4:4:4
            return 4, 1, 1
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

    def parse_tile(self, reader: BitstreamReader, tile_idx: int):

        num_comps, SubWidthC, SubHeightC = self.parse_chroma_format_idc(self.chroma_format_idc)

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


        for cIdx in range(num_comps):
            # Parallelism breaks at tile component
            # Need a TileComp Class: has a bitstream/entropy decoder and compute functions.
            mb_data = BitstreamReader(reader.read_bytes(tile_data_sizes[cIdx]))
            print (f"             tile_data for component {cIdx}: {mb_data.remaining_bytes()} Bytes")

            configs = {
                "SubWidthC":       SubWidthC,
                "SubHeightC":      SubHeightC,
                "numMbColsInTile": numMbColsInTile,
                "numMbsInTile":    max_mbs_in_tile,
                "QP":              tile_qps[cIdx],
            }
            tile_comp = TileComp(cIdx,mb_data, configs)
            tile_comp.decode()
            
            subW = 1 if cIdx == 0 else SubWidthC
            subH = 1 if cIdx == 0 else SubHeightC
            if hasattr(self, 'rec_samples') and self.rec_samples[cIdx] is not None:
                for i in range(numMbRowsInTile * 16):
                    for j in range(numMbColsInTile * 16):
                        yy = y0 // subH + i
                        xx = x0 // subW  + j

                        if 0 <= yy < self.rec_samples[cIdx].shape[0] and 0 <= xx < self.rec_samples[cIdx].shape[1]:
                            # print(f"                    Writing into frame buffer component {cIdx} value {self.clip(0, 255, rec_block[i, j])}")
                            self.rec_samples[cIdx][yy, xx] = tile_comp.dump_data()[i,j]

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

        num_comps, SubWidthC, SubHeightC = self.parse_chroma_format_idc(self.chroma_format_idc)
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
            self.parse_tile(subreader, i)
        
        self.save_frame_as_image("frame_output.png",SubWidthC, SubHeightC)

    def save_frame_as_image(self, filename,SubWidthC, SubHeightC):
        if len(self.rec_samples) >= 3:
            y, cb, cr = self.rec_samples[:3]
            # print(f"{SubWidthC}, {SubHeightC}")
            if SubHeightC==2:
                cb = np.repeat( cb, 2, axis=0)
                cr = np.repeat( cr, 2, axis=0)
            if SubWidthC==2:
                cb = np.repeat( cb, 2, axis=1)
                cr = np.repeat( cr, 2, axis=1)

            print(f"{y.shape} {cr.shape}  {cb.shape}  ")
            # cb_resized = np.repeat(np.repeat(cb, 2, axis=0), 2, axis=1)
            # cr_resized = np.repeat(np.repeat(cr, 2, axis=0), 2, axis=1)

            # --- 2)  add 128 offsets & clip into uint8 --------------------
            y8, cb8, cr8 = self._prepare_planes_for_display(y, cb, cr,
                                                            full_range=False)

            # --- 3)  YCbCr → RGB  (BT.601 full-swing) ---------------------
            ycbcr = np.stack((y8, cb8, cr8), axis=2)
            rgb   = self.ycbcr_to_rgb(ycbcr)          # result already uint8

            print(f"{rgb.shape}")
            plt.imshow(rgb)
            # plt.title("Reconstructed Luma (Y)")
            plt.axis("off")
            plt.show()
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))   # 1 row × 3 cols

            planes = [
                (y,  "Reconstructed Luma (Y)"),
                (cb, "Reconstructed Chroma (Cb)"),
                (cr, "Reconstructed Chroma (Cr)"),
            ]

            for ax, (img, title) in zip(axes, planes):
                ax.imshow(img, cmap='gray')
                ax.set_title(title)
                ax.axis('off')

            plt.tight_layout()
            plt.show()

            Image.fromarray(rgb).save('binary_rgb.png')    # default mode='RGB'

            # img = Image.fromarray(y, 'RGB')
            # img.save(filename)
            # print(f"Saved reconstructed frame to {filename}")

    def _prepare_planes_for_display(self, y, cb, cr, full_range=False):
        """
        * Adds 128 to luma (if not full-range) and 128 to Cb/Cr.
        * Clips to 0-255 and converts to uint8.
        """
        if not full_range:
            y  = y  + 128.0                # shift luma up
        cb = cb + 128.0
        cr = cr + 128.0
        return (np.clip(0,255,y).astype(np.int16),
                np.clip(0,255,cb).astype(np.int16),
                np.clip(0,255,cr).astype(np.int16))
    
    def ycbcr_to_rgb(self, ycbcr):
        """
        BT.601 full-range conversion (values are uint8 in 0‥255).
        """
        y = ycbcr[...,0].astype(np.int16)
        cb= ycbcr[...,1].astype(np.int16) - 128
        cr= ycbcr[...,2].astype(np.int16) - 128

        r = y +             1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772    * cb

        rgb = np.stack((r,g,b), axis=2)
        return np.clip(rgb, 0, 255).astype(np.uint8)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APV Bitstream Parser")
    parser.add_argument("filepath", help="Path to the .apv bitstream file")
    args = parser.parse_args()

    decoder = APVDecoder(args.filepath)
    decoder.parse_access_units()