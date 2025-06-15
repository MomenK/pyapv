import numpy as np
import argparse
from bitstream_reader import BitstreamReader
from frame_viewer import FrameViewer 
from tile_decoder import TileComp

class APVDecoder:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            self.data = f.read()
        self.reader = BitstreamReader(self.data)
        self.frame_index = 0


    def setup_frame_buffer(self, width, height, num_comps, SubWidthC, SubHeightC):
        self.frame_buffer = []
        for cIdx in range(num_comps):
            w = width if cIdx == 0 else width // SubWidthC
            h = height if cIdx == 0 else height // SubHeightC
            self.frame_buffer.append(np.zeros((h, w), dtype=np.uint16))
    
    
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
        
    def parse_bitstream(self):
        r = self.reader
        au_index = 0
        while r.more_data():
            try:
                au_size = int.from_bytes(r.read_bytes(4), 'big')
                au_data = r.read_bytes(au_size)
                au_reader = BitstreamReader(au_data)

                signature = au_reader.read_bytes(4).decode('ascii', errors='replace')
                print(f"\nAccess Unit {au_index}:")
                print(f"  Signature: {signature}")
                print(f"  Total size: {au_size} bytes")

                self.parse_access_units(au_reader)
                au_index += 1

            except Exception as e:
                print(f"Error parsing AU {au_index}: {e}")
                break


    def parse_access_units(self, au_reader: BitstreamReader):
        pbu_index = 0
        while au_reader.remaining_bytes():
            pbu_size = int.from_bytes(au_reader.read_bytes(4), 'big')
            pbu_data = au_reader.read_bytes(pbu_size)

            pbu_reader = BitstreamReader(pbu_data)
            pbu_type = int.from_bytes(pbu_reader.read_bytes(1), 'big')
            group_id = int.from_bytes(pbu_reader.read_bytes(2), 'big')
            reserved = int.from_bytes(pbu_reader.read_bytes(1), 'big')

            print(f"    PBU {pbu_index}:")
            print(f"      Size     : {pbu_size} bytes")
            print(f"      Type     : {pbu_type} (0x{pbu_type:02X})")
            print(f"      Group ID : {group_id}")
            print(f"      Reserved : {reserved} (should be 0)")

            if 1 <= pbu_type <= 2 or 25 <= pbu_type <= 27:
                print(f"      -> Contains frame() data")
                self.parse_frame_pbu(pbu_reader)
            elif pbu_type == 65:
                print(f"      -> Contains au_info()")
            elif pbu_type == 66:
                print(f"      -> Contains metadata()")
            elif pbu_type == 67:
                print(f"      -> Contains filler()")
            else:
                print(f"      -> Unknown or unhandled pbu_type")

            pbu_index += 1


    def parse_frame_pbu(self, reader: BitstreamReader):
        print("        Parsing frame_header()")
        self.frame_index += 1
        profile_idc = int.from_bytes(reader.read_bytes(1), 'big')
        level_idc = int.from_bytes(reader.read_bytes(1), 'big')

        byte = int.from_bytes(reader.read_bytes(1), 'big')
        band_idc = (byte >> 5) & 0x07
        reserved5 = byte & 0x1F

        frame_width = int.from_bytes(reader.read_bytes(3), 'big')
        frame_height = int.from_bytes(reader.read_bytes(3), 'big')

        byte = int.from_bytes(reader.read_bytes(1), 'big')
        self.chroma_format_idc = (byte >> 4) & 0x0F
        self.bit_depth_minus8 = byte & 0x0F
        self.BitDepth = self.bit_depth_minus8 + 8

        num_comps, SubWidthC, SubHeightC = self.parse_chroma_format_idc(self.chroma_format_idc)
        self.setup_frame_buffer(frame_width, frame_height, num_comps, SubWidthC, SubHeightC)

        capture_time_distance = int.from_bytes(reader.read_bytes(1), 'big')
        reserved_zero_8bits = int.from_bytes(reader.read_bytes(1), 'big')

        print(f"          frame_info():")
        print(f"            profile_idc: {profile_idc}")
        print(f"            level_idc: {level_idc}")
        print(f"            band_idc: {band_idc}")
        print(f"            frame_width: {frame_width}")
        print(f"            frame_height: {frame_height}")
        print(f"            chroma_format_idc: {self.chroma_format_idc}")
        print(f"            bit_depth: {self.bit_depth_minus8 + 8}")
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
            self.q_matrix = np.ones((3, 8, 8), dtype=np.int8) 
            print(f"            quantization_matrix():")
            for comp in range(num_comps):
                print(f"              Component {comp}:")
                for y in range(8):
                    for x in range(8):
                        # read one byte and store it
                        self.q_matrix[comp, y, x] = reader.read_bits(8)
                    print(f"                {self.q_matrix[comp, y].tolist()}")
        else:
            self.q_matrix = np.ones((3, 8, 8), dtype=np.int8)*16

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
            tile_reader = BitstreamReader(tile_data)
            print(f"              tile_size: {tile_size}")
            self.parse_tile(tile_reader, i)
        
        frame_viewer = FrameViewer("reconstructed_frame_"+str(self.frame_index),SubWidthC, SubHeightC, self.BitDepth, self.frame_buffer)
        frame_viewer.save_frame_as_image()


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

        xIdx = self.ColStarts[tile_idx %  self.TileCols]
        yIdx = self.RowStarts[tile_idx // self.TileCols]
        numMbColsInTile  = (self.ColStarts[(tile_idx %  self.TileCols)+1] - xIdx) // MbWidth
        numMbRowsInTile  = (self.RowStarts[(tile_idx // self.TileCols)+1] - yIdx) // MbHeight


        for cIdx in range(num_comps):
            subW = 1 if cIdx == 0 else SubWidthC
            subH = 1 if cIdx == 0 else SubHeightC

            # Parallelism breaks at tile component
            configs = {
                # required for figuring out where to write data in frame buffer
                "cIdx"              : cIdx,
                "xIdx"              : xIdx,
                "yIdx"              : yIdx,
                # "tileByteSize"      : tile_data_sizes[cIdx],
                # required for decoder
                "subW"              : subW,
                "subH"              : subH,
                "numMbColsInTile"   : numMbColsInTile,
                "numMbRowsInTile"   : numMbRowsInTile,
                "bit_depth_minus8"  : self.bit_depth_minus8,
                "QP"                : tile_qps[cIdx],
                "QMatrix"           : self.q_matrix[cIdx]
            }     
            tile_data = BitstreamReader(reader.read_bytes(tile_data_sizes[cIdx]))
            reader.byte_align()
            print (f"             tile_data for component {cIdx}: {tile_data.remaining_bytes()} Bytes")

            # Decode tile data
            tile_comp = TileComp(cIdx,tile_data, configs)
            tile_comp.decode()

            # Write reconstructed data into the frame buffer
            if hasattr(self, 'frame_buffer') and self.frame_buffer[cIdx] is not None:
                for i in range(numMbRowsInTile * 16): #tile height
                    for j in range(numMbColsInTile * 16): #tile width
                        yy = yIdx // subH + i
                        xx = xIdx // subW + j

                        if 0 <= yy < self.frame_buffer[cIdx].shape[0] and 0 <= xx < self.frame_buffer[cIdx].shape[1]:
                            # print(f"                    Writing into frame buffer component {cIdx} value {self.clip(0, 255, rec_block[i, j])}")
                            self.frame_buffer[cIdx][yy, xx] = tile_comp.dump_data()[i,j]


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APV Bitstream Parser")
    parser.add_argument("filepath", help="Path to the .apv bitstream file")
    args = parser.parse_args()

    decoder = APVDecoder(args.filepath)
    decoder.parse_bitstream() #TODO: Pass a decode flag/ either dump tile bitstreams and json log or decode image.