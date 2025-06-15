import numpy as np
from bitstream_reader import BitstreamReader

class TileComp:
    def __init__(self, configs, tile_data):
        
        self.reader = BitstreamReader(tile_data)
        self.configs = configs
    
        self.MbWidth = 16
        self.MbHeight = 16
        self.TrSize = 8
        self.subW = configs["subW"]
        self.subH = configs["subH"]
        self.blkWidth    = self.MbWidth  // self.subW
        self.blkHeight   = self.MbHeight // self.subH
        self.qp = configs["QP"]
        self.BitDepth = configs["bit_depth_minus8"] + 8
        self.QpBdOffset = configs["bit_depth_minus8"] * 6
        self.QMatrix = configs["QMatrix"]


        self.PrevDC = 0
        self.PrevDCDiff = 20
        self.Prev1stAcLevel = 0

        tile_height =  self.configs["numMbRowsInTile"] * self.blkHeight
        tile_width  =  self.configs["numMbColsInTile"] * self.blkWidth 
        # print(f"Creating tile recon buffer of size {tile_height} X {tile_width} for component {configs['cIdx']}")
        self.tile_recon_buffer = np.zeros((tile_height,tile_width), dtype=np.uint16)


    def decode(self):
        # macroblock_layer iteration.
        numMbsInTile = self.configs["numMbRowsInTile"] * self.configs["numMbColsInTile"]
        for i in range(numMbsInTile):
            xMb =  ((i % self.configs["numMbColsInTile"]) * self.MbWidth)
            yMb =  ((i // self.configs["numMbColsInTile"]) * self.MbHeight)
            # print(f"                macroblock_layer at xMb={xMb}, yMb={yMb} {i}/{numMbsInTile} - MB={self.blkHeight} x {self.blkWidth} ")


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
                    #- Inverse quantization
                    scaled_block = self.scale_transform_coefficients(self.TransCoeff)
                    #- Inverse transform
                    rec_block = self.inverse_transform(scaled_block)
                    #- Reconstruction scaling
                    bdShift = (20 - self.BitDepth)
                    for ix in range(self.TrSize):
                        for jy in range(self.TrSize):
                            val =  (rec_block[ix,jy] + (1 << (bdShift - 1))) >> bdShift
                            mid_val = (1 << (self.BitDepth - 1))
                            max_val = (1 << self.BitDepth) - 1
                            rec_block[ix,jy] = self.clip(0, max_val, val + mid_val)

                    #Store data in local memory
                    yy0 = yMb // self.subH + y
                    xx0 = xMb // self.subW + x
                    self.tile_recon_buffer[yy0:yy0+self.TrSize, xx0:xx0+self.TrSize] = rec_block

    def dump_data(self):
        return self.tile_recon_buffer


    def clip(self, min_val, max_val, val):
        return max(min_val, min(max_val, val))

    def scale_transform_coefficients(self, coeff_block):
        levelScale = [40, 45, 51, 57, 64, 71]
        qP = self.qp
        bdShift = self.BitDepth  - 2 #+ ((3 + 3) // 2) - 5

        d = np.zeros((8, 8), dtype=np.int32)
        for y in range(8):
            for x in range(8):
                val = (coeff_block[y][x] * self.QMatrix[y][x] * levelScale[qP % 6]) << (qP // 6)
                val = (val + (1 << (bdShift - 1))) >> bdShift
                d[y][x] = self.clip(-32768, 32767, val) #int16
                # d[x][y] = self.clip(-32768, 32767,((coeff_block[x][y] * QMatrix[x][y] * levelScale[qP % 6] << (qP//6)) + (1 << (bdShift-1)) >> bdShift))

        return d

    def inverse_transform(self, block):
        transMatrix = np.array([
            [  64,  64,  64,  64,  64,  64,  64,  64 ],
            [  89,  75,  50,  18, -18, -50, -75, -89 ],
            [  84,  35, -35, -84, -84, -35,  35,  84 ],
            [  75, -18, -89, -50,  50,  89,  18, -75 ],
            [  64, -64, -64,  64,  64, -64, -64,  64 ],
            [  50, -89,  18,  75, -75, -18,  89, -50 ],
            [  35, -84,  84, -35, -35,  84, -84,  35 ],
            [  18, -50,  75, -89,  89, -75,  50, -18 ]
        ])
        e_arr = np.dot(transMatrix.T, block)
        g_arr = (e_arr + 64) >> 7
        result = np.dot(g_arr, transMatrix)
        return result

