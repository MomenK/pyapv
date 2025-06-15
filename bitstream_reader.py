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
 