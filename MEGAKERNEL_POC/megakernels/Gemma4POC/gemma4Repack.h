#pragma once

// Host-side weight repacking for the Gemma-4 FFN MegaKernel (shared by the
// graph transformation and research/gemma4_ffn_bench).
//
// A u4 [N, K] matrix with group-64 f16 scales and u8/u4 zero points is
// rewritten so that one sub-group (16 lanes = 16 consecutive rows) reads one
// contiguous 576-byte record per quantisation group:
//
//   record(rb, g) = 128 dwords  weights:  dword j*16 + l = row rb*16+l, bytes g*32 + 4j
//                    16 dwords  meta:     dword l       = f16 scale | zero point << 16
//
// so intel_sub_group_block_read8 + intel_sub_group_block_read hand every lane
// its own row's group with fully contiguous memory traffic. Records of a row
// block are consecutive in g, then blocks follow each other.
//
// Why: with the stock layout each lane gathers its row's 32 bytes, scale and
// zero point from 16 different rows; on B60 that capped the MoE GEMV at
// ~200 GB/s, and the scale/zp gathers alone cost 25% of its time.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mk {

constexpr int kRepackRows = 16;
constexpr int kRepackGroupBytes = 32;                      // 64 int4 weights
constexpr int kRepackRecordDwords = 128 + 16;
constexpr int kRepackRecordBytes = kRepackRecordDwords * 4;

inline size_t repacked_bytes(size_t N, size_t K) {
    return N / kRepackRows * (K / 64) * kRepackRecordBytes;
}

// Repacks row blocks [rb_begin, rb_end). zp is u8 per (row, group), or u4
// packed in row-major (row, group) order when zp_u4 is set.
inline void repack_row_blocks(const uint8_t* w, const uint16_t* scale, const uint8_t* zp, bool zp_u4, size_t K,
                              size_t rb_begin, size_t rb_end, uint8_t* out) {
    const size_t G = K / 64;
    const size_t row_bytes = K / 2;
    for (size_t rb = rb_begin; rb < rb_end; ++rb) {
        for (size_t g = 0; g < G; ++g) {
            uint32_t* rec = reinterpret_cast<uint32_t*>(out + (rb * G + g) * kRepackRecordBytes);
            for (int l = 0; l < kRepackRows; ++l) {
                const size_t row = rb * kRepackRows + l;
                const uint8_t* src = w + row * row_bytes + g * kRepackGroupBytes;
                for (int j = 0; j < 8; ++j)
                    std::memcpy(&rec[j * 16 + l], src + 4 * j, 4);
                const size_t gi = row * G + g;
                const uint32_t z = zp_u4 ? ((gi & 1) ? (zp[gi >> 1] >> 4) : (zp[gi >> 1] & 15)) : zp[gi];
                rec[128 + l] = static_cast<uint32_t>(scale[gi]) | (z << 16);
            }
        }
    }
}

}  // namespace mk
