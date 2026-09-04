// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// control
// 0x000 : Control signals
//         bit 0  - ap_start (Read/Write/COH)
//         bit 1  - ap_done (Read)
//         bit 2  - ap_idle (Read)
//         bit 3  - ap_ready (Read/COR)
//         bit 4  - ap_continue (Read/Write/SC)
//         bit 7  - auto_restart (Read/Write)
//         bit 9  - interrupt (Read)
//         others - reserved
// 0x004 : Global Interrupt Enable Register
//         bit 0  - Global Interrupt Enable (Read/Write)
//         others - reserved
// 0x008 : IP Interrupt Enable Register (Read/Write)
//         bit 0 - enable ap_done interrupt (Read/Write)
//         bit 1 - enable ap_ready interrupt (Read/Write)
//         others - reserved
// 0x00c : IP Interrupt Status Register (Read/TOW)
//         bit 0 - ap_done (Read/TOW)
//         bit 1 - ap_ready (Read/TOW)
//         others - reserved
// 0x010 : Data signal of tokens
//         bit 31~0 - tokens[31:0] (Read/Write)
// 0x014 : Data signal of tokens
//         bit 31~0 - tokens[63:32] (Read/Write)
// 0x018 : reserved
// 0x01c : Data signal of w_sf_0
//         bit 31~0 - w_sf_0[31:0] (Read/Write)
// 0x020 : Data signal of w_sf_0
//         bit 31~0 - w_sf_0[63:32] (Read/Write)
// 0x024 : reserved
// 0x028 : Data signal of w_0
//         bit 31~0 - w_0[31:0] (Read/Write)
// 0x02c : Data signal of w_0
//         bit 31~0 - w_0[63:32] (Read/Write)
// 0x030 : reserved
// 0x034 : Data signal of w_sf_1
//         bit 31~0 - w_sf_1[31:0] (Read/Write)
// 0x038 : Data signal of w_sf_1
//         bit 31~0 - w_sf_1[63:32] (Read/Write)
// 0x03c : reserved
// 0x040 : Data signal of w_1
//         bit 31~0 - w_1[31:0] (Read/Write)
// 0x044 : Data signal of w_1
//         bit 31~0 - w_1[63:32] (Read/Write)
// 0x048 : reserved
// 0x04c : Data signal of weights
//         bit 31~0 - weights[31:0] (Read/Write)
// 0x050 : Data signal of weights
//         bit 31~0 - weights[63:32] (Read/Write)
// 0x054 : reserved
// 0x058 : Data signal of key_cache
//         bit 31~0 - key_cache[31:0] (Read/Write)
// 0x05c : Data signal of key_cache
//         bit 31~0 - key_cache[63:32] (Read/Write)
// 0x060 : reserved
// 0x064 : Data signal of value_cache
//         bit 31~0 - value_cache[31:0] (Read/Write)
// 0x068 : Data signal of value_cache
//         bit 31~0 - value_cache[63:32] (Read/Write)
// 0x06c : reserved
// 0x070 : Data signal of POS_r
//         bit 31~0 - POS_r[31:0] (Read/Write)
// 0x074 : reserved
// 0x078 : Data signal of QKV_W
//         bit 31~0 - QKV_W[31:0] (Read/Write)
// 0x07c : reserved
// 0x080 : Data signal of QKV_sf_W
//         bit 31~0 - QKV_sf_W[31:0] (Read/Write)
// 0x084 : reserved
// 0x088 : Data signal of Out_W
//         bit 31~0 - Out_W[31:0] (Read/Write)
// 0x08c : reserved
// 0x090 : Data signal of Out_sf_W
//         bit 31~0 - Out_sf_W[31:0] (Read/Write)
// 0x094 : reserved
// 0x098 : Data signal of FF_w1w3_W
//         bit 31~0 - FF_w1w3_W[31:0] (Read/Write)
// 0x09c : reserved
// 0x0a0 : Data signal of FF_w1w3_sf_W
//         bit 31~0 - FF_w1w3_sf_W[31:0] (Read/Write)
// 0x0a4 : reserved
// 0x0a8 : Data signal of FF_w2_W
//         bit 31~0 - FF_w2_W[31:0] (Read/Write)
// 0x0ac : reserved
// 0x0b0 : Data signal of FF_w2_sf_W
//         bit 31~0 - FF_w2_sf_W[31:0] (Read/Write)
// 0x0b4 : reserved
// 0x0b8 : Data signal of Embed_W
//         bit 31~0 - Embed_W[31:0] (Read/Write)
// 0x0bc : reserved
// 0x0c0 : Data signal of Embed_sf_W
//         bit 31~0 - Embed_sf_W[31:0] (Read/Write)
// 0x0c4 : reserved
// 0x0c8 : Data signal of rms_att_W
//         bit 31~0 - rms_att_W[31:0] (Read/Write)
// 0x0cc : reserved
// 0x0d0 : Data signal of rms_ffn_W
//         bit 31~0 - rms_ffn_W[31:0] (Read/Write)
// 0x0d4 : reserved
// 0x0d8 : Data signal of rms_final_W
//         bit 31~0 - rms_final_W[31:0] (Read/Write)
// 0x0dc : reserved
// 0x0e0 : Data signal of curr_token
//         bit 31~0 - curr_token[31:0] (Read/Write)
// 0x0e4 : Data signal of curr_token
//         bit 31~0 - curr_token[63:32] (Read/Write)
// 0x0e8 : reserved
// 0x0ec : Data signal of temperature
//         bit 31~0 - temperature[31:0] (Read/Write)
// 0x0f0 : reserved
// 0x0f4 : Data signal of coin
//         bit 31~0 - coin[31:0] (Read/Write)
// 0x0f8 : reserved
// 0x0fc : Data signal of init_rms_flag
//         bit 0  - init_rms_flag[0] (Read/Write)
//         others - reserved
// 0x100 : reserved
// 0x104 : Data signal of prefill_flag
//         bit 0  - prefill_flag[0] (Read/Write)
//         others - reserved
// 0x108 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL            0x000
#define XTRANSFORMER_CU_CONTROL_ADDR_GIE                0x004
#define XTRANSFORMER_CU_CONTROL_ADDR_IER                0x008
#define XTRANSFORMER_CU_CONTROL_ADDR_ISR                0x00c
#define XTRANSFORMER_CU_CONTROL_ADDR_TOKENS_DATA        0x010
#define XTRANSFORMER_CU_CONTROL_BITS_TOKENS_DATA        64
#define XTRANSFORMER_CU_CONTROL_ADDR_W_SF_0_DATA        0x01c
#define XTRANSFORMER_CU_CONTROL_BITS_W_SF_0_DATA        64
#define XTRANSFORMER_CU_CONTROL_ADDR_W_0_DATA           0x028
#define XTRANSFORMER_CU_CONTROL_BITS_W_0_DATA           64
#define XTRANSFORMER_CU_CONTROL_ADDR_W_SF_1_DATA        0x034
#define XTRANSFORMER_CU_CONTROL_BITS_W_SF_1_DATA        64
#define XTRANSFORMER_CU_CONTROL_ADDR_W_1_DATA           0x040
#define XTRANSFORMER_CU_CONTROL_BITS_W_1_DATA           64
#define XTRANSFORMER_CU_CONTROL_ADDR_WEIGHTS_DATA       0x04c
#define XTRANSFORMER_CU_CONTROL_BITS_WEIGHTS_DATA       64
#define XTRANSFORMER_CU_CONTROL_ADDR_KEY_CACHE_DATA     0x058
#define XTRANSFORMER_CU_CONTROL_BITS_KEY_CACHE_DATA     64
#define XTRANSFORMER_CU_CONTROL_ADDR_VALUE_CACHE_DATA   0x064
#define XTRANSFORMER_CU_CONTROL_BITS_VALUE_CACHE_DATA   64
#define XTRANSFORMER_CU_CONTROL_ADDR_POS_R_DATA         0x070
#define XTRANSFORMER_CU_CONTROL_BITS_POS_R_DATA         32
#define XTRANSFORMER_CU_CONTROL_ADDR_QKV_W_DATA         0x078
#define XTRANSFORMER_CU_CONTROL_BITS_QKV_W_DATA         32
#define XTRANSFORMER_CU_CONTROL_ADDR_QKV_SF_W_DATA      0x080
#define XTRANSFORMER_CU_CONTROL_BITS_QKV_SF_W_DATA      32
#define XTRANSFORMER_CU_CONTROL_ADDR_OUT_W_DATA         0x088
#define XTRANSFORMER_CU_CONTROL_BITS_OUT_W_DATA         32
#define XTRANSFORMER_CU_CONTROL_ADDR_OUT_SF_W_DATA      0x090
#define XTRANSFORMER_CU_CONTROL_BITS_OUT_SF_W_DATA      32
#define XTRANSFORMER_CU_CONTROL_ADDR_FF_W1W3_W_DATA     0x098
#define XTRANSFORMER_CU_CONTROL_BITS_FF_W1W3_W_DATA     32
#define XTRANSFORMER_CU_CONTROL_ADDR_FF_W1W3_SF_W_DATA  0x0a0
#define XTRANSFORMER_CU_CONTROL_BITS_FF_W1W3_SF_W_DATA  32
#define XTRANSFORMER_CU_CONTROL_ADDR_FF_W2_W_DATA       0x0a8
#define XTRANSFORMER_CU_CONTROL_BITS_FF_W2_W_DATA       32
#define XTRANSFORMER_CU_CONTROL_ADDR_FF_W2_SF_W_DATA    0x0b0
#define XTRANSFORMER_CU_CONTROL_BITS_FF_W2_SF_W_DATA    32
#define XTRANSFORMER_CU_CONTROL_ADDR_EMBED_W_DATA       0x0b8
#define XTRANSFORMER_CU_CONTROL_BITS_EMBED_W_DATA       32
#define XTRANSFORMER_CU_CONTROL_ADDR_EMBED_SF_W_DATA    0x0c0
#define XTRANSFORMER_CU_CONTROL_BITS_EMBED_SF_W_DATA    32
#define XTRANSFORMER_CU_CONTROL_ADDR_RMS_ATT_W_DATA     0x0c8
#define XTRANSFORMER_CU_CONTROL_BITS_RMS_ATT_W_DATA     32
#define XTRANSFORMER_CU_CONTROL_ADDR_RMS_FFN_W_DATA     0x0d0
#define XTRANSFORMER_CU_CONTROL_BITS_RMS_FFN_W_DATA     32
#define XTRANSFORMER_CU_CONTROL_ADDR_RMS_FINAL_W_DATA   0x0d8
#define XTRANSFORMER_CU_CONTROL_BITS_RMS_FINAL_W_DATA   32
#define XTRANSFORMER_CU_CONTROL_ADDR_CURR_TOKEN_DATA    0x0e0
#define XTRANSFORMER_CU_CONTROL_BITS_CURR_TOKEN_DATA    64
#define XTRANSFORMER_CU_CONTROL_ADDR_TEMPERATURE_DATA   0x0ec
#define XTRANSFORMER_CU_CONTROL_BITS_TEMPERATURE_DATA   32
#define XTRANSFORMER_CU_CONTROL_ADDR_COIN_DATA          0x0f4
#define XTRANSFORMER_CU_CONTROL_BITS_COIN_DATA          32
#define XTRANSFORMER_CU_CONTROL_ADDR_INIT_RMS_FLAG_DATA 0x0fc
#define XTRANSFORMER_CU_CONTROL_BITS_INIT_RMS_FLAG_DATA 1
#define XTRANSFORMER_CU_CONTROL_ADDR_PREFILL_FLAG_DATA  0x104
#define XTRANSFORMER_CU_CONTROL_BITS_PREFILL_FLAG_DATA  1

