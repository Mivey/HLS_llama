// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 4  - ap_continue (Read/Write/SC)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0 - ap_done (Read/TOW)
//        bit 1 - ap_ready (Read/TOW)
//        others - reserved
// 0x10 : Data signal of tokens
//        bit 31~0 - tokens[31:0] (Read/Write)
// 0x14 : Data signal of tokens
//        bit 31~0 - tokens[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of w_sf_0
//        bit 31~0 - w_sf_0[31:0] (Read/Write)
// 0x20 : Data signal of w_sf_0
//        bit 31~0 - w_sf_0[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of w_0
//        bit 31~0 - w_0[31:0] (Read/Write)
// 0x2c : Data signal of w_0
//        bit 31~0 - w_0[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of w_sf_1
//        bit 31~0 - w_sf_1[31:0] (Read/Write)
// 0x38 : Data signal of w_sf_1
//        bit 31~0 - w_sf_1[63:32] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of w_1
//        bit 31~0 - w_1[31:0] (Read/Write)
// 0x44 : Data signal of w_1
//        bit 31~0 - w_1[63:32] (Read/Write)
// 0x48 : reserved
// 0x4c : Data signal of weights
//        bit 31~0 - weights[31:0] (Read/Write)
// 0x50 : Data signal of weights
//        bit 31~0 - weights[63:32] (Read/Write)
// 0x54 : reserved
// 0x58 : Data signal of key_cache
//        bit 31~0 - key_cache[31:0] (Read/Write)
// 0x5c : Data signal of key_cache
//        bit 31~0 - key_cache[63:32] (Read/Write)
// 0x60 : reserved
// 0x64 : Data signal of value_cache
//        bit 31~0 - value_cache[31:0] (Read/Write)
// 0x68 : Data signal of value_cache
//        bit 31~0 - value_cache[63:32] (Read/Write)
// 0x6c : reserved
// 0x70 : Data signal of POS_r
//        bit 31~0 - POS_r[31:0] (Read/Write)
// 0x74 : reserved
// 0x78 : Data signal of QKV_W
//        bit 31~0 - QKV_W[31:0] (Read/Write)
// 0x7c : reserved
// 0x80 : Data signal of QKV_sf_W
//        bit 31~0 - QKV_sf_W[31:0] (Read/Write)
// 0x84 : reserved
// 0x88 : Data signal of Out_W
//        bit 31~0 - Out_W[31:0] (Read/Write)
// 0x8c : reserved
// 0x90 : Data signal of Out_sf_W
//        bit 31~0 - Out_sf_W[31:0] (Read/Write)
// 0x94 : reserved
// 0x98 : Data signal of FF_w1w3_W
//        bit 31~0 - FF_w1w3_W[31:0] (Read/Write)
// 0x9c : reserved
// 0xa0 : Data signal of FF_w1w3_sf_W
//        bit 31~0 - FF_w1w3_sf_W[31:0] (Read/Write)
// 0xa4 : reserved
// 0xa8 : Data signal of FF_w2_W
//        bit 31~0 - FF_w2_W[31:0] (Read/Write)
// 0xac : reserved
// 0xb0 : Data signal of FF_w2_sf_W
//        bit 31~0 - FF_w2_sf_W[31:0] (Read/Write)
// 0xb4 : reserved
// 0xb8 : Data signal of Embed_W
//        bit 31~0 - Embed_W[31:0] (Read/Write)
// 0xbc : reserved
// 0xc0 : Data signal of Embed_sf_W
//        bit 31~0 - Embed_sf_W[31:0] (Read/Write)
// 0xc4 : reserved
// 0xc8 : Data signal of rms_att_W
//        bit 31~0 - rms_att_W[31:0] (Read/Write)
// 0xcc : reserved
// 0xd0 : Data signal of rms_ffn_W
//        bit 31~0 - rms_ffn_W[31:0] (Read/Write)
// 0xd4 : reserved
// 0xd8 : Data signal of rms_final_W
//        bit 31~0 - rms_final_W[31:0] (Read/Write)
// 0xdc : reserved
// 0xe0 : Data signal of temperature
//        bit 31~0 - temperature[31:0] (Read/Write)
// 0xe4 : reserved
// 0xe8 : Data signal of coin
//        bit 31~0 - coin[31:0] (Read/Write)
// 0xec : reserved
// 0xf0 : Data signal of pick
//        bit 31~0 - pick[31:0] (Read)
// 0xf4 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL           0x00
#define XTRANSFORMER_CU_CONTROL_ADDR_GIE               0x04
#define XTRANSFORMER_CU_CONTROL_ADDR_IER               0x08
#define XTRANSFORMER_CU_CONTROL_ADDR_ISR               0x0c
#define XTRANSFORMER_CU_CONTROL_ADDR_TOKENS_DATA       0x10
#define XTRANSFORMER_CU_CONTROL_BITS_TOKENS_DATA       64
#define XTRANSFORMER_CU_CONTROL_ADDR_W_SF_0_DATA       0x1c
#define XTRANSFORMER_CU_CONTROL_BITS_W_SF_0_DATA       64
#define XTRANSFORMER_CU_CONTROL_ADDR_W_0_DATA          0x28
#define XTRANSFORMER_CU_CONTROL_BITS_W_0_DATA          64
#define XTRANSFORMER_CU_CONTROL_ADDR_W_SF_1_DATA       0x34
#define XTRANSFORMER_CU_CONTROL_BITS_W_SF_1_DATA       64
#define XTRANSFORMER_CU_CONTROL_ADDR_W_1_DATA          0x40
#define XTRANSFORMER_CU_CONTROL_BITS_W_1_DATA          64
#define XTRANSFORMER_CU_CONTROL_ADDR_WEIGHTS_DATA      0x4c
#define XTRANSFORMER_CU_CONTROL_BITS_WEIGHTS_DATA      64
#define XTRANSFORMER_CU_CONTROL_ADDR_KEY_CACHE_DATA    0x58
#define XTRANSFORMER_CU_CONTROL_BITS_KEY_CACHE_DATA    64
#define XTRANSFORMER_CU_CONTROL_ADDR_VALUE_CACHE_DATA  0x64
#define XTRANSFORMER_CU_CONTROL_BITS_VALUE_CACHE_DATA  64
#define XTRANSFORMER_CU_CONTROL_ADDR_POS_R_DATA        0x70
#define XTRANSFORMER_CU_CONTROL_BITS_POS_R_DATA        32
#define XTRANSFORMER_CU_CONTROL_ADDR_QKV_W_DATA        0x78
#define XTRANSFORMER_CU_CONTROL_BITS_QKV_W_DATA        32
#define XTRANSFORMER_CU_CONTROL_ADDR_QKV_SF_W_DATA     0x80
#define XTRANSFORMER_CU_CONTROL_BITS_QKV_SF_W_DATA     32
#define XTRANSFORMER_CU_CONTROL_ADDR_OUT_W_DATA        0x88
#define XTRANSFORMER_CU_CONTROL_BITS_OUT_W_DATA        32
#define XTRANSFORMER_CU_CONTROL_ADDR_OUT_SF_W_DATA     0x90
#define XTRANSFORMER_CU_CONTROL_BITS_OUT_SF_W_DATA     32
#define XTRANSFORMER_CU_CONTROL_ADDR_FF_W1W3_W_DATA    0x98
#define XTRANSFORMER_CU_CONTROL_BITS_FF_W1W3_W_DATA    32
#define XTRANSFORMER_CU_CONTROL_ADDR_FF_W1W3_SF_W_DATA 0xa0
#define XTRANSFORMER_CU_CONTROL_BITS_FF_W1W3_SF_W_DATA 32
#define XTRANSFORMER_CU_CONTROL_ADDR_FF_W2_W_DATA      0xa8
#define XTRANSFORMER_CU_CONTROL_BITS_FF_W2_W_DATA      32
#define XTRANSFORMER_CU_CONTROL_ADDR_FF_W2_SF_W_DATA   0xb0
#define XTRANSFORMER_CU_CONTROL_BITS_FF_W2_SF_W_DATA   32
#define XTRANSFORMER_CU_CONTROL_ADDR_EMBED_W_DATA      0xb8
#define XTRANSFORMER_CU_CONTROL_BITS_EMBED_W_DATA      32
#define XTRANSFORMER_CU_CONTROL_ADDR_EMBED_SF_W_DATA   0xc0
#define XTRANSFORMER_CU_CONTROL_BITS_EMBED_SF_W_DATA   32
#define XTRANSFORMER_CU_CONTROL_ADDR_RMS_ATT_W_DATA    0xc8
#define XTRANSFORMER_CU_CONTROL_BITS_RMS_ATT_W_DATA    32
#define XTRANSFORMER_CU_CONTROL_ADDR_RMS_FFN_W_DATA    0xd0
#define XTRANSFORMER_CU_CONTROL_BITS_RMS_FFN_W_DATA    32
#define XTRANSFORMER_CU_CONTROL_ADDR_RMS_FINAL_W_DATA  0xd8
#define XTRANSFORMER_CU_CONTROL_BITS_RMS_FINAL_W_DATA  32
#define XTRANSFORMER_CU_CONTROL_ADDR_TEMPERATURE_DATA  0xe0
#define XTRANSFORMER_CU_CONTROL_BITS_TEMPERATURE_DATA  32
#define XTRANSFORMER_CU_CONTROL_ADDR_COIN_DATA         0xe8
#define XTRANSFORMER_CU_CONTROL_BITS_COIN_DATA         32
#define XTRANSFORMER_CU_CONTROL_ADDR_PICK_DATA         0xf0
#define XTRANSFORMER_CU_CONTROL_BITS_PICK_DATA         32

