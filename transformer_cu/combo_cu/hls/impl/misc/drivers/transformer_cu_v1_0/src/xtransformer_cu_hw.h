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
// 0x1c : Data signal of w_0
//        bit 31~0 - w_0[31:0] (Read/Write)
// 0x20 : Data signal of w_0
//        bit 31~0 - w_0[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of w_1
//        bit 31~0 - w_1[31:0] (Read/Write)
// 0x2c : Data signal of w_1
//        bit 31~0 - w_1[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of weights
//        bit 31~0 - weights[31:0] (Read/Write)
// 0x38 : Data signal of weights
//        bit 31~0 - weights[63:32] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of key_cache
//        bit 31~0 - key_cache[31:0] (Read/Write)
// 0x44 : Data signal of key_cache
//        bit 31~0 - key_cache[63:32] (Read/Write)
// 0x48 : reserved
// 0x4c : Data signal of value_cache
//        bit 31~0 - value_cache[31:0] (Read/Write)
// 0x50 : Data signal of value_cache
//        bit 31~0 - value_cache[63:32] (Read/Write)
// 0x54 : reserved
// 0x58 : Data signal of POS_r
//        bit 31~0 - POS_r[31:0] (Read/Write)
// 0x5c : reserved
// 0x60 : Data signal of rms_att_W
//        bit 31~0 - rms_att_W[31:0] (Read/Write)
// 0x64 : reserved
// 0x68 : Data signal of rms_ffn_W
//        bit 31~0 - rms_ffn_W[31:0] (Read/Write)
// 0x6c : reserved
// 0x70 : Data signal of rms_final_W
//        bit 31~0 - rms_final_W[31:0] (Read/Write)
// 0x74 : reserved
// 0x78 : Data signal of curr_token
//        bit 31~0 - curr_token[31:0] (Read/Write)
// 0x7c : Data signal of curr_token
//        bit 31~0 - curr_token[63:32] (Read/Write)
// 0x80 : reserved
// 0x84 : Data signal of temperature
//        bit 31~0 - temperature[31:0] (Read/Write)
// 0x88 : reserved
// 0x8c : Data signal of coin
//        bit 31~0 - coin[31:0] (Read/Write)
// 0x90 : reserved
// 0x94 : Data signal of init_rms_flag
//        bit 0  - init_rms_flag[0] (Read/Write)
//        others - reserved
// 0x98 : reserved
// 0x9c : Data signal of prefill_flag
//        bit 0  - prefill_flag[0] (Read/Write)
//        others - reserved
// 0xa0 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL            0x00
#define XTRANSFORMER_CU_CONTROL_ADDR_GIE                0x04
#define XTRANSFORMER_CU_CONTROL_ADDR_IER                0x08
#define XTRANSFORMER_CU_CONTROL_ADDR_ISR                0x0c
#define XTRANSFORMER_CU_CONTROL_ADDR_TOKENS_DATA        0x10
#define XTRANSFORMER_CU_CONTROL_BITS_TOKENS_DATA        64
#define XTRANSFORMER_CU_CONTROL_ADDR_W_0_DATA           0x1c
#define XTRANSFORMER_CU_CONTROL_BITS_W_0_DATA           64
#define XTRANSFORMER_CU_CONTROL_ADDR_W_1_DATA           0x28
#define XTRANSFORMER_CU_CONTROL_BITS_W_1_DATA           64
#define XTRANSFORMER_CU_CONTROL_ADDR_WEIGHTS_DATA       0x34
#define XTRANSFORMER_CU_CONTROL_BITS_WEIGHTS_DATA       64
#define XTRANSFORMER_CU_CONTROL_ADDR_KEY_CACHE_DATA     0x40
#define XTRANSFORMER_CU_CONTROL_BITS_KEY_CACHE_DATA     64
#define XTRANSFORMER_CU_CONTROL_ADDR_VALUE_CACHE_DATA   0x4c
#define XTRANSFORMER_CU_CONTROL_BITS_VALUE_CACHE_DATA   64
#define XTRANSFORMER_CU_CONTROL_ADDR_POS_R_DATA         0x58
#define XTRANSFORMER_CU_CONTROL_BITS_POS_R_DATA         32
#define XTRANSFORMER_CU_CONTROL_ADDR_RMS_ATT_W_DATA     0x60
#define XTRANSFORMER_CU_CONTROL_BITS_RMS_ATT_W_DATA     32
#define XTRANSFORMER_CU_CONTROL_ADDR_RMS_FFN_W_DATA     0x68
#define XTRANSFORMER_CU_CONTROL_BITS_RMS_FFN_W_DATA     32
#define XTRANSFORMER_CU_CONTROL_ADDR_RMS_FINAL_W_DATA   0x70
#define XTRANSFORMER_CU_CONTROL_BITS_RMS_FINAL_W_DATA   32
#define XTRANSFORMER_CU_CONTROL_ADDR_CURR_TOKEN_DATA    0x78
#define XTRANSFORMER_CU_CONTROL_BITS_CURR_TOKEN_DATA    64
#define XTRANSFORMER_CU_CONTROL_ADDR_TEMPERATURE_DATA   0x84
#define XTRANSFORMER_CU_CONTROL_BITS_TEMPERATURE_DATA   32
#define XTRANSFORMER_CU_CONTROL_ADDR_COIN_DATA          0x8c
#define XTRANSFORMER_CU_CONTROL_BITS_COIN_DATA          32
#define XTRANSFORMER_CU_CONTROL_ADDR_INIT_RMS_FLAG_DATA 0x94
#define XTRANSFORMER_CU_CONTROL_BITS_INIT_RMS_FLAG_DATA 1
#define XTRANSFORMER_CU_CONTROL_ADDR_PREFILL_FLAG_DATA  0x9c
#define XTRANSFORMER_CU_CONTROL_BITS_PREFILL_FLAG_DATA  1

