// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xtransformer_cu.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XTransformer_cu_CfgInitialize(XTransformer_cu *InstancePtr, XTransformer_cu_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XTransformer_cu_Start(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL) & 0x80;
    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XTransformer_cu_IsDone(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XTransformer_cu_IsIdle(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XTransformer_cu_IsReady(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XTransformer_cu_Continue(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL) & 0x80;
    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL, Data | 0x10);
}

void XTransformer_cu_EnableAutoRestart(XTransformer_cu *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XTransformer_cu_DisableAutoRestart(XTransformer_cu *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL, 0);
}

void XTransformer_cu_Set_tokens(XTransformer_cu *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_TOKENS_DATA, (u32)(Data));
    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_TOKENS_DATA + 4, (u32)(Data >> 32));
}

u64 XTransformer_cu_Get_tokens(XTransformer_cu *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_TOKENS_DATA);
    Data += (u64)XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_TOKENS_DATA + 4) << 32;
    return Data;
}

void XTransformer_cu_Set_w_0(XTransformer_cu *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_W_0_DATA, (u32)(Data));
    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_W_0_DATA + 4, (u32)(Data >> 32));
}

u64 XTransformer_cu_Get_w_0(XTransformer_cu *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_W_0_DATA);
    Data += (u64)XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_W_0_DATA + 4) << 32;
    return Data;
}

void XTransformer_cu_Set_w_1(XTransformer_cu *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_W_1_DATA, (u32)(Data));
    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_W_1_DATA + 4, (u32)(Data >> 32));
}

u64 XTransformer_cu_Get_w_1(XTransformer_cu *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_W_1_DATA);
    Data += (u64)XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_W_1_DATA + 4) << 32;
    return Data;
}

void XTransformer_cu_Set_weights(XTransformer_cu *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_WEIGHTS_DATA, (u32)(Data));
    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_WEIGHTS_DATA + 4, (u32)(Data >> 32));
}

u64 XTransformer_cu_Get_weights(XTransformer_cu *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_WEIGHTS_DATA);
    Data += (u64)XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_WEIGHTS_DATA + 4) << 32;
    return Data;
}

void XTransformer_cu_Set_key_cache(XTransformer_cu *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_KEY_CACHE_DATA, (u32)(Data));
    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_KEY_CACHE_DATA + 4, (u32)(Data >> 32));
}

u64 XTransformer_cu_Get_key_cache(XTransformer_cu *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_KEY_CACHE_DATA);
    Data += (u64)XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_KEY_CACHE_DATA + 4) << 32;
    return Data;
}

void XTransformer_cu_Set_value_cache(XTransformer_cu *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_VALUE_CACHE_DATA, (u32)(Data));
    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_VALUE_CACHE_DATA + 4, (u32)(Data >> 32));
}

u64 XTransformer_cu_Get_value_cache(XTransformer_cu *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_VALUE_CACHE_DATA);
    Data += (u64)XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_VALUE_CACHE_DATA + 4) << 32;
    return Data;
}

void XTransformer_cu_Set_POS_r(XTransformer_cu *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_POS_R_DATA, Data);
}

u32 XTransformer_cu_Get_POS_r(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_POS_R_DATA);
    return Data;
}

void XTransformer_cu_Set_rms_att_W(XTransformer_cu *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_RMS_ATT_W_DATA, Data);
}

u32 XTransformer_cu_Get_rms_att_W(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_RMS_ATT_W_DATA);
    return Data;
}

void XTransformer_cu_Set_rms_ffn_W(XTransformer_cu *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_RMS_FFN_W_DATA, Data);
}

u32 XTransformer_cu_Get_rms_ffn_W(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_RMS_FFN_W_DATA);
    return Data;
}

void XTransformer_cu_Set_rms_final_W(XTransformer_cu *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_RMS_FINAL_W_DATA, Data);
}

u32 XTransformer_cu_Get_rms_final_W(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_RMS_FINAL_W_DATA);
    return Data;
}

void XTransformer_cu_Set_curr_token(XTransformer_cu *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_CURR_TOKEN_DATA, (u32)(Data));
    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_CURR_TOKEN_DATA + 4, (u32)(Data >> 32));
}

u64 XTransformer_cu_Get_curr_token(XTransformer_cu *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_CURR_TOKEN_DATA);
    Data += (u64)XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_CURR_TOKEN_DATA + 4) << 32;
    return Data;
}

void XTransformer_cu_Set_temperature(XTransformer_cu *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_TEMPERATURE_DATA, Data);
}

u32 XTransformer_cu_Get_temperature(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_TEMPERATURE_DATA);
    return Data;
}

void XTransformer_cu_Set_coin(XTransformer_cu *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_COIN_DATA, Data);
}

u32 XTransformer_cu_Get_coin(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_COIN_DATA);
    return Data;
}

void XTransformer_cu_Set_init_rms_flag(XTransformer_cu *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_INIT_RMS_FLAG_DATA, Data);
}

u32 XTransformer_cu_Get_init_rms_flag(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_INIT_RMS_FLAG_DATA);
    return Data;
}

void XTransformer_cu_Set_prefill_flag(XTransformer_cu *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_PREFILL_FLAG_DATA, Data);
}

u32 XTransformer_cu_Get_prefill_flag(XTransformer_cu *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_PREFILL_FLAG_DATA);
    return Data;
}

void XTransformer_cu_InterruptGlobalEnable(XTransformer_cu *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_GIE, 1);
}

void XTransformer_cu_InterruptGlobalDisable(XTransformer_cu *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_GIE, 0);
}

void XTransformer_cu_InterruptEnable(XTransformer_cu *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_IER);
    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_IER, Register | Mask);
}

void XTransformer_cu_InterruptDisable(XTransformer_cu *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_IER);
    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_IER, Register & (~Mask));
}

void XTransformer_cu_InterruptClear(XTransformer_cu *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTransformer_cu_WriteReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_ISR, Mask);
}

u32 XTransformer_cu_InterruptGetEnabled(XTransformer_cu *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_IER);
}

u32 XTransformer_cu_InterruptGetStatus(XTransformer_cu *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XTransformer_cu_ReadReg(InstancePtr->Control_BaseAddress, XTRANSFORMER_CU_CONTROL_ADDR_ISR);
}

