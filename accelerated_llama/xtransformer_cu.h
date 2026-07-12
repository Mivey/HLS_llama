// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XTRANSFORMER_CU_H
#define XTRANSFORMER_CU_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xtransformer_cu_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XTransformer_cu_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XTransformer_cu;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XTransformer_cu_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XTransformer_cu_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XTransformer_cu_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XTransformer_cu_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XTransformer_cu_Initialize(XTransformer_cu *InstancePtr, UINTPTR BaseAddress);
XTransformer_cu_Config* XTransformer_cu_LookupConfig(UINTPTR BaseAddress);
#else
int XTransformer_cu_Initialize(XTransformer_cu *InstancePtr, u16 DeviceId);
XTransformer_cu_Config* XTransformer_cu_LookupConfig(u16 DeviceId);
#endif
int XTransformer_cu_CfgInitialize(XTransformer_cu *InstancePtr, XTransformer_cu_Config *ConfigPtr);
#else
int XTransformer_cu_Initialize(XTransformer_cu *InstancePtr, const char* InstanceName);
int XTransformer_cu_Release(XTransformer_cu *InstancePtr);
#endif

void XTransformer_cu_Start(XTransformer_cu *InstancePtr);
u32 XTransformer_cu_IsDone(XTransformer_cu *InstancePtr);
u32 XTransformer_cu_IsIdle(XTransformer_cu *InstancePtr);
u32 XTransformer_cu_IsReady(XTransformer_cu *InstancePtr);
void XTransformer_cu_Continue(XTransformer_cu *InstancePtr);
void XTransformer_cu_EnableAutoRestart(XTransformer_cu *InstancePtr);
void XTransformer_cu_DisableAutoRestart(XTransformer_cu *InstancePtr);

void XTransformer_cu_Set_tokens(XTransformer_cu *InstancePtr, u64 Data);
u64 XTransformer_cu_Get_tokens(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_w_sf_0(XTransformer_cu *InstancePtr, u64 Data);
u64 XTransformer_cu_Get_w_sf_0(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_w_0(XTransformer_cu *InstancePtr, u64 Data);
u64 XTransformer_cu_Get_w_0(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_w_sf_1(XTransformer_cu *InstancePtr, u64 Data);
u64 XTransformer_cu_Get_w_sf_1(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_w_1(XTransformer_cu *InstancePtr, u64 Data);
u64 XTransformer_cu_Get_w_1(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_weights(XTransformer_cu *InstancePtr, u64 Data);
u64 XTransformer_cu_Get_weights(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_key_cache(XTransformer_cu *InstancePtr, u64 Data);
u64 XTransformer_cu_Get_key_cache(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_value_cache(XTransformer_cu *InstancePtr, u64 Data);
u64 XTransformer_cu_Get_value_cache(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_POS_r(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_POS_r(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_QKV_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_QKV_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_QKV_sf_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_QKV_sf_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_Out_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_Out_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_Out_sf_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_Out_sf_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_FF_w1w3_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_FF_w1w3_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_FF_w1w3_sf_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_FF_w1w3_sf_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_FF_w2_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_FF_w2_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_FF_w2_sf_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_FF_w2_sf_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_Embed_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_Embed_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_Embed_sf_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_Embed_sf_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_rms_att_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_rms_att_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_rms_ffn_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_rms_ffn_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_rms_final_W(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_rms_final_W(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_temperature(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_temperature(XTransformer_cu *InstancePtr);
void XTransformer_cu_Set_coin(XTransformer_cu *InstancePtr, u32 Data);
u32 XTransformer_cu_Get_coin(XTransformer_cu *InstancePtr);
u32 XTransformer_cu_Get_pick(XTransformer_cu *InstancePtr);

void XTransformer_cu_InterruptGlobalEnable(XTransformer_cu *InstancePtr);
void XTransformer_cu_InterruptGlobalDisable(XTransformer_cu *InstancePtr);
void XTransformer_cu_InterruptEnable(XTransformer_cu *InstancePtr, u32 Mask);
void XTransformer_cu_InterruptDisable(XTransformer_cu *InstancePtr, u32 Mask);
void XTransformer_cu_InterruptClear(XTransformer_cu *InstancePtr, u32 Mask);
u32 XTransformer_cu_InterruptGetEnabled(XTransformer_cu *InstancePtr);
u32 XTransformer_cu_InterruptGetStatus(XTransformer_cu *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
