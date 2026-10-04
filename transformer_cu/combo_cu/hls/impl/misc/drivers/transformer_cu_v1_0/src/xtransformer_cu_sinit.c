// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xtransformer_cu.h"

extern XTransformer_cu_Config XTransformer_cu_ConfigTable[];

#ifdef SDT
XTransformer_cu_Config *XTransformer_cu_LookupConfig(UINTPTR BaseAddress) {
	XTransformer_cu_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XTransformer_cu_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XTransformer_cu_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XTransformer_cu_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XTransformer_cu_Initialize(XTransformer_cu *InstancePtr, UINTPTR BaseAddress) {
	XTransformer_cu_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XTransformer_cu_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XTransformer_cu_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XTransformer_cu_Config *XTransformer_cu_LookupConfig(u16 DeviceId) {
	XTransformer_cu_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XTRANSFORMER_CU_NUM_INSTANCES; Index++) {
		if (XTransformer_cu_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XTransformer_cu_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XTransformer_cu_Initialize(XTransformer_cu *InstancePtr, u16 DeviceId) {
	XTransformer_cu_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XTransformer_cu_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XTransformer_cu_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

