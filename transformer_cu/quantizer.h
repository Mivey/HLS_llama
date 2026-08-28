#ifndef MARK_QUANT
#define MARK_QUANT
#include "mha.h"
#include "mha_forward.h"
#include <cstdio>

void quantizer_kernel(hls::stream<my_float_t>  &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens, const int N_DIM);
void quantizer_kernel(hls::stream<my_float_t>  &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens, const int N_DIM, fdata_v_t *data_out, const int SAVE_ADDR);
void dequantize_kernel(fdata_v_t* internal_token, idata_v_t* tokq, fdata_v_t* toksf, const int curr_token, const int wcls_offset_q, const int wcls_sf);
#endif
