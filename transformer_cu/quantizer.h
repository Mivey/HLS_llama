#ifndef MARK_QUANT
#define MARK_QUANT
#include "mha.h"
#include "mha_forward.h"
#include <cstdio>

void quantizer_kernel(hls::stream<my_float_t>  &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens, const int N_DIM);
#endif
