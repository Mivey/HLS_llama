#ifndef MARK_SWI
#define MARK_SWI
#include "mha_forward.h"

// void swiglu_kernel(s_fdata_v_t &output, fdata_v_t *w1w3);
void swiglu_kernel(hls::stream<my_float_t> &tok_sf_out, s_idata_v_t &tok_out, fdata_v_t *w1w3);

#endif
