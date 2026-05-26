#ifndef MARK_RMS
#define MARK_RMS
#include "mha_forward.h"

// void rmsnorm_kernel(s_fdata_v_t &s_tokens_out, fdata_v_t *diff, fdata_v_t *weights, const int CURR_LAYER, const int INIT, const int offset);
void rmsnorm_kernel(s_idata_v_t &s_w, hls::stream<my_float_t> &s_sf, fdata_v_t *diff, fdata_v_t *weights, const int CURR_LAYER, const int INIT, const int offset);

#endif
