#ifndef MARK_RMS
#define MARK_RMS
#include "mha_forward.h"

void rmsnorm_kernel(s_fdata_v_t &s_tokens_out, fdata_v_t *diff, fdata_v_t *weights, const int CURR_LAYER, const int INIT, const int offset);

#endif
