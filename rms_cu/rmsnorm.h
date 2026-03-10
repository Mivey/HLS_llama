#ifndef MARK_RMS
#define MARK_RMS
#include "../forward.h"

// void rmsnorm_kernel(fdata_v_t *tokens, fdata_v_t *stokens, fdata_v_t *weights/*, const int CURR_LAYER*/);
void rmsnorm_kernel(fdata_v_t *output, fdata_v_t *tokens, fdata_v_t *diff, fdata_v_t *weights, const int CURR_LAYER);
#endif
