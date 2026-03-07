
#ifndef MARK_MM
#define MARK_MM
#include "../forward.h"
// void matmult_kernel(fdata_v_t *out, fdata_v_t *fl_tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER);
void double_matmult_kernel(fdata_v_t *out, fdata_v_t *fl_tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER, const int W_Off);
// void matmult_kernel(mfdata_v_t *out, mfdata_v_t *fl_tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int READ_OFFSET, const int WRITE_OFFSET);

#endif
