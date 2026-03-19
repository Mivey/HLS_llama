
#ifndef MARK_MM
#define MARK_MM
// #include "../forward.h"
#include "mha_forward.h"
// void GeMV_kernel(fdata_v_t *out, fdata_v_t *fl_tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER, const int W_Off);
void GeMV_kernel(fdata_v_t *out, s_fdata_v_t &tok_sf, s_idata_v_t &tok_q, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER, const int W_Off);
// void GeMV_kernel(fdata_v_t *out, s_fdata_v_t &xb_out, s_fdata_v_t &tok_sf, s_idata_v_t &tok_q, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER, const int W_Off, const int AXI_SEL);

#endif
