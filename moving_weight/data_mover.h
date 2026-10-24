#include "../transformer_cu/mha_forward.h"

void data_mover_cu(s_wide_t &s_w_0, s_wide_t &s_w_1, s_wide_t &s_sf_0, s_wide_t &s_sf_1, const wide_t *w_0, const wide_t *w_1, 
                  const bool prefill_flag)
