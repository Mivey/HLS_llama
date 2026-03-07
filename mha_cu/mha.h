
#ifndef MARK_MHA_H__
#define MARK_MHA_H__

#include "../forward.h"

// void mha_kernel(mfdata_v_t *tokens, mfdata_v_t *key_cache, mfdata_v_t *value_cache, 
// 	mfdata_v_t *key_cache_in, mfdata_v_t *value_cache_in, const int POS, const int CURR_LAYER);


void mha_kernel(mfdata_v_t *tokens, //6 mha_kernel
                mfdata_v_t *key_cache, 
                mfdata_v_t *value_cache, 
								mfdata_v_t *key_cache_in, mfdata_v_t *value_cache_in,
                const int POS, const int CURR_LAYER);
#endif
