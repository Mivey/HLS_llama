#ifndef MARK_ROPE
#define MARK_ROPE
#include "../forward.h"
#include <cstddef>

// template<int HEAD>
// void init_freq_arr(float arr[HEAD]){
//   for (int i = 0; i < HEAD; i++) {
//   arr[i] = 1.0f / hls::powf(10000.0f, ((i) / (float) MODEL_HEAD_SIZE));
//   }
// }

// template<typename T, size_t N, int N_DIM = MODEL_ELEMENTS>
// void rope_kernel (hls::stream<hls::vector<T, N>> &o, hls::stream<hls::vector<T, N>> &in, const int POS){
//   float arr[MODEL_HEAD_SIZE];
//   init_freq_arr<MODEL_HEAD_SIZE>(arr);
//   ROPE_MAIN:
//   for (int i = 0; i < (N_DIM / N); i++) {
//     #pragma HLS loop_flatten 
//  // increment by number of element in fdata_v_t
  
//   int k = i * N;
//     hls::vector<T, N> tmp = in.read();
//     hls::vector<T, N> tmp_o;
//     head_dim_unroll_loop:
//     for (int j = 0 ; j < (N / 2); j++) {
//       #pragma HLS PIPELINE
//       #pragma HLS UNROLL factor = 2
//       int head_dim = (k + j * 2) % MODEL_HEAD_SIZE;
//       float freq =  arr[head_dim]; /*1.0f / hls::powf(10000.0f, (float)head_dim/HEAD_SIZE);*/ 
//       float val = POS * freq;
//       float fcr;
//       float fci;
//       hls::sincosf(val, &fci, &fcr);
//       float v0 = tmp[j * 2 + 0];
//       float v1 = tmp[j * 2 + 1];
//       tmp_o[j * 2 + 0] = v0 * fcr - v1 * fci;
//       tmp_o[j * 2 + 1] = v0 * fci + v1 * fcr;
//     }
//     o.write(tmp_o);
//   }
// }
#endif