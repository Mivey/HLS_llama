
#include "mha_forward.h"

int32_t rtl_code(wide_t a, wide_t b){
  int32_t acc = 0;
  for (int i = 0; i < 64; i++)
  {
    int8_t ax, bz;
    ax = a.range((8 * (i + 1) - 1), (8 * i));
    bz = b.range((8 * (i + 1) - 1), (8 * i));

    acc += ax * bz;
  }
  
return acc;
}