#include "../forward.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

int main(){
	std::cout<<"Begin\n";

	std::ifstream logits_dat("tb_files/199_logits_out.bin", std::ios::binary);
	float random_u32 = 0.999;
	
	if (!logits_dat.is_open()) {
		std::cout<<"File not opened\n";
		exit(EXIT_FAILURE);
	}
	
	std::vector<fdata_v_t> logits_arr(MODEL_TOKENS/SM_FL_ELEM	);
	logits_dat.read(reinterpret_cast<char*>(logits_arr.data()), MODEL_TOKENS * sizeof(my_float_t));
	float temperature = 0.009;
	int pick;

	systolic_sort(logits_arr.data(), &pick, temperature, random_u32);
	std::cout<<"value is "<<pick<<std::endl;
	return 0;
}