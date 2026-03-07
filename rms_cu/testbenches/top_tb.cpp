#include "../rmsnorm.h"

#include <cstdio>
#include <cstring>
#include <hls_math.h>
#include <hls_stream.h>
#include <stdio.h>
#include <streambuf>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <bitset>
#include "tb_main.h"


int top_tb(){
	std::cout<<"starting First Third testbench"<<std::endl;
	/* ===================== open neede files =========================== */

	/* ========== input files ============================= */
	std::ifstream input_tokens_dat("seed_42069/150_input_tokens.bin", std::ios::binary);
	// std::ifstream input_tokens_dat("top_data/TP_25_tokens.bin", std::ios::binary);

	std::ifstream rms_att_w("weights/rms_att_w.bin", std::ios::binary);
	// std::ifstream rms_final_w("weights/rms_final_w.bin", std::ios::binary);

	std::ifstream rms_tokens_out("seed_42069/qkv_tokens.bin", std::ios::binary);
	
/* =================== check if files opened successfully */
	if (!input_tokens_dat.is_open() ) {
	std::cout<<"input_tokens_dat. Already off to a bad start. boop."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!rms_att_w.is_open() ) {
	std::cout<<"rms_att_w. Already off to a bad start. boop."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!rms_tokens_out.is_open() ) {
	std::cout<<"rms_tokens_out. Already off to a bad start. boop."<<std::endl;
	exit(EXIT_FAILURE);
	}
	std::cout<<"Opened all the files sucessfully"<<std::endl;

	/* ============================== constants related to tb ===================================== */
	int pos = 150;
	const int layer_cnt = 12;
	const int out_data_size = MODEL_ELEMENTS * 4;
	const int rms_w_size = MODEL_ELEMENTS * 4 * layer_cnt;
	const int tokens_size = MODEL_ELEMENTS * 4;
	
	const int out_data_cnt = out_data_size / sizeof(fdata_v_t);
	const int rms_w_cnt = rms_w_size / sizeof(fdata_v_t);
	const int tokens_cnt = tokens_size / sizeof(fdata_v_t);

	

/* ===================================== declare our vectors ===================================== */

	std::vector<fdata_v_t> tokens_arr(tokens_cnt);

	std::vector<fdata_v_t> rms_w_arr(rms_w_cnt);

	std::vector<fdata_v_t> tokesns_out_gold(tokens_cnt);

	/* ================================== read data into array =================================== */

	input_tokens_dat.read(reinterpret_cast<char *>(tokens_arr.data()), tokens_size);
	rms_att_w.read(reinterpret_cast<char *>(rms_w_arr.data()), rms_w_size);
	rms_tokens_out.read(reinterpret_cast<char *>(tokesns_out_gold.data()), tokens_size);
	
	std::cout<<"Loaded the files into memory"<<std::endl;


/* ===================================== Declare the streams ========================================= */
/* remember - if it's suppsoed to be m_axi, no need to create a stream. just use the created vector, ie: foo_arr.data() */

	/* ============================ write inputs to the streams ====================== */
		

int curr_pos = 150;
	std::cout<<"Delcared and Loaded the Streams"<<std::endl;

	/* ============================ Call the function(s) ====================================== */

	std::cout<<"first one done"<<std::endl;

	rmsnorm_kernel(tokens_arr.data(), rms_w_arr.data(), 0);
	
	std::cout<< "========================= Tokens output array data ========================"<<std::endl;
	parse_results<fdata_v_t, float>(tokesns_out_gold, tokens_arr);

	return 0;

}