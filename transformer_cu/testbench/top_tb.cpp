// #include "../forward.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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


struct axi_reg{
	int POS;
	int N_DIM;
	int M_DIM; 
	int QKV_W;
	int QKV_sf_W;
	int Out_W;
	int Out_sf_W;
	int FF_w1w3_W;
	int FF_w1w3_sf_W;
	int FF_w2_W;
	int FF_w2_sf_W; 
	int Embed_W;
	int Embed_sf_W; 
	int rms_att_W;
	int rms_ffn_W; 
	int rms_final_W;
};

int top_tb(){
	std::cout<<"starting First Third testbench"<<std::endl;
	
	axi_reg axi_reg;
	std::cout<<"Opened all the files sucessfully"<<std::endl;


	std::string checkpoint = "weights/stories110M_q8.bin";
	std::ifstream file(checkpoint, std::ios::binary | std::ios::ate);
	std::ifstream out_value_dat("seed_42069_conv/150_output_value_cache_head_maj.bin", std::ios::binary);
	std::ifstream out_key_dat("seed_42069_conv/150_output_key_cache_head_maj.bin", std::ios::binary);
	// std::ifstream tokens_dat("seed_42069_conv/150_output_key_cache_head_maj.bin", std::ios::binary);

	std::ifstream key_output("seed_42069/150_output_k_tokens.bin", std::ios::binary);
	std::ifstream value_output("seed_42069/150_output_v_tokens.bin", std::ios::binary);
	std::ifstream query_output("seed_42069/150_output_q_tokens.bin", std::ios::binary);
	std::ifstream input_tokens("seed_42069/150_input_tokens.bin", std::ios::binary);
	// std::ifstream w2_output("seed_42069/TOP_25_xb2_mm_output_A1.bin", std::ios::binary);
	// std::ifstream w2_output("seed_42069/150_output_w2_tokens.bin", std::ios::binary);
	std::ifstream w2_output("seed_42069/150_output_w2_tokens.bin", std::ios::binary);
	std::ifstream w1_output("seed_42069/150_output_w1_tokens.bin", std::ios::binary);
	std::ifstream w3_output("seed_42069/150_output_w3_tokens.bin", std::ios::binary);

	if (!out_value_dat.is_open() ) {
	std::cout<<"out_value_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!out_key_dat.is_open() ) {
	std::cout<<"out_key_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!file.is_open() ) {
	std::cout<<"No. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!input_tokens.is_open() ) {
	std::cout<<"No input. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!w2_output.is_open() ) {
	std::cout<<"No w2. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!key_output.is_open() ) {
	std::cout<<"No k. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!value_output.is_open() ) {
	std::cout<<"No v. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!query_output.is_open() ) {
	std::cout<<"No q. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
//todo: chedk if file is open


	size_t rms_att_size = (MODEL_ELEMENTS * 12 * sizeof(my_float_t));
	size_t rms_ffn_size = rms_att_size;
	size_t rms_final_size = MODEL_ELEMENTS * sizeof(my_float_t);
	
	size_t nn_size = MODEL_ELEMENTS * MODEL_ELEMENTS;
	size_t nn_sf_size = nn_size * sizeof(float) / MODEL_SCALING_FACTOR;
	
	size_t nm_size = MODEL_ELEMENTS * MODEL_HIDDEN_DIM;
	size_t nm_sf_size = nm_size * sizeof(float) / MODEL_SCALING_FACTOR;
	
	size_t embed_size = MODEL_ELEMENTS * MODEL_TOKENS * sizeof(int8_t);
	size_t embed_sf_size = MODEL_ELEMENTS * MODEL_TOKENS * sizeof(my_float_t) / MODEL_SCALING_FACTOR;
	size_t qkv_size = MODEL_ELEMENTS * MODEL_ELEMENTS * MODEL_NUM_LAYERS * 3 * sizeof(int8_t);
	size_t qkv_sf_size = MODEL_ELEMENTS * MODEL_ELEMENTS * MODEL_NUM_LAYERS * 3 * sizeof(my_float_t) / MODEL_SCALING_FACTOR;
	size_t o_size = MODEL_ELEMENTS * MODEL_ELEMENTS * MODEL_NUM_LAYERS * 1 * sizeof(int8_t);
	size_t o_sf_size = MODEL_ELEMENTS * MODEL_ELEMENTS * MODEL_NUM_LAYERS * 1 * sizeof(my_float_t) / MODEL_SCALING_FACTOR;
	size_t w1w3_size = MODEL_ELEMENTS * MODEL_HIDDEN_DIM * MODEL_NUM_LAYERS * 2 * sizeof(int8_t);
	size_t w1w3_sf_size = MODEL_ELEMENTS * MODEL_HIDDEN_DIM * MODEL_NUM_LAYERS * 2 * sizeof(my_float_t) / MODEL_SCALING_FACTOR;
	size_t w2_size = MODEL_ELEMENTS * MODEL_HIDDEN_DIM * MODEL_NUM_LAYERS * 1 * sizeof(int8_t);
	size_t w2_sf_size = MODEL_ELEMENTS * MODEL_HIDDEN_DIM * MODEL_NUM_LAYERS * 1 * sizeof(my_float_t) / MODEL_SCALING_FACTOR;

	
	// file.seekg(0, std::ios::end);
	size_t file_size = file.tellg();
	file.seekg(0, std::ios::beg);
	
	size_t q_size = (MODEL_ELEMENTS * ((MODEL_ELEMENTS * 4 + MODEL_HIDDEN_DIM * 3 ) * MODEL_NUM_LAYERS + MODEL_TOKENS)) * sizeof(int8_t);
	size_t rms_size = (MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1)) * sizeof(my_float_t);
	size_t sf_size = (q_size * sizeof(my_float_t) / (sizeof(int8_t) * MODEL_SCALING_FACTOR));
	
	std::vector<idata_v_t> quant_w_arr(q_size / sizeof(idata_v_t));
	std::vector<fdata_v_t> sf_w_arr(sf_size / sizeof(fdata_v_t));
	std::vector<fdata_v_t> rms_w_arr(rms_size / sizeof(fdata_v_t));
	
	char * q_ptr = reinterpret_cast<char*>(quant_w_arr.data());
	char *sf_ptr = reinterpret_cast<char*>(sf_w_arr.data());
	char *rms_ptr = reinterpret_cast<char*>(rms_w_arr.data());

	size_t file_ptr = 256;
	size_t rms_idx = 0;
	file.seekg(file_ptr, std::ios::beg);
	
	axi_reg.rms_att_W = 0;
	file.read(rms_ptr + rms_idx, rms_att_size);
	rms_idx += rms_att_size;
	
	axi_reg.rms_ffn_W = axi_reg.rms_att_W + rms_att_size;
	file.read(rms_ptr + rms_idx, rms_ffn_size);
	rms_idx += rms_ffn_size;
	
	axi_reg.rms_final_W = axi_reg.rms_ffn_W + rms_ffn_size;
	file.read(rms_ptr + rms_idx, rms_final_size);
	file_ptr = file.tellg();
	
	size_t q_idx = 0;
	size_t sf_idx = 0;
	
	axi_reg.Embed_W = 0;
	file.read(q_ptr + q_idx, embed_size);
	q_idx += embed_size;
	
	axi_reg.Embed_sf_W = 0;
	file.read(sf_ptr + sf_idx, embed_sf_size);
	sf_idx += embed_sf_size;
	
	// read QKV
	axi_reg.QKV_sf_W = sf_idx;
	axi_reg.QKV_W = q_idx;
	
	file_ptr = file_ptr + embed_sf_size + embed_size;
	
	for (int i = 0; i < MODEL_NUM_LAYERS; i++) {
	
		for (int j = 0; j < 3; j++) {
			file.seekg((file_ptr + j * (nn_size + nn_sf_size) * (MODEL_NUM_LAYERS - 0)), std::ios::beg);
			file.read(q_ptr + q_idx, nn_size);
			q_idx += nn_size;
			file.read(sf_ptr + sf_idx, nn_sf_size);
			sf_idx += nn_sf_size;
		}
		file_ptr += (nn_sf_size + nn_size);
	}
	
	axi_reg.Out_sf_W = sf_idx;
	axi_reg.Out_W = q_idx;
	//already at Output
	for (int i = 0; i < MODEL_NUM_LAYERS; i++) {
		
		file.read(q_ptr + q_idx, nn_size);
		q_idx += nn_size;
		file.read(sf_ptr + sf_idx, nn_sf_size);
		sf_idx += nn_sf_size;
	}
	file_ptr = file.tellg();
	
	axi_reg.FF_w1w3_sf_W = sf_idx;
	axi_reg.FF_w1w3_W = q_idx;
	//now at w1
	for (int i = 0; i < MODEL_NUM_LAYERS; i++) {
	
		for (int j = 0; j < 2; j++) {
			file.seekg((file_ptr + j * 2 * (nm_size + nm_sf_size) * (MODEL_NUM_LAYERS - 0)), std::ios::beg); // skip over FFN2
			file.read(q_ptr + q_idx, nm_size);
			q_idx += nm_size;
			file.read(sf_ptr + sf_idx, nm_sf_size);
			sf_idx += nm_sf_size;
		}
		file_ptr += (nm_size + nm_sf_size);
	}

	axi_reg.FF_w2_W = q_idx;
	axi_reg.FF_w2_sf_W = sf_idx;
	file.seekg(file_ptr, std::ios::beg);
	for (int i = 0; i < MODEL_NUM_LAYERS; i++) {
		
		file.read(q_ptr + q_idx, nm_size);
		q_idx += nm_size;
		file.read(sf_ptr + sf_idx, nm_sf_size);
		sf_idx += nm_sf_size;
	}

	/* ============================== constants related to tb ===================================== */
	int pos = 150;
	const int layer_cnt = 12;
	const int out_data_size = MODEL_ELEMENTS * 4;
	const int quant_data_size = MODEL_ELEMENTS * MODEL_ELEMENTS * layer_cnt;
	const int sf_data_size = quant_data_size * 4 / MODEL_SCALING_FACTOR;
	const int slice_w_data_size = MODEL_ELEMENTS * MODEL_ELEMENTS;
	const int slice_sf_data_size = slice_w_data_size * 4 / MODEL_SCALING_FACTOR;
	const int rms_w_size = MODEL_ELEMENTS * 4 * layer_cnt;
	const int tokens_size = MODEL_ELEMENTS * 4;
	const int tok_w1_size = MODEL_HIDDEN_DIM * 4;
	const int logits_size = MODEL_HIDDEN_DIM * 2 * sizeof(float);//* MODEL_TOKENS * 4;//
	const int logits_quant_size = MODEL_ELEMENTS * MODEL_TOKENS * 1;
	const int logits_sf_size = MODEL_ELEMENTS * MODEL_TOKENS / MODEL_SCALING_FACTOR * 4;
	// const int sf_el = MODEL_ELEMENTS / 64;
	// const int wo_sf_size = 36864;
	
	// const int wo_size = 589824;
	const int cache_size = 1024*768*4 * 12;
	const int t_size = 768 * 4;
	const int xb2_size = 3072;
	const int hd_tok_size = MODEL_ELEMENTS * MODEL_HIDDEN_DIM * layer_cnt;
	const int hd_sf_size = hd_tok_size / MODEL_SCALING_FACTOR * 4;
	// const int rms_tok_size = MODEL_ELEMENTS * 4 * layer_cnt;
	
	const int out_data_cnt = out_data_size / sizeof(mfdata_v_t);
	const int quant_data_cnt = quant_data_size / sizeof(idata_v_t);
	const int sf_data_cnt = sf_data_size / sizeof(fdata_v_t);
	const int rms_w_cnt = rms_w_size / sizeof(fdata_v_t);
	const int tokens_cnt = tokens_size / sizeof(fdata_v_t);
	const int tok_w1_cnt = tok_w1_size / sizeof(fdata_v_t);
	const int logits_cnt = logits_size / sizeof(fdata_v_t);
	const int logits_q_cnt = logits_quant_size / sizeof(idata_v_t);
	const int logits_sf_cnt = logits_sf_size / sizeof(fdata_v_t);
	const int slice_w_data_cnt = slice_w_data_size / sizeof(mfdata_v_t);
	const int slice_sf_data_cnt = slice_sf_data_size / sizeof(fdata_v_t);
	// const int wo_cnt = wo_size / sizeof(idata_v_t);
	// const int wo_sf_cnt = wo_sf_size / sizeof(fdata_v_t);
	const int cache_cnt = cache_size / sizeof(mfdata_v_t);
	const int tok_cnt = t_size / sizeof(mfdata_v_t);
	const int xb2_cnt = xb2_size / sizeof(mfdata_v_t);
	const int hd_tok_cnt = hd_tok_size / sizeof(idata_v_t);
	const int hd_sf_cnt = hd_sf_size / sizeof(fdata_v_t);
	// const int rms_tok_cnt = rms_tok_size / sizeof(mfdata_v_t);

	

/* ===================================== declare our vectors ===================================== */

	std::vector<fdata_v_t> tokens_arr(tokens_cnt * 3);
	std::vector<fdata_v_t> swiglu_arr(hd_tok_cnt * 2);
	std::vector<adata_v_t> mha_tokens_arr((tokens_size / (sizeof(adata_v_t))));
	std::vector<fdata_v_t> output_arr(logits_cnt);
	std::vector<fdata_v_t> golden_output_arr(logits_cnt);
	// std::vector<mfdata_v_t> val_in_rope_arr(tokens_cnt);
	// std::vector<mfdata_v_t> key_in_rope_arr(tokens_cnt);
	
	// query_output.seekg(0, std::ios::end);
	// file_size = query_output.tellg();
	// query_output.seekg(0, std::ios::beg);
	char *goa = reinterpret_cast<char*>(golden_output_arr.data());
	size_t goa_idx = 0;
	// query_output.read(goa, file_size);
	// goa_idx += file_size;
	
	// key_output.seekg(0, std::ios::beg);
	// key_output.read(goa + goa_idx, file_size);
	// goa_idx += file_size;
	
	// value_output.seekg(0, std::ios::beg);
	// value_output.read(goa + goa_idx, file_size);

	w2_output.seekg(0, std::ios::end);
	file_size = w2_output.tellg();
	w2_output.seekg(0, std::ios::beg);
	
	w2_output.read(goa, file_size);


	// w1_output.seekg(0, std::ios::end);
	// file_size = w1_output.tellg();
	// w1_output.seekg(0, std::ios::beg);
	
	// w1_output.read(goa, file_size);
	// w3_output.read(goa + file_size, file_size);




	char *oa = reinterpret_cast<char*>(output_arr.data());
	input_tokens.seekg(0, std::ios::end);
	file_size = input_tokens.tellg();
	input_tokens.seekg(0, std::ios::beg);
	input_tokens.read(oa, file_size);


	// w1_output.seekg(0, std::ios::end);
	// file_size = w1_output.tellg();
	// w1_output.seekg(0, std::ios::beg);
	
	// w1_output.read(oa, file_size);
	// w3_output.read(oa + file_size, file_size);
	


	// std::vector<std::vector<fdata_v_t>> query_arr(2, std::vector<fdata_v_t>(out_data_cnt));
	std::vector<std::vector<mfdata_v_t>> key_arr(2, std::vector<mfdata_v_t>(cache_cnt));
	std::vector<std::vector<mfdata_v_t>> value_arr(2, std::vector<mfdata_v_t>(cache_cnt));
	std::vector<std::vector<mfdata_v_t>> tok_out_arr(2, std::vector<mfdata_v_t>(tokens_cnt));
	std::vector<std::vector<mfdata_v_t>> tok_w1_out_arr(2, std::vector<mfdata_v_t>(tok_w1_cnt));
	std::vector<std::vector<mfdata_v_t>> log_out_arr(2, std::vector<mfdata_v_t>(logits_cnt));
	std::vector<std::vector<my_float_t>> att_score_arr(2, std::vector<my_float_t>(MODEL_ELEMENTS));


	std::vector<mfdata_v_t> key_in_arr(tokens_cnt * 3);
	std::vector<mfdata_v_t> value_in_arr(tokens_cnt * 3);
	std::vector<mfdata_v_t> key_arr_a(cache_cnt);
	std::vector<mfdata_v_t> value_arr_a(cache_cnt);


	// w2_output.read(reinterpret_cast<char *>(golden_output_arr.data()), tokens_size);

	out_key_dat.read(reinterpret_cast<char *>(key_arr[0].data()), cache_size);
	out_value_dat.read(reinterpret_cast<char *>(value_arr[0].data()), cache_size);
	// memcpy(key_arr[0].data(), key_arr_a.data(), cache_size);
	// memcpy(value_arr[0].data(), value_arr_a.data(), cache_size);
	memcpy(key_arr_a.data(), key_arr[0].data(), cache_size);
	memcpy(value_arr_a.data(), value_arr[0].data(), cache_size);

	/* ================================== read data into array =================================== */

	
	std::cout<<"Loaded the files into memory"<<std::endl;


/* ===================================== Declare the streams ========================================= */
/* remember - if it's suppsoed to be m_axi, no need to create a stream. just use the created vector, ie: foo_arr.data() */

	/* ============================ write inputs to the streams ====================== */
		

int curr_pos = 150;
	std::cout<<"Delcared and Loaded the Streams"<<std::endl;
transformer_cu(	output_arr.data(), sf_w_arr.data(), quant_w_arr.data(), 
								output_arr.data(), sf_w_arr.data(), quant_w_arr.data(), 
								rms_w_arr.data(), key_arr_a.data(), value_arr_a.data(), 
								curr_pos, MODEL_ELEMENTS, MODEL_ELEMENTS, 
								axi_reg.QKV_W, axi_reg.QKV_sf_W, axi_reg.Out_W, axi_reg.Out_sf_W, 
								axi_reg.FF_w1w3_W, axi_reg.FF_w1w3_sf_W, axi_reg.FF_w2_W, 
								axi_reg.FF_w2_sf_W, axi_reg.Embed_W, axi_reg.Embed_sf_W, 
								axi_reg.rms_att_W, axi_reg.rms_ffn_W, axi_reg.rms_final_W, 4);

	std::fill(output_arr.begin() + 192, output_arr.end(), 0);
	std::cout<< "========================= Tokens output array data ========================"<<std::endl;
	parse_results<fdata_v_t, float>(golden_output_arr, output_arr);
	// std::cout<< "========================= Tokens output array data ========================"<<std::endl;
	// parse_results<mfdata_v_t, float>(tok_w1_out_arr[0], tok_w1_out_arr[1]);

	std::cout<< "========================= Value cache array data ========================"<<std::endl;
	parse_cache_results<mfdata_v_t, float>(value_arr[0], value_arr_a);

	std::cout<< "========================= Key cache array data ========================"<<std::endl;
	parse_cache_results<mfdata_v_t, float>(key_arr[0], key_arr_a);

	return 0;

}