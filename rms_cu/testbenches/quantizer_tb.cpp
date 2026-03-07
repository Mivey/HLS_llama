#include "../forward.h"
#include "../matmul.h"
#include <cstdint>
#include <cstdio>
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
#include "../quantizer.h"


// int mat_mul_tb(){
int quantizer_tb(){
	std::cout<<"starting testbench"<<std::endl;
	/* ===================== open neede files =========================== */

	/* ========== input files ============================= */
	
	std::ifstream qkv_dat("top_data/P_25_rms_outputt.bin", std::ios::binary);
	std::ifstream wo_dat("top_data/TOP_25_xb2_mm_output_A1.bin", std::ios::binary);

	/* =========================== output data ===================== */
	std::ifstream tok_dat("top_data/P_25_quant_tok.bin", std::ios::binary);
	std::ifstream tok_sf_dat("top_data/P_25_quant_t_sf.bin", std::ios::binary);
	std::ifstream wo_tok_dat("top_data/TAP_25_quant_tok.bin", std::ios::binary);
	std::ifstream wo_tok_sf_dat("top_data/TAP_25_quant_t_sf.bin", std::ios::binary);


/* =================== check if files opened successfully */

	if (!qkv_dat.is_open() ) {
	std::cout<<"qkv_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!wo_dat.is_open() ) {
	std::cout<<"wo_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	
	if (!tok_dat.is_open() ) {
	std::cout<<"tok_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!tok_sf_dat.is_open() ) {
	std::cout<<"tok_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!wo_tok_dat.is_open() ) {
	std::cout<<"wo_tok_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!wo_tok_sf_dat.is_open() ) {
	std::cout<<"wo_tok_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	/* ============================== constants related to tb ===================================== */
	const int pos = 150;
	const int sf_el = MODEL_ELEMENTS / 64;
	const int w_sf_size = 36864;
	const int w_size = 589824;
	const int t_sf_size = 48;
	const int t_size = 768;
	const int out_size = 3072;

	const int w_cnt = w_size / sizeof(idata_v_t);
	const int w_sf_cnt = w_sf_size / sizeof(fdata_v_t);
	const int tok_sf_cnt = t_sf_size / sizeof(fdata_v_t);
	const int tok_cnt = t_size / sizeof(idata_v_t);
	const int out_cnt = out_size / sizeof(fdata_v_t);		// (768 elements  * 4 bytes/elements) / (4 elements/ point * 4 bytes/elemnts) = 192 points
	const int rms_out_cnt = out_size / sizeof(fdata_v_t);

/* ===================================== declare our vectors ===================================== */

	std::vector<fdata_v_t> qkv_out_arr(rms_out_cnt);
	std::vector<fdata_v_t> wo_out_arr(rms_out_cnt);
	
	std::vector<std::vector<idata_v_t>> tok_arr(2, std::vector<idata_v_t>(tok_cnt));
	std::vector<std::vector<fdata_v_t>> tok_sf_arr(2, std::vector<fdata_v_t>(tok_sf_cnt));
	std::vector<std::vector<idata_v_t>> wo_tok_arr(2, std::vector<idata_v_t>(tok_cnt));
	std::vector<std::vector<fdata_v_t>> wo_tok_sf_arr(2, std::vector<fdata_v_t>(tok_sf_cnt));

	/* ================================== read data into array =================================== */

	qkv_dat.read(reinterpret_cast<char *>(qkv_out_arr.data()), out_size);
	wo_dat.read(reinterpret_cast<char *>(wo_out_arr.data()), out_size);

	tok_dat.read(reinterpret_cast<char *>(tok_arr[0].data()), t_size);
	tok_sf_dat.read(reinterpret_cast<char *>(tok_sf_arr[0].data()), t_sf_size);

	wo_tok_dat.read(reinterpret_cast<char *>(wo_tok_arr[0].data()), t_size);
	wo_tok_sf_dat.read(reinterpret_cast<char *>(wo_tok_sf_arr[0].data()), t_sf_size);


/* ===================================== Declare the streams ========================================= */
/* remember - if it's suppsoed to be m_axi, no need to create a stream. just use the created vector, ie: foo_arr.data() */

	hls::stream<fdata_v_t> qkv_output_in("Value Scaling Factor data input");
	hls::stream<fdata_v_t> wo_output_in("Value Scaling Factor data wo input");
	hls::stream<idata_v_t> tok_out("token value data output");
	hls::stream<fdata_v_t> tok_sf_out("token value Scaling Factor data output");
	hls::stream<idata_v_t> wo_tok_out("token value wo data output");
	hls::stream<fdata_v_t> wo_tok_sf_out("token value Scaling Factor wo data output");
	

	/* ============================ write inputs to the streams ====================== */
	
	
	rms_data_in_loop:
	for (int i = 0; i < rms_out_cnt; i++) {
		qkv_output_in.write(qkv_out_arr[i]);
	}
	for (int i = 0; i < rms_out_cnt; i++) {
		wo_output_in.write(wo_out_arr[i]);
	}

	/* ============================ Call the function(s) ====================================== */

	// quantizer_kernel<SCALING_FACTOR>(rms_output_in, tok_out, tok_sf_out);
	// quantizer_kernel<MODEL_SCALING_FACTOR, MODEL_ELEMENTS>(tok_sf_out, tok_out, qkv_output_in);
	// quantizer_kernel<MODEL_SCALING_FACTOR, MODEL_ELEMENTS>(wo_tok_sf_out, wo_tok_out, wo_output_in);
	top_quant(tok_sf_out, tok_out, qkv_output_in);
	top_quant(wo_tok_sf_out, wo_tok_out, wo_output_in);

	/* =============================== get all the data =================================== */

	for (int i = 0; i < tok_sf_cnt; i++) {
		tok_sf_arr[1][i] = tok_sf_out.read();
	}
	for (int i = 0; i < tok_cnt; i++) {
		tok_arr[1][i] = tok_out.read();
	}
	for (int i = 0; i < tok_sf_cnt; i++) {
		wo_tok_sf_arr[1][i] = wo_tok_sf_out.read();
	}
	for (int i = 0; i < tok_cnt; i++) {
		wo_tok_arr[1][i] = wo_tok_out.read();
	}


	/*  ====================================== process the results ============================ */

	std::cout<< "========================= tokens scaling factor data ========================"<<std::endl;
	parse_results<fdata_v_t, float>(tok_sf_arr[0], tok_sf_arr[1]);
	std::cout<< "========================= tokens quantized data ========================"<<std::endl;
	parse_results<idata_v_t, int8_t>(tok_arr[0], tok_arr[1]);
	std::cout<< "========================= wo tokens scaling factor data ========================"<<std::endl;
	parse_results<fdata_v_t, float>(wo_tok_sf_arr[0], wo_tok_sf_arr[1]);
	std::cout<< "========================= wo tokens quantized data ========================"<<std::endl;
	parse_results<idata_v_t, int8_t>(wo_tok_arr[0], wo_tok_arr[1]);

	return 0;

}