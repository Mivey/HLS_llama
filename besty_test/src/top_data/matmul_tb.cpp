#include "../forward.h"
#include "../matmul.h"
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


// int mat_mul_tb(){
int mat_mul_tb(){
	std::cout<<"starting testbench"<<std::endl;
	/* ===================== open neede files =========================== */

	/* ========== input files ============================= */
	std::ifstream qw_dat("top_data/P_25_quant_wq.bin", std::ios::binary);
	std::ifstream qw_sf_dat("top_data/P_25_quant_wq_sf.bin", std::ios::binary);

	std::ifstream kw_dat("top_data/P_25_quant_wk.bin", std::ios::binary);
	std::ifstream kw_sf_dat("top_data/P_25_quant_wk_sf.bin", std::ios::binary);

	std::ifstream vw_dat("top_data/P_25_quant_wv.bin", std::ios::binary);
	std::ifstream vw_sf_dat("top_data/P_25_quant_wv_sf.bin", std::ios::binary);

	std::ifstream tok_dat("top_data/P_25_quant_tok.bin", std::ios::binary);
	std::ifstream tok_sf_dat("top_data/P_25_quant_t_sf.bin", std::ios::binary);

	std::ifstream att_tok_dat("top_data/TAP_25_quant_tok.bin", std::ios::binary);
	std::ifstream att_tok_sf_dat("top_data/TAP_25_quant_t_sf.bin", std::ios::binary);

	std::ifstream wo_dat("top_data/TP_25_quant_wo_att_out.bin", std::ios::binary);
	std::ifstream wo_sf_dat("top_data/TP_25_quant_wo_att_out_sf.bin", std::ios::binary);

	/* =========================== output data ===================== */
	std::ifstream query_out("top_data/P_25_out_query.bin", std::ios::binary);
	std::ifstream key_out("top_data/P_25_out_key.bin", std::ios::binary);
	std::ifstream value_out("top_data/P_25_out_value.bin", std::ios::binary);
	std::ifstream wo_out("top_data/TOP_25_xb2_mm_output.bin", std::ios::binary);

/* =================== check if files opened successfully */
	

	if (!wo_out.is_open() ) {
	std::cout<<"wo_out. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	
	if (!wo_dat.is_open() ) {
	std::cout<<"wo_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!wo_sf_dat.is_open() ) {
	std::cout<<"wo_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!att_tok_dat.is_open() ) {
	std::cout<<"att_tok_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!att_tok_sf_dat.is_open() ) {
	std::cout<<"att_tok_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!qw_dat.is_open() ) {
	std::cout<<"qw_dat. Already off to a bad start. boop."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!qw_sf_dat.is_open() ) {
	std::cout<<"qw_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	
	if (!qw_dat.is_open() ) {
	std::cout<<"tok_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!tok_sf_dat.is_open() ) {
	std::cout<<"tok_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!kw_dat.is_open() ) {
	std::cout<<"kw_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!kw_sf_dat.is_open() ) {
	std::cout<<"kw_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!vw_dat.is_open() ) {
	std::cout<<"vw_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!vw_sf_dat.is_open() ) {
	std::cout<<"vw_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!query_out.is_open() ) {
	std::cout<<"query_out. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!value_out.is_open() ) {
	std::cout<<"value_out. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!key_out.is_open() ) {
	std::cout<<"key_out. Already off to a bad start."<<std::endl;
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

/* ===================================== declare our vectors ===================================== */
	std::vector<idata_v_t> qw_arr(w_cnt);
	std::vector<fdata_v_t> qw_sf_arr(w_sf_cnt);

	std::vector<idata_v_t> kw_arr(w_cnt);
	std::vector<fdata_v_t> kw_sf_arr(w_sf_cnt);

	std::vector<idata_v_t> vw_arr(w_cnt);
	std::vector<fdata_v_t> vw_sf_arr(w_sf_cnt);

	std::vector<idata_v_t> wo_arr(w_cnt);
	std::vector<fdata_v_t> wo_sf_arr(w_sf_cnt);

	std::vector<idata_v_t> tok_arr(tok_cnt);
	std::vector<fdata_v_t> tok_sf_arr(tok_sf_cnt);

	std::vector<idata_v_t> att_tok_arr(tok_cnt);
	std::vector<fdata_v_t> att_tok_sf_arr(tok_sf_cnt);
	
	std::vector<std::vector<fdata_v_t>> query_arr(2, std::vector<fdata_v_t>(out_cnt));
	std::vector<std::vector<fdata_v_t>> key_arr(2, std::vector<fdata_v_t>(out_cnt));
	std::vector<std::vector<fdata_v_t>> value_arr(2, std::vector<fdata_v_t>(out_cnt));
	std::vector<std::vector<fdata_v_t>> mm_wo_arr(2, std::vector<fdata_v_t>(out_cnt));

	/* ================================== read data into array =================================== */

	qw_dat.read(reinterpret_cast<char *>(qw_arr.data()), w_size);
	qw_sf_dat.read(reinterpret_cast<char *>(qw_sf_arr.data()), w_sf_size);

	kw_dat.read(reinterpret_cast<char *>(kw_arr.data()), w_size);
	kw_sf_dat.read(reinterpret_cast<char *>(kw_sf_arr.data()), w_sf_size);

	vw_dat.read(reinterpret_cast<char *>(vw_arr.data()), w_size);
	vw_sf_dat.read(reinterpret_cast<char *>(vw_sf_arr.data()), w_sf_size);

	tok_dat.read(reinterpret_cast<char *>(tok_arr.data()), t_size);
	tok_sf_dat.read(reinterpret_cast<char *>(tok_sf_arr.data()), t_sf_size);

	wo_dat.read(reinterpret_cast<char *>(wo_arr.data()), w_size);
	wo_sf_dat.read(reinterpret_cast<char *>(wo_sf_arr.data()), w_sf_size);

	att_tok_dat.read(reinterpret_cast<char *>(att_tok_arr.data()), t_size);
	att_tok_sf_dat.read(reinterpret_cast<char *>(att_tok_sf_arr.data()), t_sf_size);

	query_out.read(reinterpret_cast<char *>(query_arr[0].data()), out_size);
	key_out.read(reinterpret_cast<char *>(key_arr[0].data()), out_size);
	value_out.read(reinterpret_cast<char *>(value_arr[0].data()), out_size);
	wo_out.read(reinterpret_cast<char *>(mm_wo_arr[0].data()), out_size);


/* ===================================== Declare the streams ========================================= */
/* remember - if it's suppsoed to be m_axi, no need to create a stream. just use the created vector, ie: foo_arr.data() */

	hls::stream<idata_v_t> qw_in("Query data input");
	hls::stream<fdata_v_t> qw_sf_in("Query Scaling Factor data input");
	hls::stream<idata_v_t> tq_in("token query data input");
	hls::stream<fdata_v_t> tq_sf_in("token query Scaling Factor data input");

	hls::stream<idata_v_t> kw_in("Key data input");
	hls::stream<fdata_v_t> kw_sf_in("Key Scaling Factor data input");
	hls::stream<idata_v_t> tk_in("token key data input");
	hls::stream<fdata_v_t> tk_sf_in("token key Scaling Factor data input");

	hls::stream<idata_v_t> vw_in("Value data input");
	hls::stream<fdata_v_t> vw_sf_in("Value Scaling Factor data input");
	hls::stream<idata_v_t> tv_in("token value data input");
	hls::stream<fdata_v_t> tv_sf_in("token value Scaling Factor data input");

	hls::stream<idata_v_t> wo_in("Value data input");
	hls::stream<fdata_v_t> wo_sf_in("Value Scaling Factor data input");
	hls::stream<idata_v_t> att_in("token value data input");
	hls::stream<fdata_v_t> att_sf_in("token value Scaling Factor data input");

	hls::stream<fdata_v_t> qs_out("Query output Stream");
	hls::stream<fdata_v_t> ks_out("Key output Stream");
	hls::stream<fdata_v_t> vs_out("Value output Stream");
	hls::stream<fdata_v_t> wo_out_str("wo output Stream");

	/* ============================ write inputs to the streams ====================== */
	
	
	for (int i = 0; i < w_cnt; i++) {
		wo_in.write(wo_arr[i]);
	}

	for (int i = 0 ; i < w_sf_cnt; i++) {
		wo_sf_in.write(wo_sf_arr[i]);
	}

	for (int i = 0; i < tok_sf_cnt; i++) {
		att_sf_in.write(att_tok_sf_arr[i]);
	}

	for (int i = 0; i < tok_cnt; i++) {
		att_in.write(att_tok_arr[i]);
	}

	quant_data_loop:
	for (int i = 0; i < w_cnt; i++) {
		qw_in.write(qw_arr[i]);
	}
	// quant_data_loop:
	for (int i = 0; i < w_cnt; i++) {
		kw_in.write(kw_arr[i]);
	}
	// quant_data_loop:
	for (int i = 0; i < w_cnt; i++) {
		vw_in.write(vw_arr[i]);
	}

	scaling_factor_loop:
	for (int i = 0 ; i < w_sf_cnt; i++) {
		kw_sf_in.write(kw_sf_arr[i]);
	}
	for (int i = 0 ; i < w_sf_cnt; i++) {
		vw_sf_in.write(vw_sf_arr[i]);
	}
	for (int i = 0 ; i < w_sf_cnt; i++) {
		qw_sf_in.write(qw_sf_arr[i]);
	}

	token_sf_loop:
	for (int i = 0; i < tok_sf_cnt; i++) {
		tv_sf_in.write(tok_sf_arr[i]);
	}
	for (int i = 0; i < tok_sf_cnt; i++) {
		tk_sf_in.write(tok_sf_arr[i]);
	}
	for (int i = 0; i < tok_sf_cnt; i++) {
		tq_sf_in.write(tok_sf_arr[i]);
	}

	token_quant_loop:
	for (int i = 0; i < tok_cnt; i++) {
		tv_in.write(tok_arr[i]);
	}
	for (int i = 0; i < tok_cnt; i++) {
		tk_in.write(tok_arr[i]);
	}
	
	for (int i = 0; i < tok_cnt; i++) {
		tq_in.write(tok_arr[i]);
	}


	/* ============================ Call the function(s) ====================================== */

	// matmult_kernel<MODEL_ELEMENTS, MODEL_ELEMENTS, MODEL_SCALING_FACTOR>(qs_out, tq_sf_in, tq_in, qw_in, qw_sf_in);
	// matmult_kernel<MODEL_ELEMENTS, MODEL_ELEMENTS, MODEL_SCALING_FACTOR>(ks_out, tk_sf_in, tk_in, kw_in, kw_sf_in);
	// matmult_kernel<MODEL_ELEMENTS, MODEL_ELEMENTS, MODEL_SCALING_FACTOR>(vs_out, tv_sf_in, tv_in, vw_in, vw_sf_in);
	// matmult_kernel<MODEL_ELEMENTS, MODEL_ELEMENTS, MODEL_SCALING_FACTOR>(wo_out_str, att_sf_in, att_in, wo_in, wo_sf_in);

	matmult_cpp_kernel(query_arr[1].data(), ks_out, tq_sf_in, tq_in, qw_in, qw_sf_in);
	// matmult_kernel<MODEL_ELEMENTS, MODEL_ELEMENTS, MODEL_SCALING_FACTOR>(tk_sf_in, tk_in, kw_in, ks_out, kw_sf_in);
	// matmult_kernel<MODEL_ELEMENTS, MODEL_ELEMENTS, MODEL_SCALING_FACTOR>(tv_sf_in, tv_in, vw_in, vs_out, vw_sf_in);

	/* =============================== get all the data =================================== */

	// for (int i = 0; i < out_cnt; i++) {
	// 	query_arr[1][i] = qs_out.read();
	// }
	// for (int i = 0; i < out_cnt; i++) {
	// 	value_arr[1][i] = vs_out.read();
	// }
	// for (int i = 0; i < out_cnt; i++) {
	// 	key_arr[1][i] = ks_out.read();
	// }

	// for (int i = 0; i < out_cnt; i++) {
	// 	mm_wo_arr[1][i] = wo_out_str.read();
	// }


	/*  ====================================== process the results ============================ */

	std::cout<< "========================= query array data ========================"<<std::endl;
	parse_results<fdata_v_t, float>(query_arr[0], query_arr[1]);
	// std::cout<< "========================= key array data ========================"<<std::endl;
	// parse_results<fdata_v_t, float>(key_arr[0], key_arr[1]);
	// std::cout<< "========================= value array data ========================"<<std::endl;
	// parse_results<fdata_v_t, float>(value_arr[0], value_arr[1]);
	// std::cout<< "========================= wo array data ========================"<<std::endl;
	// parse_results<fdata_v_t, float>(mm_wo_arr[0], mm_wo_arr[1]);

	return 0;

}