/* Inference for Llama-2 Transformer model in pure C */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string>
#include <memory.h>
#include <vector>
#include "../forward.h"

// XRT includes
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

// #define MODEL_ELEMENTS 768
// #define MODEL_SCALING_FACTOR 64
// #define MODEL_HIDDEN_DIM 2048 // first issue is here, this should not be 12
// #define MODEL_HEAD_SIZE 12
// #define MODEL_NUM_LAYERS 12

#ifdef DEBUG
    // In Debug mode: Print to standard out
    #define LOG_DEBUG(msg) std::cout << "[DEBUG] " << msg << std::endl
#else
    // In Release mode: Compile to absolutely nothing (No-Op)
    #define LOG_DEBUG(msg) 
#endif

// void rmsnorm(float* o, float* x, float* weight, int size);

void softmax(float* x, int size) ;

// My Hardware Kernel
class MatMultClass{
	public:
		MatMultClass(xrt::device &d, xrt::uuid &u): device(d), uuid(u){
			
			kernel = xrt::kernel(device, uuid, "double_matmult_kernel");
		}
		//void matmult_kernel(mfdata_v_t *out, mfdata_v_t *fl_tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER){
		void run(xrt::bo &output, xrt::bo &input, xrt::bo &sf_bo, xrt::bo &w_bo, const int n_dim, const int m_dim, const int curr_layer){
			auto run = kernel(output, input, sf_bo, w_bo, n_dim, m_dim, curr_layer, 0);
			run.wait();
		}

		xrt::run start(xrt::bo &output, xrt::bo &input, xrt::bo &sf_bo, xrt::bo &w_bo, const int n_dim, const int m_dim, const int curr_layer){
			return kernel(output, input, sf_bo, w_bo, n_dim, m_dim, curr_layer, 0);
		}

		void wait(xrt::run &run_handle){
			run_handle.wait();
		}

	private:
	
		xrt::device device;
		xrt::uuid uuid;
		xrt::kernel kernel;
};


class MHAClass{
	public:
		MHAClass(xrt::device &d, xrt::uuid &u): device(d), uuid(u){
			
			kernel = xrt::kernel(device, uuid, "mha_kernel");
			
			allocate_cache();
		}
		// void mha_kernel(mfdata_v_t *tokens, //6 mha_kernel
    //             mfdata_v_t *key_cache, 
    //             mfdata_v_t *value_cache, 
		// 						mfdata_v_t *key_cache_in, mfdata_v_t *value_cache_in,
    //             const int POS, const int CURR_LAYER){

		void run(xrt::bo &tokens, xrt::bo &key, xrt::bo &value, const int layer_idx, const int pos){
			auto run = kernel(tokens, key_bo, value_bo, key, value, pos, layer_idx);
			run.wait();
		}

	private:
	
		xrt::device device;
		xrt::uuid uuid;
		xrt::kernel kernel;
		xrt::bo key_bo;
		xrt::bo value_bo;
		
		
		const size_t c_size = (size_t)MODEL_ELEMENTS * MODEL_SEQUENCE_LEN * MODEL_NUM_LAYERS * sizeof(float);

		void allocate_cache(){
			//create the containers for the weights
			key_bo = xrt::bo(device, c_size, kernel.group_id(0));
			value_bo = xrt::bo(device, c_size, kernel.group_id(0));
		}
};
class SwigluClass{
	public:
		SwigluClass(xrt::device &d, xrt::uuid &u): device(d), uuid(u){
			
			kernel = xrt::kernel(device, uuid, "swiglu_kernel");
			
		}

		void run(xrt::bo &out, xrt::bo &w1, xrt::bo &w3){
			auto run = kernel(out, w1, w3);
			run.wait();
		}

	private:
	
		xrt::device device;
		xrt::uuid uuid;
		xrt::kernel kernel;
		
};
class RMSClass{
	public:
		RMSClass(xrt::device &d, xrt::uuid &u): device(d), uuid(u){
			
			kernel = xrt::kernel(device, uuid, "rmsnorm_kernel");
			
		}
// void rmsnorm_kernel(fdata_v_t * output, fdata_v_t *tokens, fdata_v_t *weights, const int CURR_LAYER){
		void run(xrt::bo &output, xrt::bo &tokens, xrt::bo &w_bo, const int l){
			auto run = kernel(output, tokens, w_bo, l);
			run.wait();
		}

	private:
	
		xrt::device device;
		xrt::uuid uuid;
		xrt::kernel kernel;		
};
class ResConClass{
	public:
		ResConClass(xrt::device &d, xrt::uuid &u): device(d), uuid(u){
			
			kernel = xrt::kernel(device, uuid, "rmsnorm_kernel");
			
		}
// void rescon_kernel(fdata_v_t *out, fdata_v_t *x, fdata_v_t *xb)
		void run(xrt::bo &out, xrt::bo &x_bo, xrt::bo &xb_bo){
			auto run = kernel(out, x_bo, xb_bo);
			run.wait();
		}

	private:
	
		xrt::device device;
		xrt::uuid uuid;
		xrt::kernel kernel;		
};
typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;


class ForwardBlock{
	public:
		ForwardBlock(
			int device_id, std::string& binaryFile, 
			std::string &QKV_w_comb, std::string &QKV_sf_comb,
			std::string &out_w, std::string &out_sf, 
			std::string &ffn1_w_comb, std::string &ffn1_sf_comb, 
			std::string &ffn3_w_comb, std::string &ffn3_sf_comb, 
			std::string &ffn2_w, std::string &ffn2_sf,
			std::string &logits_w, std::string &logits_sf,
			std::string &rms_att_w, std::string &rms_ffn_w, std::string &rms_final_w,
			Transformer *t, const int mm_cu_cnt){
				
			try {
				device = xrt::device(device_id);

			std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
     	std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";
				uuid = device.load_xclbin(binaryFile);
				// xrt::kernel kernel = xrt::kernel(device, uuid, "faker_matmult_kernel");
				// g_id = kernel.group_id(2);
				// std::cout << "g_id is "<< g_id<< std::endl;
				g_id = 1;
				
			} catch (const std::exception& e) {
			throw std::runtime_error(std::string(e.what()));
			}
			
			//Convience pointers f
			p = &t->config;
			w = &t->weights;
			s = &t->state;

			
			std::cout<<"did I test it here?\n";

			int qkv_w_size = p->dim * p->dim * p->n_layers * 3;
			int qkv_sf_size = qkv_w_size / MODEL_SCALING_FACTOR * sizeof(float);
			int out_w_size = p->dim * p->dim * p->n_layers;
			int out_sf_size = out_w_size / MODEL_SCALING_FACTOR * sizeof(float);
			int ffn13_w_size = p->dim * p->hidden_dim * p->n_layers;
			int ffn13_sf_size = ffn13_w_size / MODEL_SCALING_FACTOR * sizeof(float);
			int ffn2_w_size = p->hidden_dim * p->dim * p->n_layers;
			int ffn2_sf_size = ffn2_w_size / MODEL_SCALING_FACTOR * sizeof(float);
			int logits_w_size = p->dim * p->vocab_size ;
			int logits_sf_size = logits_w_size / MODEL_SCALING_FACTOR * sizeof(float);
			int rms_size = p->dim * p->n_layers * sizeof(float);
			int rms_final_size = p->dim * sizeof(float);
			
			std::cout<<"Compiled on "<<__DATE__<<" at "<<__TIME__<<".\n";
			std::cout<<"qkv sizes:"<<qkv_w_size<<" and "<<qkv_sf_size<<"\n";
			std::cout<<" just to make sure "<<p->dim<<std::endl;
			qkv_w_bo = mem_init(QKV_w_comb, qkv_w_size);
			qkv_sf_bo = mem_init(QKV_sf_comb, qkv_sf_size);
			out_w_bo = mem_init(out_w, out_w_size);
			out_sf_bo = mem_init(out_sf, out_sf_size);
			ffn1_w_bo = mem_init(ffn1_w_comb, ffn13_w_size);
			ffn1_sf_bo = mem_init(ffn1_sf_comb, ffn13_sf_size);
			ffn3_w_bo = mem_init(ffn3_w_comb, ffn13_w_size);
			ffn3_sf_bo = mem_init(ffn3_sf_comb, ffn13_sf_size);
			ffn2_w_bo = mem_init(ffn2_w, ffn2_w_size);
			ffn2_sf_bo = mem_init(ffn2_sf, ffn2_sf_size);
			logits_w_bo = mem_init(logits_w, logits_w_size);
			logits_sf_bo = mem_init(logits_sf, logits_sf_size);
			rms_att_bo = mem_init(rms_att_w, rms_size);
			rms_ffn_bo = mem_init(rms_ffn_w, rms_size);
			rms_final_bo = mem_init(rms_final_w, rms_final_size);
			std::cout<<"Compiled on "<<__DATE__<<" at "<<__TIME__<<".\n";

			mm_cu_kernel.reserve(mm_cu_cnt);
			for (int i = 0 ; i <mm_cu_cnt; i++) {
				mm_cu_kernel.push_back(std::make_unique<MatMultClass>(device, uuid));
			}
//==============================================================================================================================================================
			// rms_kernel = std::make_unique<RMSClass>(device, uuid);
      // swiglu_kernel = std::make_unique<SwigluClass>(device, uuid);
      // mha_kernel = std::make_unique<MHAClass>(device, uuid);
      // rc_kernel = std::make_unique<ResConClass>(device, uuid);
//==============================================================================================================================================================

			size_t token_size = p->dim * sizeof(float);
			size_t hd_token_size = p->hidden_dim * sizeof(float);
			token_bo = xrt::bo(device, token_size, g_id);
			rms_bo = xrt::bo(device, token_size, g_id);
			rc_bo = xrt::bo(device, token_size, g_id);
			q_out = xrt::bo(device, token_size, g_id);
			k_out = xrt::bo(device, token_size, g_id);
			v_out = xrt::bo(device, token_size, g_id);
			o_out = xrt::bo(device, token_size, g_id);
			mha_out = xrt::bo(device, token_size, g_id);
			FFN1_out = xrt::bo(device, hd_token_size, g_id);
			FFN2_out = xrt::bo(device, token_size, g_id);
			FFN3_out = xrt::bo(device, hd_token_size, g_id);
			swiglu_out = xrt::bo(device, hd_token_size, g_id);
			mm_logits = xrt::bo(device, p->vocab_size * sizeof(float), g_id);
			// rc_bo_map = rc_bo.map<float*>();
			// std::fill(rc_bo_map, rc_bo_map + p->dim, 0.0f);
			// rc_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		
			std::cout<<"did I make it here after creating the bo?\n";

			token_bo_map = token_bo.map<float*>();
			mm_logits_map = mm_logits.map<float*>();
			std::cout<<"init complete feb 15 1:010\n";
			// std::cout<<"Compiled on "<<__DATE__<<" at "<<__TIME__<<".\n";
		}
/*===============================================================================================================
RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD 
=====================================================================================================*/
		// float* runForward(const int token, const int pos){
		// 	float *cr = w->token_embedding_table + token * p->dim;
		// 	// memcpy(x, cr, p->dim * sizeof(*x)); //wrong!
		// 	std::memcpy(token_bo_map, cr, p->dim * sizeof(float));
		// 		// std::fill(rc_bo_map, rc_bo_map + p->dim, 0.0f);
		// 		// rc_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

		// 	token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

		// 	for (int l = 0; l < p->n_layers; l++) {
				
		// 		rms_kernel->run(rms_bo, token_bo, rms_att_bo, l);
		// 		// std::cout<<"rms\n";
		// 		int q_off = l * 3;
		// 		int k_off = 1 + q_off;
		// 		int v_off = l + k_off;
		// 		// int v2_off =  p->dim / 2 + v_off;
		// 		//(xrt::bo &output, xrt::bo &input, xrt::bo &sf_bo, xrt::bo &w_bo, const int n_dim, const int m_dim, const int curr_layer)
		// 		auto q_run = mm_cu_kernel[0]->start(q_out, rms_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, q_off);
		// 		auto k_run = mm_cu_kernel[1]->start(k_out, rms_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, k_off);
		// 		auto v_run = mm_cu_kernel[0]->start(v_out, rms_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, v_off);
		// 		// auto v2_run = mm_cu_kernel[1]->start(v_out, rms_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim/2, v2_off, p->dim/2);

		// 		mm_cu_kernel[0]->wait(q_run);
		// 		mm_cu_kernel[1]->wait(k_run);
		// 		mm_cu_kernel[0]->wait(v_run);
		// 		// mm_cu_kernel[1]->wait(v2_run);
				
		// 		// std::cout<<"qkv\n";
		// 		mha_kernel->run(q_out, k_out, v_out, l, pos);
		// 		// std::cout<<"mha\n";
		// 		int o_off = l;
		// 		// int o2_off = 1 + o_off;
				
		// 		// mm_cu_kernel[0]->run(o_out, q_out, out_sf_bo, out_w_bo, p->dim, p->dim, q_off, 0);
		// 		auto o_run = mm_cu_kernel[0]->start(o_out, q_out, out_sf_bo, out_w_bo, p->dim, p->dim, k_off);
		// 		// auto o2_run = mm_cu_kernel[1]->start(o_out, q_out, out_sf_bo, out_w_bo, p->dim, p->dim/2, o2_off, p->dim/2);
		// 		mm_cu_kernel[0]->wait(o_run);
		// 		// mm_cu_kernel[1]->wait(o2_run);
				
		// 		rc_kernel->run(rc_bo, token_bo, o_out);
		// 		rms_kernel->run(rms_bo, rc_bo, rms_ffn_bo, l);
				
		// 		int ffn1_off = l * 2;
		// 		int ffn3_off = 1 + ffn1_off;
		// 		auto ffn1_run = mm_cu_kernel[0]->start(FFN1_out, rms_bo, ffn13_sf_bo, ffn13_w_bo, p->dim, p->hidden_dim, ffn1_off); //FFN1_mm_kernel->start(FFN1_out, o_out, l);
		// 		auto ffn3_run = mm_cu_kernel[1]->start(FFN3_out, rms_bo, ffn13_sf_bo, ffn13_w_bo, p->dim, p->hidden_dim, ffn3_off); //FFN3_mm_kernel->start(FFN3_out, o_out, l);

		// 		mm_cu_kernel[0]->wait(ffn1_run);
		// 		mm_cu_kernel[1]->wait(ffn3_run);
				
		// 		swiglu_kernel->run(swiglu_out, FFN1_out, FFN3_out);
				
		// 		int ffn2_off = l;
		// 		// int ffn22_off = ffn2_off + p->dim/2;
		// 		auto ffn2_run = mm_cu_kernel[0]->start(FFN2_out, swiglu_out, ffn2_sf_bo, ffn2_w_bo, p->hidden_dim, p->dim, ffn2_off);
		// 		// auto ffn22_run = mm_cu_kernel[1]->start(FFN2_out, swiglu_out, ffn2_sf_bo, ffn2_w_bo, p->hidden_dim, p->dim/2, ffn22_off, p->dim/2);
				
		// 		mm_cu_kernel[0]->wait(ffn2_run);
		// 		// mm_cu_kernel[1]->wait(ffn22_run);

		// 		rc_kernel->run(token_bo, rc_bo, FFN2_out);
		// 	}
		// 	rms_kernel->run(rc_bo, token_bo, rms_final_bo, 0);
		// 	auto logit_run = mm_cu_kernel[0]->start(mm_logits, rc_bo, logits_sf_bo, logits_w_bo, p->dim, p->vocab_size, 0);
		// 	// auto logit2_run = mm_cu_kernel[1]->start(mm_logits, rc_bo, logits_sf_bo, logits_w_bo, p->dim, p->vocab_size/2, p->vocab_size/2, p->vocab_size/2);
		// 		mm_cu_kernel[0]->wait(logit_run);
		// 		// mm_cu_kernel[1]->wait(logit2_run);
			
		// 	mm_logits.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 	return mm_logits_map;
		// }


/*===============================================================================================================
RUN BACKWARDS RUN BACKWARDS RUN BACKWARDS RUN BACKWARDS RUN BACKWARDS RUN BACKWARDS RUN BACKWARDS RUN BACKWARDS RUN BACKWARDS RUN BACKWARDS  
=====================================================================================================*/

		// float* runBackward(const int token, const int pos){
		// 	float *cr = w->token_embedding_table + token * p->dim;
		// 	// memcpy(x, cr, p->dim * sizeof(*x)); //wrong!
		// 	std::memcpy(token_bo_map, cr, p->dim * sizeof(float));
		// 		std::fill(rc_bo_map, rc_bo_map + p->hidden_dim, 0.0f);
		// 		rc_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

		// 	token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

		// 	for (int l = 0; l < p->n_layers; l++) {
				
		// 		// rmsnorm_kernel(mm_in, x, mm_out, w->rms_att_weight + l * p->dim, p->dim);
		// 		// rms_att_kernel->run(token_bo, rc_bo, l);
		// 		rms_kernel->run(token_bo, rc_bo, rms_att_bo, l);
		// 		// std::cout<<"rms\n";
		// 		int q_off = p->dim * l * 3;
		// 		int k_off = p->dim + q_off;
		// 		int v_off = p->dim + k_off;
		// 		int v2_off =  p->dim / 2 + v_off;
				
		// 		auto q_run = mm_cu_kernel[0]->start(q_out, token_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, q_off, 0);
		// 		auto k_run = mm_cu_kernel[1]->start(k_out, token_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, k_off, 0);
		// 		auto v_run = mm_cu_kernel[0]->start(v_out, token_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim/2, v_off, 0);
		// 		auto v2_run = mm_cu_kernel[1]->start(v_out, token_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim/2, v2_off, p->dim/2);

		// 		mm_cu_kernel[0]->wait(q_run);
		// 		mm_cu_kernel[1]->wait(k_run);
		// 		mm_cu_kernel[0]->wait(v_run);
		// 		mm_cu_kernel[1]->wait(v2_run);

		// 		q_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 		k_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 		v_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
				
		// 		sw_mha(mha_out.map<float*>(), q_out.map<float*>(), k_out.map<float*>(), v_out.map<float*>(), l, pos);
		// 		// std::cout<<"qkv\n";
		// 		// mha_kernel->run(q_out, k_out, v_out, l, pos);
		// 		// std::cout<<"mha\n";
		// 		int o_off = p->dim * l;
		// 		int o2_off = p->dim / 2 + o_off;
		// 		mha_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
				
		// 		// mm_cu_kernel[0]->run(o_out, q_out, out_sf_bo, out_w_bo, p->dim, p->dim, q_off, 0);
		// 		auto o_run = mm_cu_kernel[0]->start(o_out, mha_out, out_sf_bo, out_w_bo, p->dim, p->dim/2, o_off, 0);
		// 		auto o2_run = mm_cu_kernel[1]->start(o_out, mha_out, out_sf_bo, out_w_bo, p->dim, p->dim/2, o2_off, p->dim/2);
		// 		mm_cu_kernel[0]->wait(o_run);
		// 		mm_cu_kernel[1]->wait(o2_run);
		// 		// out_mm_kernel->run(o_out, mha_out, l);
		// 		// std::cout<<"out\n";
		// 		// rms_ffn_kernel->run(o_out, rc_bo, l);
		// 		rms_kernel->run(o_out, rc_bo, rms_ffn_bo, l);
		// 		// rmsnorm_kernel(mm_in, x, mm_out, w->rms_ffn_weight + l* p->dim, p->dim);
		// 		// std::cout<<"rms\n";
		// 		int ffn1_off = p->hidden_dim * l * 2;
		// 		int ffn3_off = p->hidden_dim + ffn1_off;
		// 		auto ffn1_run = mm_cu_kernel[0]->start(FFN1_out, o_out, ffn13_sf_bo, ffn13_w_bo, p->dim, p->hidden_dim, ffn1_off, 0); //FFN1_mm_kernel->start(FFN1_out, o_out, l);
		// 		auto ffn3_run = mm_cu_kernel[1]->start(FFN3_out, o_out, ffn13_sf_bo, ffn13_w_bo, p->dim, p->hidden_dim, ffn3_off, 0); //FFN3_mm_kernel->start(FFN3_out, o_out, l);

		// 		mm_cu_kernel[0]->wait(ffn1_run);
		// 		mm_cu_kernel[1]->wait(ffn3_run);
				
		// 		// FFN1_mm_kernel->wait(ffn1_run);
		// 		// FFN3_mm_kernel->wait(ffn3_run);
		// 		// std::cout<<"FFN1\n";
		// 		swiglu_kernel->run(swiglu_out, FFN1_out, FFN3_out);
		// 		// swiglu_kernel->run(mm_in, mm_out);
		// 		// std::cout<<"swiglu\n";
		// 		// FFN2_mm_kernel->run(token_bo, swiglu_out, l);
		// 		int ffn2_off = p->dim * l;
		// 		int ffn22_off = ffn2_off + p->dim/2;
		// 		// mm_cu_kernel[0]->run(token_bo, swiglu_out, ffn2_sf_bo, ffn2_w_bo, p->hidden_dim, p->dim, ffn1_off, 0); //FFN2_mm_kernel->start(token_bo, swiglu_out, l);
		// 		auto ffn2_run = mm_cu_kernel[0]->start(token_bo, swiglu_out, ffn2_sf_bo, ffn2_w_bo, p->hidden_dim, p->dim/2, ffn2_off, 0);
		// 		auto ffn22_run = mm_cu_kernel[1]->start(token_bo, swiglu_out, ffn2_sf_bo, ffn2_w_bo, p->hidden_dim, p->dim/2, ffn22_off, p->dim/2);
				
		// 		// FFN2_mm_kernel->wait(ffn2_run);
		// 		mm_cu_kernel[0]->wait(ffn2_run);
		// 		mm_cu_kernel[1]->wait(ffn22_run);
				
		// 		// std::cout<<"FF2\n";
		// 	}
		// 	// std::cout<<"loop exit\n";
		// 	rms_kernel->run(token_bo, rc_bo, rms_final_bo, 0);
		// 	// rms_final_kernel->run(token_bo, rc_bo, 0);
		// 	// rmsnorm_kernel(mm_in, x, mm_out, w->rms_final_weight, p->dim);
		// 		// std::cout<<"rms\n";
		// 	// logits_kernel->run(mm_logits, token_bo, 0);
		// 	// mm_cu_kernel[0]->run(mm_logits, token_bo, logits_sf_bo, logits_w_bo, p->dim, p->vocab_size, 0, 0);
		// 	auto logit_run = mm_cu_kernel[0]->start(mm_logits, token_bo, logits_sf_bo, logits_w_bo, p->dim, p->vocab_size/2, 0, 0);
		// 	auto logit2_run = mm_cu_kernel[1]->start(mm_logits, token_bo, logits_sf_bo, logits_w_bo, p->dim, p->vocab_size/2, p->vocab_size/2, p->vocab_size/2);
		// 		mm_cu_kernel[0]->wait(logit_run);
		// 		mm_cu_kernel[1]->wait(logit2_run);
			
		// 	mm_logits.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 	return mm_logits_map;
		// }

		// float* mha_speedup(float* q, float* k, float* v, const int l, const int pos){
		// 	std::memcpy(q_out.map<float*>(), q, p->dim * sizeof(float));
		// 	std::memcpy(k_out.map<float*>(), k, p->dim * sizeof(float));
		// 	std::memcpy(v_out.map<float*>(), v, p->dim * sizeof(float));
		// 	// rc_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

		// 	// rms_kernel->run(rc_bo, token_bo, rms_att_bo, l);
		// 	// std::memcpy(k_out.map<float*>(), k, p->dim * sizeof(float));
		// 	// std::memcpy(v_out.map<float*>(), v, p->dim * sizeof(float));
			
		// 	// k_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		// 	// v_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			
		// 	// int q_off = p->dim * l * 3;
		// 	// int k_off = p->dim + q_off;
		// 	// int v_off = p->dim + k_off;
		// 	// int v2_off =  p->dim / 2 + v_off;
			
		// 	// auto q_run = mm_cu_kernel[0]->start(q_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, q_off, 0);
		// 	// auto k_run = mm_cu_kernel[1]->start(k_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, k_off, 0);
		// 	// auto v_run = mm_cu_kernel[0]->start(v_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, v_off, 0);
		// 	// auto v2_run = mm_cu_kernel[1]->start(v_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim/2, v2_off, p->dim/2);
		// 	// int o_off = p->dim * l;
		// 	// int o2_off = p->dim / 2 + o_off;

		// 	// mm_cu_kernel[0]->wait(q_run);
		// 	// mm_cu_kernel[1]->wait(k_run);
		// 	// mm_cu_kernel[0]->wait(v_run);
		// 	// mm_cu_kernel[1]->wait(v2_run);

		// 	q_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 	k_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 	v_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

			
		// 	mha_kernel->run(q_out, k_out, v_out, l, pos);

		// 	// auto o_run = mm_cu_kernel[0]->start(o_out, q_out, out_sf_bo, out_w_bo, p->dim, p->dim/2, o_off, 0);
		// 	// auto o2_run = mm_cu_kernel[1]->start(o_out, q_out, out_sf_bo, out_w_bo, p->dim, p->dim/2, o2_off, p->dim/2);
		// 	// mm_cu_kernel[0]->wait(o_run);
		// 	// mm_cu_kernel[1]->wait(o2_run);
			
		// 	// rc_kernel->run(token_bo, o_out);
		// 	// rms_kernel->run(rms_bo, token_bo, rms_ffn_bo, l);
		// 	q_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 	// token_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    //   //   for (int i = 0; i < p->dim; i++) {
    //   //       token_bo.map<float*>()[i] += o_out.map<float*>()[i];
    //   //   }

		// 	// token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		// 	// rms_kernel->run(rms_bo, token_bo, rms_ffn_bo, l);
				
		// 	// int ffn1_off = p->hidden_dim * l * 2;
		// 	// int ffn3_off = p->hidden_dim + ffn1_off;
		// 	// auto ffn1_run = mm_cu_kernel[0]->start(FFN1_out, rms_bo, ffn13_sf_bo, ffn13_w_bo, p->dim, p->hidden_dim, ffn1_off, 0); //FFN1_mm_kernel->start(FFN1_out, o_out, l);
		// 	// auto ffn3_run = mm_cu_kernel[1]->start(FFN3_out, rms_bo, ffn13_sf_bo, ffn13_w_bo, p->dim, p->hidden_dim, ffn3_off, 0); //FFN3_mm_kernel->start(FFN3_out, o_out, l);

		// 	// mm_cu_kernel[0]->wait(ffn1_run);
		// 	// mm_cu_kernel[1]->wait(ffn3_run);
			
		// 	// swiglu_kernel->run(swiglu_out, FFN1_out, FFN3_out);
		// 	q_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
			
		// 	return q_out.map<float*>();
		// }

		// void rms_slow(float * token, float *output, const int l){
		// 	std::memcpy(token_bo.map<float*>(), token, p->dim * sizeof(float));
		// 	token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			
		// 	rms_kernel->run(rc_bo, token_bo, rms_att_bo, l);
		// 	rc_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 	std::memcpy(output, rc_bo.map<float*>(), p->dim * sizeof(float));
		// }
		// void rmsf_slow(float * token, float *output, const int l){
		// 	std::memcpy(token_bo.map<float*>(), token, p->dim * sizeof(float));
		// 	token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			
		// 	rms_kernel->run(rc_bo, token_bo, rms_ffn_bo, l);
		// 	rc_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 	std::memcpy(output, rc_bo.map<float*>(), p->dim * sizeof(float));
		// }

		// void rmsfnl_slow(float * token, float *output, const int l){
		// 	std::memcpy(token_bo.map<float*>(), token, p->dim * sizeof(float));
		// 	token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			
		// 	rms_kernel->run(rc_bo, token_bo, rms_final_bo, l);
		// 	rc_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 	std::memcpy(output, rc_bo.map<float*>(), p->dim * sizeof(float));
		// }
		/*============================================================================================================
		
		MATMULT MATMULT MATMULT MATMULT MATMULT MATMULT MATMULT MATMULT MATMULT MATMULT MATMULT MATMULT MATMULT MATMULT MATMULT 
		
		============================================================================================================
		*/
		void mmq_slow(float *xb, float *output, const int l){
			std::memcpy(rc_bo.map<float*>(), xb, p->dim * sizeof(float));
			rc_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			int q_off = p->dim * l * 3;
			mm_cu_kernel[0]->run(q_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, l * 3);
			q_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			std::memcpy(output, q_out.map<float*>(), p->dim * sizeof(float));
		}
		void mmkv_slow(float *xb, float *outputk, float *outputv, const int l){
			std::memcpy(rc_bo.map<float*>(), xb, p->dim * sizeof(float));
			rc_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			int q_off = p->dim * l * 3;
			int k_off = p->dim + q_off;
			int v_off = p->dim + k_off;
			// mm_cu_kernel[0]->run(q_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, q_off, 0);
			auto k_run = mm_cu_kernel[0]->start(k_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, (l* 3 + 1));
			mm_cu_kernel[0]->wait(k_run);
			auto v_run = mm_cu_kernel[0]->start(v_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, (l* 3 + 2));
			mm_cu_kernel[0]->wait(v_run);
			k_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			v_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			std::memcpy(outputv, v_out.map<float*>(), p->dim * sizeof(float));
			std::memcpy(outputk, k_out.map<float*>(), p->dim * sizeof(float));
		}
		void mmo_slow(float *xb, float *output, const int l){
			std::memcpy(rc_bo.map<float*>(), xb, p->dim * sizeof(float));
			rc_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			int q_off = p->dim * l * 3;
				int o_off = p->dim * l;
				int o2_off = p->dim / 2 + o_off;
			mm_cu_kernel[0]->run(o_out, rc_bo, out_sf_bo, out_w_bo, p->dim, p->dim, l);
			o_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			std::memcpy(output, o_out.map<float*>(), p->dim * sizeof(float));
		}
		// void mmo2_slow(float *xb, float *output, const int l){
		// 	std::memcpy(rc_bo.map<float*>(), xb, p->dim * sizeof(float));
		// 	rc_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		// 	int q_off = p->dim * l * 3;
		// 	int o_off = p->dim * l;
		// 	int o2_off = p->dim / 2 + o_off;
		// 	// mm_cu_kernel[0]->run(q_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, q_off, 0);
		// 	auto k_run = mm_cu_kernel[1]->start(o_out, rc_bo, out_sf_bo, out_w_bo, p->dim, p->dim/2, o_off, 0);
		// 	auto v_run = mm_cu_kernel[0]->start(o_out, rc_bo, out_sf_bo, out_w_bo, p->dim, p->dim/2, o2_off, p->dim/2);
		// 	mm_cu_kernel[1]->wait(k_run);
		// 	mm_cu_kernel[0]->wait(v_run);
		// 	o_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		// 	std::memcpy(output, o_out.map<float*>(), p->dim * sizeof(float));
		// }
		// void mml2_slow(float *xb, float *output, const int l){
		// 	std::memcpy(rc_bo.map<float*>(), xb, p->dim * sizeof(float));
		// 	rc_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		// 	int q_off = p->dim * l * 3;
		// 	int o_off = p->dim * l;
		// 	int o2_off = p->dim / 2 + o_off;
		// 	// mm_cu_kernel[0]->run(q_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, q_off, 0);
		// 	auto k_run = mm_cu_kernel[1]->start(mm_logits, rc_bo, logits_sf_bo, logits_w_bo, p->dim, p->vocab_size/2, 0, 0);
		// 	auto v_run = mm_cu_kernel[0]->start(mm_logits, rc_bo, logits_sf_bo, logits_w_bo, p->dim, p->vocab_size/2, p->vocab_size/2, p->vocab_size/2);
		// 	mm_cu_kernel[1]->wait(k_run);
		// 	mm_cu_kernel[0]->wait(v_run);
		// 	mm_logits.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		// 	std::memcpy(output, mm_logits.map<float*>(), p->dim * sizeof(float));;
		// }
		void mml_slow(float *xb, float *output, const int l){
			std::memcpy(rc_bo.map<float*>(), xb, p->dim * sizeof(float));
			rc_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			// mm_cu_kernel[0]->run(q_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, q_off, 0);
			mm_cu_kernel[0]->run(mm_logits, rc_bo, logits_sf_bo, logits_w_bo, p->dim, p->vocab_size, 0);
			// auto v_run = mm_cu_kernel[0]->start(o_out, rc_bo, logits_sf_bo, logits_w_bo, p->dim, p->vocab_size/2, p->vocab_size/2, p->vocab_size/2);
			// mm_cu_kernel[1]->wait(k_run);
			// mm_cu_kernel[0]->wait(v_run);
			mm_logits.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
			std::memcpy(output, mm_logits.map<float*>(), p->vocab_size * sizeof(float));;
		}

		void mmffn2_slow(float *xb, float *output, const int l){
			std::memcpy(swiglu_out.map<float*>(), xb, p->hidden_dim * sizeof(float));
			swiglu_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			int ffn2_off = p->dim * l;
			mm_cu_kernel[0]->run(FFN2_out, swiglu_out, ffn2_sf_bo, ffn2_w_bo, p->hidden_dim, p->dim, l);
			FFN2_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
			std::memcpy(output, FFN2_out.map<float*>(), p->dim * sizeof(float));
		}

		void mmffn13_slow(float *xb, float *w1, float *w3, const int l){
			std::memcpy(swiglu_out.map<float*>(), xb, p->dim * sizeof(float));
			swiglu_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			// int ffn2_off = p->dim * l;
			auto w1_run = mm_cu_kernel[0]->start(FFN1_out, swiglu_out, ffn1_sf_bo, ffn1_w_bo, p->dim, p->hidden_dim, l);
			auto w3_run = mm_cu_kernel[1]->start(FFN3_out, swiglu_out, ffn3_sf_bo, ffn3_w_bo, p->dim, p->hidden_dim, l);
			mm_cu_kernel[1]->wait(w1_run);
			mm_cu_kernel[0]->wait(w3_run);
			FFN1_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
			FFN3_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
			std::memcpy(w1, FFN1_out.map<float*>(), p->hidden_dim * sizeof(float));
			std::memcpy(w3, FFN3_out.map<float*>(), p->hidden_dim * sizeof(float));
		}


		// void swiglu_speedup(float* x, float* hb, float* hb2){
		// 	// std::memcpy(rc_bo.map<float*>(), xb, p->dim * sizeof(float));
		// 	// std::memcpy(token_bo.map<float*>(), x, p->dim * sizeof(float));
		// 	// // rc_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		// 	// token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		// 	// // rc_kernel->run(token_bo, rc_bo);
		// 	// rms_kernel->run(rms_bo, token_bo, rms_ffn_bo, l);
				
		// 	// int ffn1_off = p->hidden_dim * l * 2;
		// 	// int ffn3_off = p->hidden_dim + ffn1_off;
		// 	// auto ffn1_run = mm_cu_kernel[0]->start(FFN1_out, rms_bo, ffn13_sf_bo, ffn13_w_bo, p->dim, p->hidden_dim, ffn1_off, 0); //FFN1_mm_kernel->start(FFN1_out, o_out, l);
		// 	// auto ffn3_run = mm_cu_kernel[1]->start(FFN3_out, rms_bo, ffn13_sf_bo, ffn13_w_bo, p->dim, p->hidden_dim, ffn3_off, 0); //FFN3_mm_kernel->start(FFN3_out, o_out, l);
			

		// 	// mm_cu_kernel[0]->wait(ffn1_run);
		// 	// mm_cu_kernel[1]->wait(ffn3_run);
			
				
		// 	// int ffn2_off = p->dim * l;
		// 	// int ffn22_off = ffn2_off + p->dim/2;
		// 	// auto ffn2_run = mm_cu_kernel[0]->start(FFN2_out, swiglu_out, ffn2_sf_bo, ffn2_w_bo, p->hidden_dim, p->dim/2, ffn2_off, 0);
		// 	// auto ffn22_run = mm_cu_kernel[1]->start(FFN2_out, swiglu_out, ffn2_sf_bo, ffn2_w_bo, p->hidden_dim, p->dim/2, ffn22_off, p->dim/2);
			
		// 	// mm_cu_kernel[0]->wait(ffn2_run);
		// 	// mm_cu_kernel[1]->wait(ffn22_run);

		// 	// rc_kernel->run(token_bo, FFN2_out);
		// 	// FFN2_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 	std::memcpy(FFN1_out.map<float*>(), hb, p->hidden_dim * sizeof(float));
		// 	std::memcpy(FFN3_out.map<float*>(), hb2, p->hidden_dim * sizeof(float));
		// 	FFN1_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		// 	FFN3_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			
		// 	swiglu_kernel->run(swiglu_out, FFN1_out, FFN3_out);
		// 	swiglu_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 	std::memcpy(x, swiglu_out.map<float*>(), p->hidden_dim * sizeof(float));

			
		// }

		/* 
		=========================================================================================
		PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE 
		==============================================================================================
		*/
	private:
		xrt::device device;
		xrt::uuid uuid;

		xrt::bo qkv_w_bo, qkv_sf_bo, out_w_bo, out_sf_bo, ffn1_w_bo, ffn1_sf_bo, ffn3_w_bo, ffn3_sf_bo, ffn2_w_bo, ffn2_sf_bo, logits_w_bo, logits_sf_bo;
		xrt::bo rms_att_bo, rms_ffn_bo, rms_final_bo;

		std::vector<std::unique_ptr<MatMultClass>> mm_cu_kernel;
		std::unique_ptr<RMSClass> rms_kernel;
		std::unique_ptr<SwigluClass>  swiglu_kernel;
		std::unique_ptr<MHAClass> mha_kernel;
		std::unique_ptr<ResConClass> rc_kernel;


		Config *p;
		RunState *s;
		TransformerWeights *w;
		// float *x;
		int g_id; 

		xrt::bo token_bo;
		xrt::bo rc_bo;
		xrt::bo q_out;
		xrt::bo k_out;
		xrt::bo v_out;
		xrt::bo o_out;
		xrt::bo mha_out;
		xrt::bo FFN1_out;
		xrt::bo FFN3_out;
		xrt::bo mm_logits;
		xrt::bo swiglu_out;
		xrt::bo FFN2_out;
		xrt::bo rms_bo;

		float* token_bo_map;
		float* rc_bo_map;
		float* mm_logits_map;
		
		
		xrt::bo mem_init(std::string &sw, int w_size){
			//create the containers for the weights
			xrt::bo w_bo = xrt::bo(device, w_size, g_id);
			//handles for the files that have the weights
			std::ifstream w_dat (sw, std::ios::binary);
			//check if we can read, throw error otherwise
			if (!w_dat.is_open()) {
				throw std::runtime_error(sw + ". Try again, but do it right this time.");
			}
			//map contents of bo to host mem
			auto w_bo_map = w_bo.map<char*>();
			w_dat.read(reinterpret_cast<char*>(w_bo_map), w_bo.size());
			w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			return w_bo;
		}


};

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x =(float*) calloc(p->dim, sizeof(float));
    s->xb = (float*) calloc(p->dim, sizeof(float));
    s->xb2 = (float*) calloc(p->dim, sizeof(float));
    s->hb = (float*) calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*) calloc(p->hidden_dim, sizeof(float));
    s->q = (float*) calloc(p->dim, sizeof(float));
    s->key_cache = (float*) calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*) calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = (float*) calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*) calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data =(float *) mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
	}
void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos, ForwardBlock &f) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
				// f.rms_slow(x, s->xb, l);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        // matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
				f.mmq_slow(s->xb, s->q, l);
				
        // matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        // matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
				f.mmkv_slow(s->xb, s->k, s->v, l);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

				

				// s->xb = f.mha_speedup(s->q, s->k, s->v, l, pos);
        // final matmul to get the output of the attention
        // matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
				f.mmo_slow(s->xb, s->xb2, l);
				// f.mmo2_slow(s->xb, s->xb2, l);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

				// f.swiglu_speedup(x, s->hb, s->hb2, l);

        // // // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
				// f.rmsf_slow(s->x, s->xb, l)	;

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        // matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        // matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
				f.mmffn13_slow(s->xb, s->hb, s->hb2, l);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }
				// s->xb = f.swiglu_speedup(s->xb, l, pos);
				// f.swiglu_speedup(s->hb, s->hb, s->hb2);

        // final matmul to get the output of the ffn
        // matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
				f.mmffn2_slow(s->hb, s->xb, l);

				// s->xb2 = f.swiglu_speedup(s->xb, l, pos);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }

    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);
		// f.rmsfnl_slow(x, s->xb, 0);

    // classifier into logits
    // matmul(s->logits, s->x, w->wcls, p->dim, p->vocab_size);
		f.mml_slow(x, s->logits, 0);
		// f.mml2_slow(s->xb, s->logits, 0);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*) bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex*) malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char*) malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex *) malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps, ForwardBlock &f) {
	//  std::cout << "inside generate" << std::endl << std::flush;
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos, f);
				// float* logits = f.runForward(token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps, ForwardBlock &f) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos, f);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> <xclbin> <device_id> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  -d <string> (REQUIRED) device ID\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
		// char *device_id = 0;
		// char *xclbin_file = NULL;
		std::string xclbin_file;
		std::string device_id = "0";

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 3) { checkpoint_path = argv[1]; xclbin_file = argv[2];} else { error_usage(); }
    for (int i = 3; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'd') { device_id = argv[i + 1]; }
        else { error_usage(); }
    }
	std::cout<< "parsed"<<std::endl;
    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

	std::cout<< "forward block start"<<std::endl;
		//init ForwardBlock
		int d = std::stoi(device_id);
		std::string qkv_w = "qkv_w_comb.bin";
		std::string qkv_sf = "qkv_sf_comb.bin";
		std::string out_w = "output_w.bin";
		std::string out_sf = "output_sf.bin";
		std::string w1_w = "MLP_1_q.bin";
		std::string w1_sf = "MLP_1_sf.bin";
		std::string w3_w = "MLP_3_q.bin";
		std::string w3_sf = "MLP_3_sf.bin";
		std::string w2_w = "FFN2_w.bin";
		std::string w2_sf = "FFN2_sf.bin";
		std::string logits_w = "wcls_w.bin";
		std::string logits_sf = "wcls_sf.bin";
		std::string att_w = "rms_att_w.bin";
		std::string ffn_w = "rms_ffn_w.bin";
		std::string final_w = "rms_final_w.bin";
		ForwardBlock f = ForwardBlock(d, xclbin_file, 
																	qkv_w, qkv_sf, 
																	out_w, out_sf, 
																	w1_w, w1_sf, 
																	w3_w, w3_sf, 
																	w2_w, w2_sf, 
																	logits_w, logits_sf, 
																	att_w, ffn_w, final_w, 
																	&transformer, 2);

	// std::cout<< "forward block done  ?"<<mode<<std::endl;
    // run!
std::cout << "forward block done  ?" << mode << std::endl;
std::cout << "transformer.config.vocab_size = " << transformer.config.vocab_size << std::endl;
std::cout << "transformer.weights ptr = " << (void*)transformer.weights.token_embedding_table << std::endl;
std::cout << "prompt = " << (prompt ? prompt : "null") << std::endl;
std::cout << "steps = " << steps << std::endl;
std::cout << std::flush;
generate(&transformer, &tokenizer, &sampler, prompt, steps, f);
    // if (strcmp(mode, "generate") == 0) {
        // generate(&transformer, &tokenizer, &sampler, prompt, steps, f);
    // } else if (strcmp(mode, "chat") == 0) {
    //     chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps, f);
    // } else {
    //     fprintf(stderr, "unknown mode: %s\n", mode);
    //     error_usage();
    // }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
