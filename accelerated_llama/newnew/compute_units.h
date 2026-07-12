// XRT includes
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "experimental/xrt_ip.h"
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <cstring>
#include "../transformer_cu/transformer_cu/hls/impl/misc/drivers/transformer_cu_v1_0/src/xtransformer_cu_hw.h"



typedef struct {
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
} axi_reg;

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
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float* token_embedding_table; // same, but dequantized

    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
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
		// ForwardBlock(
		// 	int device_id, std::string& binaryFile, const std::string &checkpoint, Transformer *t){
				
		// 	try {
		// 		device = xrt::device(device_id);
		// 		std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
    //  		std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";
		// 		uuid = device.load_xclbin(binaryFile);
		// 		kernel = xrt::kernel(device, uuid, "transformer_cu");
				
		// 	} catch (const std::exception& e) {
		// 		throw std::runtime_error(std::string(e.what()));
		// 	}

		// 	allocate_cache_init();
		// 	weights_init(checkpoint);
		// 	run_init();
			
		// 	p = &t->config;
		// 	w = &t->weights;
		// 	s = &t->state;
		// 	std::cout<<"did I test it here?\n";

		// }
		ForwardBlock(
			int device_id, std::string& binaryFile, const std::string &checkpoint, Transformer *t){
				
			try {
				device = xrt::device(device_id);
				std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
     		std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";
				uuid = device.load_xclbin(binaryFile);
				// kernel = xrt::kernel(device, uuid, "transformer_cu");
				transformer_ip = xrt::ip(device, uuid, "transformer_cu");
				
			} catch (const std::exception& e) {
				throw std::runtime_error(std::string(e.what()));
			}

			allocate_cache_init();
			weights_init(checkpoint);
			run_ip_init();
			
			p = &t->config;
			w = &t->weights;
			s = &t->state;
			std::cout<<"did I test it here?\n";

		}

		
/*===============================================================================================================
RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD 
=====================================================================================================*/
		// float* runForward(const int token, const int pos){
		// 	float *cr = w->token_embedding_table + token * p->dim;
		// 	std::memcpy(token_map_c, cr, p->dim * sizeof(float));

		// 	token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		// 	transformer_run.set_arg(8, pos);
		// 	transformer_run.start();
		// 	transformer_run.wait();
			
		// 	token_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		// 	return token_map_f;
		// }

		void startForward(const int token, const int pos, const float coin){
			// float *cr = w->token_embedding_table + token * p->dim;
			std::memcpy(token_map_c, w->token_embedding_table + token * p->dim, p->dim * sizeof(float));

			float tmm_val = coin;
			uint32_t tmp_bits;
			std::memcpy(&tmp_bits, &tmm_val, sizeof(float));
			
			token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_POS_R_DATA,pos);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_COIN_DATA,tmp_bits);
			transformer_ip.write_register(0x00, 0x01);
		}

		int endForward(){
			int timeout_cntr = 0;
			const int MAX_TIMEOUT = 2000; //20 milliseconds, right?
			int status = 0;

			while ((status & 0x02) == 0) {
				status = transformer_ip.read_register(0x00);
				
				if (timeout_cntr++ > MAX_TIMEOUT) {
					throw std::runtime_error("FPGA Frozen. FUCK.");
					// break;
				}
				
				std::this_thread::sleep_for(std::chrono::microseconds(10));
			}
			return transformer_ip.read_register(XTRANSFORMER_CU_CONTROL_ADDR_PICK_DATA);
			
		}

		/* 
		=========================================================================================
		PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE 
		==============================================================================================
		*/
	private:
		xrt::device device;
		xrt::uuid uuid;
		xrt::kernel kernel;
		xrt::run transformer_run; // is this legal?
		xrt::ip transformer_ip;

		// uint32_t status = 0;
		
		xrt::bo parent_rms_bo;
		xrt::bo parent_w_bo;
		xrt::bo parent_sf_bo;
		xrt::bo key_cache_bo;
		xrt::bo value_cache_bo;
		
		xrt::bo qkv_w_bo;
		xrt::bo qkv_sf_bo;
		xrt::bo out_w_bo;
		xrt::bo out_sf_bo;
		xrt::bo gate_up_w_bo;
		xrt::bo gate_up_sf_bo;
		xrt::bo down_w_bo;
		xrt::bo down_sf_bo;
		xrt::bo embedding_w_bo;
		xrt::bo embedding_sf_bo;
		xrt::bo rms_att_w_bo, rms_ffn_w_bo, rms_final_w_bo;
		xrt::bo token_bo;
		float* token_map_f;
		char* token_map_c;
		

		int MODEL_ELEMENTS = 768;
		int MODEL_HIDDEN_DIM = 2048;
		int MODEL_SCALING_FACTOR = 64;
		int MODEL_SEQUENCE_LEN = 1024;
		int MODEL_NUM_LAYERS = 12;
		int MODEL_TOKENS = 32000;

		axi_reg tt;
		
		Config* p;
		TransformerWeights* w;
		RunState* s;

		//private method to store the KV cache objects.
		void allocate_cache_init(){
				//create the containers for the weights
			size_t c_size = (size_t)MODEL_ELEMENTS * MODEL_SEQUENCE_LEN * MODEL_NUM_LAYERS * sizeof(float);
			key_cache_bo = xrt::bo(device, c_size, 0);
			value_cache_bo = xrt::bo(device, c_size, 0);
		}

		void weights_init(const std::string &checkpoint){
			
			//init token_bo
			token_bo = xrt::bo(device, MODEL_TOKENS * sizeof(float), 0);
			token_map_f = token_bo.map<float*>();
			token_map_c = token_bo.map<char*>();
			std::ifstream file(checkpoint, std::ios::binary);
			// size_t file_size = file.tellg();
			// file.seekg(0, std::ios::beg);
			
			size_t nn_size = MODEL_ELEMENTS * MODEL_ELEMENTS;
			size_t nm_size = MODEL_ELEMENTS * MODEL_HIDDEN_DIM;
			
			size_t nn_sf_size = nn_size * sizeof(float) / MODEL_SCALING_FACTOR;
			size_t nm_sf_size = nm_size * sizeof(float) / MODEL_SCALING_FACTOR;
			
			size_t rms_att_size = (MODEL_ELEMENTS * MODEL_NUM_LAYERS * sizeof(float));
			size_t rms_ffn_size = rms_att_size;
			size_t rms_final_size = MODEL_ELEMENTS * sizeof(float);
			
			size_t embed_size = MODEL_ELEMENTS * MODEL_TOKENS * sizeof(int8_t);
			size_t embed_sf_size = MODEL_ELEMENTS * MODEL_TOKENS * sizeof(float) / MODEL_SCALING_FACTOR;

			size_t q_size = (MODEL_ELEMENTS * ((MODEL_ELEMENTS * 4 + MODEL_HIDDEN_DIM * 3 ) * MODEL_NUM_LAYERS + MODEL_TOKENS)) * sizeof(int8_t);
			size_t rms_size = (MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1)) * sizeof(float);
			size_t sf_size = (q_size * sizeof(float) / (sizeof(int8_t) * MODEL_SCALING_FACTOR));
			
			parent_rms_bo = xrt::bo(device, q_size, 0);
			parent_w_bo = xrt::bo(device, rms_size, 0);
			parent_sf_bo = xrt::bo(device, sf_size, 0);
			
			
			char* q_ptr = parent_w_bo.map<char*>();
			char* sf_ptr = parent_sf_bo.map<char*>();
			char* rms_ptr = parent_rms_bo.map<char*>();

			size_t file_ptr = 256;
			size_t rms_idx = 0;
			file.seekg(file_ptr, std::ios::beg);
			
			tt.rms_att_W = 0;
			file.read(rms_ptr + rms_idx, rms_att_size);
			rms_idx += rms_att_size;
			
			tt.rms_ffn_W = tt.rms_att_W + rms_att_size;
			file.read(rms_ptr + rms_idx, rms_ffn_size);
			rms_idx += rms_ffn_size;
			
			tt.rms_final_W = tt.rms_ffn_W + rms_ffn_size;
			file.read(rms_ptr + rms_idx, rms_final_size);
			file_ptr = file.tellg();
			
			size_t q_idx = 0;
			size_t sf_idx = 0;
			
			tt.Embed_W = 0;
			file.read(q_ptr + q_idx, embed_size);
			q_idx += embed_size;
			
			tt.Embed_sf_W = 0;
			file.read(sf_ptr + sf_idx, embed_sf_size);
			sf_idx += embed_sf_size;
			
			// read QKV
			tt.QKV_sf_W = sf_idx;
			tt.QKV_W = q_idx;
			
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
			
			tt.Out_sf_W = sf_idx;
			tt.Out_W = q_idx;
			//already at Output
			for (int i = 0; i < MODEL_NUM_LAYERS; i++) {
				
				file.read(q_ptr + q_idx, nn_size);
				q_idx += nn_size;
				file.read(sf_ptr + sf_idx, nn_sf_size);
				sf_idx += nn_sf_size;
			}
			file_ptr = file.tellg();
			
			tt.FF_w1w3_sf_W = sf_idx;
			tt.FF_w1w3_W = q_idx;
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

			tt.FF_w2_W = q_idx;
			tt.FF_w2_sf_W = sf_idx;
			file.seekg(file_ptr, std::ios::beg);
			for (int i = 0; i < MODEL_NUM_LAYERS; i++) {
				
				file.read(q_ptr + q_idx, nm_size);
				q_idx += nm_size;
				file.read(sf_ptr + sf_idx, nm_sf_size);
				sf_idx += nm_sf_size;
			}

			tt.N_DIM = MODEL_ELEMENTS;
			tt.M_DIM = MODEL_ELEMENTS;

			parent_rms_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			parent_sf_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			parent_w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			file.close();
		}

		void run_init(){

			/*
			void transformer_cu(
				fdata_v_t *tokens,
				fdata_v_t *w_sf_0, 
				idata_v_t *w_0, 
				fdata_v_t *w_sf_1, 
				idata_v_t *w_1, 
				fdata_v_t *weights, 
				mfdata_v_t *key_cache, 
				mfdata_v_t *value_cache, 
				const int POS,
				const int QKV_W, 
				const int QKV_sf_W,
				const int Out_W, 
				const int Out_sf_W,
				const int FF_w1w3_W, 
				const int FF_w1w3_sf_W,
				const int FF_w2_W, 
				const int FF_w2_sf_W, 
				const int Embed_W, 
				const int Embed_sf_W, 
				const int rms_att_W, 
				const int rms_ffn_W, 
				const int rms_final_W,
			#ifdef __DEBUG__
				const int faker ,const int INIT, const int CURR_LAYER, const int NEXT_STATE,
			#endif
				const float temperature, 
				const float coin, 
				int* pick
			)
				*/
			
			transformer_run = xrt::run(kernel);
			transformer_run.set_arg(0, token_bo);
			transformer_run.set_arg(1, parent_sf_bo);
			transformer_run.set_arg(2, parent_w_bo);
			transformer_run.set_arg(3, parent_sf_bo);
			transformer_run.set_arg(4, parent_w_bo);
			transformer_run.set_arg(5, parent_rms_bo);
			transformer_run.set_arg(6, key_cache_bo);
			transformer_run.set_arg(7, value_cache_bo);
			// transformer_run.set_arg(8, );
			transformer_run.set_arg(9,tt.QKV_W);
			transformer_run.set_arg(10,tt.QKV_sf_W);
			transformer_run.set_arg(11,tt.Out_W);
			transformer_run.set_arg(12,tt.Out_sf_W);
			transformer_run.set_arg(13,tt.FF_w1w3_W);
			transformer_run.set_arg(14,tt.FF_w1w3_sf_W);
			transformer_run.set_arg(15,tt.FF_w2_W);
			transformer_run.set_arg(16,tt.FF_w2_sf_W);
			transformer_run.set_arg(17,tt.Embed_W);
			transformer_run.set_arg(18,tt.Embed_sf_W);
			transformer_run.set_arg(19,tt.rms_att_W);
			transformer_run.set_arg(20,tt.rms_ffn_W);
			transformer_run.set_arg(21,tt.rms_final_W);
			transformer_run.set_arg(22, 0.9f);
			// transformer_run.set_arg(23,1);
			// transformer_run.set_arg(24,0);
			// transformer_run.set_arg(25,0);
		}

		void run_ip_init(){

			/*
			void transformer_cu(
				fdata_v_t *tokens,
				fdata_v_t *w_sf_0, 
				idata_v_t *w_0, 
				fdata_v_t *w_sf_1, 
				idata_v_t *w_1, 
				fdata_v_t *weights, 
				mfdata_v_t *key_cache, 
				mfdata_v_t *value_cache, 
				const int POS,
				const int QKV_W, 
				const int QKV_sf_W,
				const int Out_W, 
				const int Out_sf_W,
				const int FF_w1w3_W, 
				const int FF_w1w3_sf_W,
				const int FF_w2_W, 
				const int FF_w2_sf_W, 
				const int Embed_W, 
				const int Embed_sf_W, 
				const int rms_att_W, 
				const int rms_ffn_W, 
				const int rms_final_W,
			#ifdef __DEBUG__
				const int faker ,const int INIT, const int CURR_LAYER, const int NEXT_STATE,
			#endif
				const float temperature, 
				const float coin, 
				int* pick
			)
				*/

			float tmm_val = 0.9f;
			uint32_t tmp_bits;
			std::memcpy(&tmp_bits, &tmm_val, sizeof(float));
			
			uint64_t token_bo_addr = token_bo.address();
			uint64_t parent_sf_bo_addr = parent_sf_bo.address();
			uint64_t parent_w_bo_addr = parent_w_bo.address();
			uint64_t parent_rms_bo_addr = parent_rms_bo.address();
			uint64_t key_cache_bo_addr = key_cache_bo.address();
			uint64_t value_cache_bo_addr = value_cache_bo.address();
			
			// transformer_run = xrt::run(kernel);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_TOKENS_DATA, static_cast<uint32_t>(token_bo_addr & 0xFFFFFFFF));
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_TOKENS_DATA + 4, static_cast<uint32_t>(token_bo_addr >> 32));
			
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_W_SF_0_DATA, static_cast<uint32_t>(parent_sf_bo_addr & 0xFFFFFFFF));
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_W_SF_0_DATA + 4, static_cast<uint32_t>(parent_sf_bo_addr >> 32));
			
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_W_0_DATA, static_cast<uint32_t>(parent_w_bo_addr & 0xFFFFFFFF));
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_W_0_DATA + 4, static_cast<uint32_t>(parent_w_bo_addr >> 32));
			
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_W_SF_1_DATA, static_cast<uint32_t>(parent_sf_bo_addr & 0xFFFFFFFF));
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_W_SF_1_DATA + 4, static_cast<uint32_t>(parent_sf_bo_addr >> 32));
			
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_W_1_DATA, static_cast<uint32_t>(parent_w_bo_addr & 0xFFFFFFFF));
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_W_1_DATA + 4, static_cast<uint32_t>(parent_w_bo_addr >> 32));
			
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_WEIGHTS_DATA, static_cast<uint32_t>(parent_rms_bo_addr & 0xFFFFFFFF));
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_WEIGHTS_DATA + 4, static_cast<uint32_t>(parent_rms_bo_addr >> 32));
			
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_KEY_CACHE_DATA, static_cast<uint32_t>(key_cache_bo_addr & 0xFFFFFFFF));
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_KEY_CACHE_DATA + 4, static_cast<uint32_t>(key_cache_bo_addr >> 32));
			
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_VALUE_CACHE_DATA, static_cast<uint32_t>(value_cache_bo_addr & 0xFFFFFFFF));
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_VALUE_CACHE_DATA + 4, static_cast<uint32_t>(value_cache_bo_addr >> 32));
			

			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_QKV_W_DATA,tt.QKV_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_QKV_SF_W_DATA,tt.QKV_sf_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_OUT_W_DATA,tt.Out_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_OUT_SF_W_DATA,tt.Out_sf_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_FF_W1W3_W_DATA,tt.FF_w1w3_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_FF_W1W3_SF_W_DATA,tt.FF_w1w3_sf_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_FF_W2_W_DATA,tt.FF_w2_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_FF_W2_SF_W_DATA,tt.FF_w2_sf_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_EMBED_W_DATA,tt.Embed_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_EMBED_SF_W_DATA,tt.Embed_sf_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_RMS_ATT_W_DATA,tt.rms_att_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_RMS_FFN_W_DATA,tt.rms_ffn_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_RMS_FINAL_W_DATA,tt.rms_final_W);
			transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_TEMPERATURE_DATA, tmp_bits);
		}

};
