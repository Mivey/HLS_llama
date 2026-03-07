// XRT includes
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>


struct Quantized_bo{
	xrt::bo q;
	xrt::bo s;
};
struct bo_weights{
	xrt::bo rms_att;
	xrt::bo rms_ffn;
	xrt::bo rms_final;
	Quantized_bo query;
	Quantized_bo key;
	Quantized_bo value;
	Quantized_bo out;
	Quantized_bo FFN1;
	Quantized_bo FFN2;
	Quantized_bo FFN3;
	Quantized_bo logits;
};
class MatMultClass{
	public:
	//Matrix Multiply Init
		MatMultClass(xrt::device &d, xrt::uuid &u): device(d), uuid(u){
			kernel = xrt::kernel(device, uuid, "matmult_kernel");
		}
		//Run sequential Method - Start and wait for CU to complete
		//void matmult_kernel(mfdata_v_t *out, mfdata_v_t *fl_tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int READ_OFFSET, const int WRITE_OFFSET)
		void run(xrt::bo &output, xrt::bo &input, xrt::bo &sf_bo, xrt::bo &w_bo, const int n_dim, const int m_dim, const int read_os, const int wwrite_os){
			auto run = kernel(output, input, sf_bo, w_bo, n_dim, m_dim, read_os, wwrite_os);
			run.wait();
		}
		//Run parallel Method - Start multiple CU's 
		xrt::run start(xrt::bo &output, xrt::bo &input, xrt::bo &sf_bo, xrt::bo &w_bo, const int n_dim, const int m_dim, const int read_os, const int wwrite_os){
			return kernel(output, input, sf_bo, w_bo, n_dim, m_dim, read_os, wwrite_os);
		}

		// Wait parallel Method - wait for the handles to return
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
	//MHA Init
		MHAClass(xrt::device &d, xrt::uuid &u): device(d), uuid(u){
			kernel = xrt::kernel(device, uuid, "mha_kernel");
			allocate_cache();
		}
		//Run Sequential Method
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

		//private method to store the KV cache objects.
		void allocate_cache(){
			//create the containers for the weights
			key_bo = xrt::bo(device, c_size, kernel.group_id(0));
			value_bo = xrt::bo(device, c_size, kernel.group_id(0));
		}
};

class SwigluClass{
	public:
	//SWIGLU Init
		SwigluClass(xrt::device &d, xrt::uuid &u): device(d), uuid(u){
			kernel = xrt::kernel(device, uuid, "swiglu_kernel");
		}
		//Sequential Run Method
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
// void rescon_kernel(fdata_v_t *x, fdata_v_t *xb){
		void run(xrt::bo &x_bo, xrt::bo &xb_bo){
			auto run = kernel(x_bo, xb_bo);
			run.wait();
		}

	private:
		xrt::device device;
		xrt::uuid uuid;
		xrt::kernel kernel;		
};


class ForwardBlock{
	public:
		ForwardBlock(
			int device_id, std::string& binaryFile, 
			// std::string &QKV_w_comb, std::string &QKV_sf_comb,
			// std::string &out_w, std::string &out_sf, 
			// std::string &ffn1_3_w_comb, std::string &ffn1_3_sf_comb, 
			// std::string &ffn2_w, std::string &ffn2_sf,
			// std::string &logits_w, std::string &logits_sf,
			// std::string &rms_att_w, std::string &rms_ffn_w, std::string &rms_final_w,
			std::string &checkpoint,
			Transformer *t, const int mm_cu_cnt){
				
			try {
				device = xrt::device(device_id);
				std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
     		std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";
				uuid = device.load_xclbin(binaryFile);
				xrt::kernel kernel = xrt::kernel(device, uuid, "matmult_kernel");
				g_id = kernel.group_id(2);
			} catch (const std::exception& e) {
				throw std::runtime_error(std::string(e.what()));
			}
			
			p = &t->config;
			w = &t->weights;
			s = &t->state;
			std::cout<<"did I test it here?\n";
			
			std::ifstream file(checkpoint, std::ios::binary | std::ios::ate);
			size_t file_size = file.tellg();
			file.seekg(0, std::ios::beg);
			parent_bo = xrt::bo(device, file_size,kernel.group_id(0));
			file.read(parent_bo.map<char*>(), file_size);
			parent_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

			size_t offset = 256; // first bytes are for config file
			int GS = 64;
			// maybe later we read this directly and use it inside of runForward intead of passing transformer t.

			//lambda helper slicer upper
			auto slice = [&](size_t size) -> xrt::bo {
				xrt::bo sub = xrt::bo(parent, size, offset);
				offset += size;
				return sub;
			};

			auto init_quantized_slice = [&](size_t size, size_t n) -> Quantized_bo{
				
			}

			w_bo.rms_att = slice(p->n_layers * p->dim * sizeof(float));
			w_bo.rms_ffn = slice(p->n_layers * p->dim * sizeof(float));
			w_bo.rms_final = slice( p->dim * sizeof(float));
			
			w_bo.query.q = slice(p->n_layers * p->dim * p->dim * sizeof(int8_t));
			w_bo.query.s = slice(p->n_layers * p->dim * p->dim * sizeof(float) / GS);
			w_bo.key.q = slice(p->n_layers * p->dim * p->dim * sizeof(int8_t));
			w_bo.key.s = slice(p->n_layers * p->dim * p->dim * sizeof(float) / GS);
			w_bo.value.q = slice(p->n_layers * p->dim * p->dim * sizeof(int8_t));
			w_bo.value.s = slice(p->n_layers * p->dim * p->dim * sizeof(float) / GS);
			w_bo.out.q = slice(p->n_layers * p->dim * p->dim * sizeof(int8_t));
			w_bo.out.s = slice(p->n_layers * p->dim * p->dim * sizeof(float) / GS);
			w_bo.FFN1.q = slice(p->n_layers * p->dim * p->hidden_dim * sizeof(int8_t));
			w_bo.FFN1.s = slice(p->n_layers * p->dim * p->hidden_dim * sizeof(float) / GS);
			w_bo.FFN2.q = slice(p->n_layers * p->dim * p->hidden_dim * sizeof(int8_t));
			w_bo.FFN2.s = slice(p->n_layers * p->dim * p->hidden_dim * sizeof(float) / GS);
			w_bo.FFN3.q = slice(p->n_layers * p->dim * p->hidden_dim * sizeof(int8_t));
			w_bo.FFN3.s = slice(p->n_layers * p->dim * p->hidden_dim * sizeof(float) / GS);
			w_bo.logits.q = slice(p->dim * p->vocab_size * sizeof(int8_t));
			w_bo.logits.s = slice(p->dim * p->vocab_size * sizeof(float) / GS);
			

			mm_cu_kernel.reserve(mm_cu_cnt);
			for (int i = 0 ; i <mm_cu_cnt; i++) {
				mm_cu_kernel.push_back(std::make_unique<MatMultClass>(device, uuid));
			}

			rms_kernel = std::make_unique<RMSClass>(device, uuid);
      swiglu_kernel = std::make_unique<SwigluClass>(device, uuid);
      mha_kernel = std::make_unique<MHAClass>(device, uuid);
      rc_kernel = std::make_unique<ResConClass>(device, uuid);

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

			token_bo_map = token_bo.map<float*>();
			mm_logits_map = mm_logits.map<float*>();
			std::cout<<"init complete feb 14 5:30\n";
		}
/*===============================================================================================================
RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD RUN FORWARD 
=====================================================================================================*/
		float* runForward(const int token, const int pos){
			float *cr = w->token_embedding_table + token * p->dim;
			// memcpy(x, cr, p->dim * sizeof(*x)); //wrong!
			std::memcpy(token_bo_map, cr, p->dim * sizeof(float));
				// std::fill(rc_bo_map, rc_bo_map + p->dim, 0.0f);
				// rc_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

			token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

			for (int l = 0; l < p->n_layers; l++) {
				
				rms_kernel->run(rms_bo, token_bo, rms_att_bo, l);
				// std::cout<<"rms\n";
				int q_off = p->dim * l * 3;
				int k_off = p->dim + q_off;
				int v_off = p->dim + k_off;
				int v2_off =  p->dim / 2 + v_off;
				
				auto q_run = mm_cu_kernel[0]->start(q_out, rms_bo, w_bo.query.s, w_bo.query.q, p->dim, p->dim, q_off, 0);
				auto k_run = mm_cu_kernel[1]->start(k_out, rms_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, k_off, 0);
				auto v_run = mm_cu_kernel[0]->start(v_out, rms_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim/2, v_off, 0);
				auto v2_run = mm_cu_kernel[1]->start(v_out, rms_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim/2, v2_off, p->dim/2);

				mm_cu_kernel[0]->wait(q_run);
				mm_cu_kernel[1]->wait(k_run);
				mm_cu_kernel[0]->wait(v_run);
				mm_cu_kernel[1]->wait(v2_run);
				
				// std::cout<<"qkv\n";
				mha_kernel->run(q_out, k_out, v_out, l, pos);
				// std::cout<<"mha\n";
				int o_off = p->dim * l;
				int o2_off = p->dim / 2 + o_off;
				
				// mm_cu_kernel[0]->run(o_out, q_out, out_sf_bo, out_w_bo, p->dim, p->dim, q_off, 0);
				auto o_run = mm_cu_kernel[0]->start(o_out, q_out, out_sf_bo, out_w_bo, p->dim, p->dim/2, o_off, 0);
				auto o2_run = mm_cu_kernel[1]->start(o_out, q_out, out_sf_bo, out_w_bo, p->dim, p->dim/2, o2_off, p->dim/2);
				mm_cu_kernel[0]->wait(o_run);
				mm_cu_kernel[1]->wait(o2_run);
				
				rc_kernel->run(token_bo, o_out);
				rms_kernel->run(rms_bo, token_bo, rms_ffn_bo, l);
				
				int ffn1_off = p->hidden_dim * l * 2;
				int ffn3_off = p->hidden_dim + ffn1_off;
				auto ffn1_run = mm_cu_kernel[0]->start(FFN1_out, rms_bo, ffn13_sf_bo, ffn13_w_bo, p->dim, p->hidden_dim, ffn1_off, 0); //FFN1_mm_kernel->start(FFN1_out, o_out, l);
				auto ffn3_run = mm_cu_kernel[1]->start(FFN3_out, rms_bo, ffn13_sf_bo, ffn13_w_bo, p->dim, p->hidden_dim, ffn3_off, 0); //FFN3_mm_kernel->start(FFN3_out, o_out, l);

				mm_cu_kernel[0]->wait(ffn1_run);
				mm_cu_kernel[1]->wait(ffn3_run);
				
				swiglu_kernel->run(swiglu_out, FFN1_out, FFN3_out);
				
				int ffn2_off = p->dim * l;
				int ffn22_off = ffn2_off + p->dim/2;
				auto ffn2_run = mm_cu_kernel[0]->start(FFN2_out, swiglu_out, ffn2_sf_bo, ffn2_w_bo, p->hidden_dim, p->dim/2, ffn2_off, 0);
				auto ffn22_run = mm_cu_kernel[1]->start(FFN2_out, swiglu_out, ffn2_sf_bo, ffn2_w_bo, p->hidden_dim, p->dim/2, ffn22_off, p->dim/2);
				
				mm_cu_kernel[0]->wait(ffn2_run);
				mm_cu_kernel[1]->wait(ffn22_run);

				rc_kernel->run(token_bo, FFN2_out);
			}
			rms_kernel->run(rc_bo, token_bo, rms_final_bo, 0);
			auto logit_run = mm_cu_kernel[0]->start(mm_logits, rc_bo, logits_sf_bo, logits_w_bo, p->dim, p->vocab_size/2, 0, 0);
			auto logit2_run = mm_cu_kernel[1]->start(mm_logits, rc_bo, logits_sf_bo, logits_w_bo, p->dim, p->vocab_size/2, p->vocab_size/2, p->vocab_size/2);
				mm_cu_kernel[0]->wait(logit_run);
				mm_cu_kernel[1]->wait(logit2_run);
			
			mm_logits.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
			return mm_logits_map;
		}


		float* mha_speedup(float* xb, const int l, const int pos){
			std::memcpy(token_bo.map<float*>(), xb, p->dim * sizeof(float));
			token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

			rms_kernel->run(rc_bo, token_bo, rms_att_bo, l);
			// std::memcpy(k_out.map<float*>(), k, p->dim * sizeof(float));
			// std::memcpy(v_out.map<float*>(), v, p->dim * sizeof(float));
			
			// k_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			// v_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			
				int q_off = p->dim * l * 3;
				int k_off = p->dim + q_off;
				int v_off = p->dim + k_off;
				int v2_off =  p->dim / 2 + v_off;
				
				auto q_run = mm_cu_kernel[0]->start(q_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, q_off, 0);
				auto k_run = mm_cu_kernel[1]->start(k_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim, k_off, 0);
				auto v_run = mm_cu_kernel[0]->start(v_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim/2, v_off, 0);
				auto v2_run = mm_cu_kernel[1]->start(v_out, rc_bo, qkv_sf_bo, qkv_w_bo, p->dim, p->dim/2, v2_off, p->dim/2);

				mm_cu_kernel[0]->wait(q_run);
				mm_cu_kernel[1]->wait(k_run);
				mm_cu_kernel[0]->wait(v_run);
				mm_cu_kernel[1]->wait(v2_run);

				q_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
				k_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
				v_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

			
			mha_kernel->run(q_out, k_out, v_out, l, pos);
			q_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
			return q_out.map<float*>();
		}

		/* 
		=========================================================================================
		PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE PRIVATE 
		==============================================================================================
		*/
	private:
		xrt::device device;
		xrt::uuid uuid;

		bo_weights w_bo;
		xrt::bo parent_bo;

		xrt::bo qkvo_w_bo, qkvo_sf_bo, w1w2w3_w_bo, w1w2w3_sf_bo, logits_w_bo, logits_sf_bo;
		xrt::bo rms_aff_bo;
		/* const */int q_off, k_off, v_off, o_off, ffn1_off, ffn2_off, ffn3_off, rms_att_off, rms_ffn_off, rms_final_off;

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

		void quantized_mem_init(std::string &sw, xrt::bo &sf, xrt::bo &w, int offset, const int l, const int n, const int m, const int c){
			std::ifstream w_dat(sw, std::ios::binary);
			//check if we can read, throw error otherwise
			if (!w_dat.is_open()) {
				throw std::runtime_error(sw + ". Try again, but do it right this time.");
			}
			w_dat.seekg(offset, ifstream::beg);

			for (int i = 0; i < l; i++) {
				for (int j = 0; j < c; j++) {
					w_dat.read(reinterpret_cast<char*>(w_bo.map<char*>()), n * m)
				}
			}
		}

		void sw_mha(float* sxb, float* sq, float* sk, float *sv, int l, int pos){
			
			int dim = p->dim;
			int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
			int head_size = dim / p->n_heads;		
    int kv_mul = p->n_heads / p->n_kv_heads;
        // RoPE relative positional encoding: complex-valued rotate q and k in each head
			for (int i = 0; i < dim; i+=2) {
				int head_dim = i % head_size;
				float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
				float val = pos * freq;
				float fcr = cosf(val);
				float fci = sinf(val);
				int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
				for (int v = 0; v < rotn; v++) {
					float* vec = v == 0 ? sq : sk; // the vector to rotate (query or key)
					float v0 = vec[i];
					float v1 = vec[i+1];
					vec[i]   = v0 * fcr - v1 * fci;
					vec[i+1] = v0 * fci + v1 * fcr;
				}
			}


			int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
			float* key_cache_row = s->key_cache + loff + pos * kv_dim;
			float* value_cache_row = s->value_cache + loff + pos * kv_dim;
			memcpy(key_cache_row, sk, kv_dim * sizeof(*key_cache_row));
			memcpy(value_cache_row, sv, kv_dim * sizeof(*value_cache_row));


			// multihead attention. iterate over all heads
			int h;
			#pragma omp parallel for private(h)
			for (h = 0; h < p->n_heads; h++) {
				// get the query vector for this head
				float* q = sq + h * head_size;
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
				float* xb = sxb + h * head_size;
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
		}

};
