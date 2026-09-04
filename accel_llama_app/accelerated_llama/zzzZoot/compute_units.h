// XRT includes
#include "xrt/xrt_device.h"
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

// ... [Keep your axi_reg, Config, and Transformer structs here] ...

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
	float temperature = 0.9;
	float coin;
	bool init_rms_flag = true;
	bool pf_dc_flag = true;
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
    Config config; // the hyperparameters of the architecture (the blueprint)
    // TransformerWeights weights; // the weights of the model
    // RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;


class FastForward {
public:
    FastForward(int device_id, std::string& binaryFile, const std::string& checkpoint) {
        try {
            device = xrt::device(device_id);
            std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
            std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";
            std::cout << "Compiled on " << __DATE__ << " at " << __TIME__ << std::endl;
            
            uuid = device.load_xclbin(binaryFile);
            
            // Switch to xrt::ip for bare-metal AXI-Lite register access
            transformer_ip = xrt::ip(device, uuid, "transformer_cu");
            
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string(e.what()));
        }

        allocate_cache_init();
        weights_init(checkpoint);
        run_init();
        
        std::cout << "FastForward XRT::IP interface initialized successfully.\n";
    }

/*====================================================================================
RUN FORWARD
=====================================================================================*/

    void startForward(const int token, const int pos, const float coin) {
        // Write the token and pos integers directly to the IP
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_CURR_TOKEN_I_DATA, token);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_POS_R_DATA, pos);

        // Convert coin float to uint32_t for AXI-Lite transport
        std::memcpy(&tmp_bits, &coin, sizeof(float));  
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_COIN_DATA, tmp_bits);
        
        // Assert AP_START (bit 0) to kick off the kernel
        uint32_t ctrl = transformer_ip.read_register(XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL, ctrl | 0x01);
    }

    int endForward() {
        // Poll AP_DONE (bit 1) to synchronize 
        while ((transformer_ip.read_register(XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL) & 0x02) == 0) {
            // Optional: std::this_thread::yield() if CPU spinning becomes a bottleneck
        }
        
        // The HLS core will clear AP_DONE automatically upon the next AP_START assertion.
        return transformer_ip.read_register(XTRANSFORMER_CU_CONTROL_ADDR_CURR_TOKEN_O_DATA);
    }

    void set_rms_flag(const bool x) {
        tt.init_rms_flag = x;
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_INIT_RMS_FLAG_DATA, x ? 1 : 0);
    }
    
    void enable_decode() {
        tt.pf_dc_flag = true;
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_PF_DC_FLAG_DATA, 1);
    }

    void enable_prefill() {
        tt.pf_dc_flag = false;
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_PF_DC_FLAG_DATA, 0);
    }

private:
    xrt::device device;
    xrt::uuid uuid;
    xrt::ip transformer_ip;

    xrt::bo parent_rms_bo;
    xrt::bo parent_w_bo;
    xrt::bo parent_sf_bo;
    xrt::bo key_cache_bo;
    xrt::bo value_cache_bo;
    xrt::bo token_bo;
    
    int MODEL_ELEMENTS = 768;
    int MODEL_HIDDEN_DIM = 2048;
    int MODEL_SCALING_FACTOR = 64;
    int MODEL_SEQUENCE_LEN = 1024;
    int MODEL_NUM_LAYERS = 12;
    int MODEL_TOKENS = 32000;
		uint32_t tmp_bits;

    axi_reg tt;
    Config* p;

    // Helper to map 64-bit physical device addresses to consecutive 32-bit AXI-Lite registers
    void write_bo_address(uint32_t offset, xrt::bo& bo) {
        uint64_t addr = bo.address();
        transformer_ip.write_register(offset, static_cast<uint32_t>(addr & 0xFFFFFFFF));
        transformer_ip.write_register(offset + 4, static_cast<uint32_t>(addr >> 32));
    }

    void allocate_cache_init() {
        size_t c_size = (size_t)MODEL_ELEMENTS * MODEL_SEQUENCE_LEN * MODEL_NUM_LAYERS * sizeof(float);
        key_cache_bo = xrt::bo(device, c_size, 0);
        value_cache_bo = xrt::bo(device, c_size, 0);
    }

		void weights_init(const std::string &checkpoint){
			
			//init token_bo
			size_t embed_float_size = MODEL_ELEMENTS * MODEL_TOKENS * sizeof(float);
			token_bo = xrt::bo(device, embed_float_size, 0);
			float* token_map_f = token_bo.map<float*>();
			// token_map_i = token_bo.map<int*>();
			// token_map_c = token_bo.map<char*>();
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
			
			parent_rms_bo = xrt::bo(device, rms_size, 0);
			parent_w_bo = xrt::bo(device, q_size, 0);
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
			
			size_t emb_tok_ptr = file.tellg();
			size_t q_idx = 0;
			size_t sf_idx = 0;
			
			
			tt.Embed_W = 0;
			file.read(q_ptr + q_idx, embed_size);
			q_idx += embed_size;
			
			size_t emb_tok_sf_ptr = file.tellg();
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

			size_t total_embed_elements = MODEL_ELEMENTS * MODEL_TOKENS;
			size_t total_sf_elements = total_embed_elements / MODEL_SCALING_FACTOR;
			
			std::vector<int8_t> tmp_q_embed(total_embed_elements);
			std::vector<float> tmp_sf_embed(total_sf_elements);

			// 3. Perform exactly TWO bulk file reads (Fixing the I/O bottleneck)
			file.seekg(emb_tok_ptr, std::ios::beg);
			file.read(reinterpret_cast<char*>(tmp_q_embed.data()), tmp_q_embed.size());

			file.seekg(emb_tok_sf_ptr, std::ios::beg);
			file.read(reinterpret_cast<char*>(tmp_sf_embed.data()), tmp_sf_embed.size() * sizeof(float));

			// 4. Dequantize purely in memory, directly into the XRT buffer (Fixing the math)
			for (int i = 0; i < total_embed_elements; i++) {
					int group = i / MODEL_SCALING_FACTOR;
					token_map_f[i] = static_cast<float>(tmp_q_embed[i]) * tmp_sf_embed[group];
			}

			parent_rms_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			parent_sf_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			parent_w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			file.close();
		}

    void run_init() {
        // Map 64-bit Memory Pointers
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_TOKENS_DATA, token_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_W_SF_0_DATA, parent_sf_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_W_0_DATA, parent_w_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_W_SF_1_DATA, parent_sf_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_W_1_DATA, parent_w_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_WEIGHTS_DATA, parent_rms_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_KEY_CACHE_DATA, key_cache_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_VALUE_CACHE_DATA, value_cache_bo);

        // Map 32-bit Offsets & Dimensions
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_QKV_W_DATA, tt.QKV_W);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_QKV_SF_W_DATA, tt.QKV_sf_W);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_OUT_W_DATA, tt.Out_W);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_OUT_SF_W_DATA, tt.Out_sf_W);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_FF_W1W3_W_DATA, tt.FF_w1w3_W);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_FF_W1W3_SF_W_DATA, tt.FF_w1w3_sf_W);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_FF_W2_W_DATA, tt.FF_w2_W);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_FF_W2_SF_W_DATA, tt.FF_w2_sf_W);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_EMBED_W_DATA, tt.Embed_W);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_EMBED_SF_W_DATA, tt.Embed_sf_W);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_RMS_ATT_W_DATA, tt.rms_att_W);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_RMS_FFN_W_DATA, tt.rms_ffn_W);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_RMS_FINAL_W_DATA, tt.rms_final_W);
        
        // Base Hyperparameters
        uint32_t tmp_temp;
        std::memcpy(&tmp_temp, &tt.temperature, sizeof(float));
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_TEMPERATURE_DATA, tmp_temp);

        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_INIT_RMS_FLAG_DATA, tt.init_rms_flag ? 1 : 0);
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_PF_DC_FLAG_DATA, tt.pf_dc_flag ? 1 : 0);
    }
};
