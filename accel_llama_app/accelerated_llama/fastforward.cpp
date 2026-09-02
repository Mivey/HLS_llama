#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <string>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
#include "compute_units.h"

// Globals
int GS = 0; 

// ----------------------------------------------------------------------------
// File IO

void read_checkpoint(char* checkpoint, Config* config, 
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); exit(EXIT_FAILURE); }
    
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 2) { fprintf(stderr, "Bad version %d, need version 2\n", version); exit(EXIT_FAILURE); }
    
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    
    uint8_t shared_classifier; 
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    
    int group_size; 
    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    GS = group_size; 
    
    fseek(file, 0, SEEK_END); 
    *file_size = ftell(file); 
    fclose(file);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->fd, &t->data, &t->file_size);
}

void free_transformer(Transformer* t) {
    if (t->fd != -1) { close(t->fd); }
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer

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
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; 
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; 
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
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; 
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str }; 
    TokenIndex *res = (TokenIndex*) bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex*) malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = (char*) malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) { str_len = 0; }
        str_buffer[str_len++] = *c; 
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) { continue; }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; 
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) { break; }

        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; 
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler (RNG Components Maintained)

typedef struct {
    float prob;
    int index;
} ProbIndex; 

typedef struct {
    int vocab_size;
    ProbIndex* probindex; 
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = (ProbIndex*) malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { 
    return (random_u32(state) >> 8) / 16777216.0f;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void newgen(Transformer *transformer, Tokenizer *tokenizer, Sampler* sampler, char* prompt, int steps, FastForward &f){
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); 
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;  
    int next;        
    int pos = 0;     
    int token = prompt_tokens[pos]; 
    float flipped = random_f32(&sampler->rng_state);
    
    start = time_in_ms();
    
    // 1) & 3) Prefill Modes
    f.enable_prefill(); 
    f.set_rms_flag(true); 
    
    next = token;
    for (int ii = 0; ii < (num_prompt_tokens - 1); ii++) {
        f.startForward(next, pos, flipped);
        // pos++;
				next = prompt_tokens[++pos];
        if (ii != 0) {
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); 
            fflush(stdout);
        }
        token = next;
        flipped = random_f32(&sampler->rng_state);
        // next = prompt_tokens[pos];//
				f.endForward();
    }

    // 2) Decode unroll entry (preps the transition to 4)
    f.enable_decode();
    f.set_rms_flag(true);
    f.startForward(next, pos, flipped);
    pos++;
    
    char* piece = decode(tokenizer, token, next);
    safe_printf(piece); 
    fflush(stdout);
    token = next;
    flipped = random_f32(&sampler->rng_state);
    next = f.endForward();
    
    // 4) Continuous Decode
    f.set_rms_flag(false);
    do {
        f.startForward(next, pos, flipped);
        pos++;
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); 
        fflush(stdout);
        token = next;
        flipped = random_f32(&sampler->rng_state);
        next = f.endForward();
    } while (pos < steps);
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   fastforward <checkpoint> <xclbin_file> [options]\n");
    fprintf(stderr, "Example: fastforward model.bin llama_pen.xclbin -n 256 -i \"Once upon a time\"\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;  
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   
    float topp = 0.9f;          
    int steps = 256;            
    char *prompt = NULL;        
    unsigned long long rng_seed = 0; 
    char *mode = "generate";    
    char *system_prompt = NULL; 
    std::string xclbin_file;
    std::string device_id = "0";
        
    if (argc >= 3) { 
        checkpoint_path = argv[1]; 
        xclbin_file = argv[2];
    } else { 
        error_usage(); 
    }
    
    for (int i = 3; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } 
        if (argv[i][0] != '-') { error_usage(); } 
        if (strlen(argv[i]) != 2) { error_usage(); } 
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; 

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

		std::cout<< " ███████╗ █████╗ ███████╗████████╗    ███████╗ ██████╗ ██████╗ ██╗    ██╗ █████╗ ██████╗ ██████╗ "<<std::endl;
		std::cout<< " ██╔════╝██╔══██╗██╔════╝╚══██╔══╝    ██╔════╝██╔═══██╗██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔══██╗"<<std::endl;
		std::cout<< " █████╗  ███████║███████╗   ██║       █████╗  ██║   ██║██████╔╝██║ █╗ ██║███████║██████╔╝██║  ██║"<<std::endl;
		std::cout<< " ██╔══╝  ██╔══██║╚════██║   ██║       ██╔══╝  ██║   ██║██╔══██╗██║███╗██║██╔══██║██╔══██╗██║  ██║"<<std::endl;
		std::cout<< " ██║     ██║  ██║███████║   ██║       ██║     ╚██████╔╝██║  ██║╚███╔███╔╝██║  ██║██║  ██║██████╔╝"<<std::endl;
		std::cout<< " ╚═╝     ╚═╝  ╚═╝╚══════╝   ╚═╝       ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ "<<std::endl;

    std::cout << "Init FastForward class\n";
    FastForward f = FastForward(std::stoi(device_id), xclbin_file, checkpoint_path);

    if (strcmp(mode, "generate") == 0) {
        newgen(&transformer, &tokenizer, &sampler, prompt, steps, f);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif