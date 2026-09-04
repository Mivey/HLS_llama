// ============================================================================
//  generate_loop.h
//
//  Drop-in replacement for newgen() in fastforward.cpp. Templated on the engine
//  so the same loop drives either FastForward (xrt::ip) or RunForward
//  (xrt::kernel).
//
//  #include this in fastforward.cpp AFTER Tokenizer, Sampler, decode(),
//  safe_printf(), encode() and random_f32() are declared -- it references them
//  by unqualified name at template-definition time.
// ============================================================================
#ifndef GENERATE_LOOP_H
#define GENERATE_LOOP_H

#include "compute_units.h"

// Mirrors runq.c's generate():
//   pos <  num_prompt_tokens-1  ->  prefill; the kernel only fills the KV cache
//                                   and endForward() hands back the prompt
//                                   token the host already seeded
//   pos >= num_prompt_tokens-1  ->  decode; the kernel runs the LM head and
//                                   writes curr_token[pos+1]
template <typename Engine>
void newgen(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
            char* prompt, int steps, Engine& f)
{
    char* empty_prompt = const_cast<char*>("");
    if (prompt == NULL) { prompt = empty_prompt; }

    int  num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // curr_token[pos+1] is written, so the last usable pos is SEQ_LEN-2.
    const int max_steps = llama_hw::MODEL_SEQUENCE_LEN - 1;
    if (steps > max_steps) steps = max_steps;
    if (num_prompt_tokens > max_steps) {
        fprintf(stderr, "prompt is longer than the KV cache (%d tokens)\n", max_steps);
        exit(EXIT_FAILURE);
    }

    // Seed the whole prompt into the device-side curr_token[] up front. The
    // kernel reads curr_token[pos] each step and, in decode, appends to
    // curr_token[pos+1].
    f.seed_prompt(prompt_tokens, num_prompt_tokens);

    long start = 0;
    int  pos   = 0;
    int  token = prompt_tokens[0];
    int  next  = token;

    bool prefill_mode = (num_prompt_tokens > 1);
    if (prefill_mode) f.enable_prefill(); else f.enable_decode();

    // internal_rms_weights is a static URAM array in the kernel, so the RMS
    // weights only need loading on the very first invocation.
    f.set_rms_flag(true);

    while (pos < steps) {
        const bool want_prefill = (pos < num_prompt_tokens - 1);
        if (want_prefill != prefill_mode) {
            prefill_mode = want_prefill;
            if (prefill_mode) f.enable_prefill(); else f.enable_decode();
        }

        const float coin = random_f32(&sampler->rng_state);

        f.startForward(pos, coin);
        next = f.endForward(pos);

        if (pos == 0) {
            f.set_rms_flag(false);
            start = time_in_ms();
        }

        // BOS (=1) terminates the sequence, as in runq.c.
        if (next == 1) { pos++; break; }
        if (next < 0 || next >= tokenizer->vocab_size) {
            fprintf(stderr, "\n[host] kernel returned out-of-range token %d at pos %d\n",
                    next, pos);
            break;
        }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);

        token = next;
        pos++;
    }
    printf("\n");

    if (pos > 1 && start != 0) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }

    free(prompt_tokens);
}

#endif // GENERATE_LOOP_H
