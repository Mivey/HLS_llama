
#ifndef MARK_TB_H
#define MARK_TB_H
// #include "../../forward.h"
// #include "../matmul.h"
// #include "../quantizer.h"
// #include "../rmsnorm.h"
// #include "../rope.h"
#include "../mha.h"
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

int mat_mul_tb();
int quantizer_tb();
int rmsnorm_tb();
int rope_tb();
int mha_tb();
int rqm_rope_tb();
int rqm_swiglu_tb();
int top_tb();
int rqm_rope_mha_tb();

// template<typename T, typename S>
// void parse_results (std::vector<T> &gold, std::vector<T> &obs){
//     int pass = 0;
//     int fail = 0;
// 		int size = gold.size();

//     // Use .size() for robust loop bounds
//     for (size_t i = 0; i < gold.size(); i++) {
//         for (size_t j = 0; j < gold[i].size(); j++) {
//             S a = gold[i][j];
//             S b = obs[i][j];

//             // 1. Calculate tolerance based on ABSOLUTE value
//             S tolerance = std::abs(a) * 0.001;
//             double diff = std::abs(a - b);

//             if (diff <= tolerance) { // Use <= to include exact matches
//                 pass++;
//             } else {
//                 fail++;
//                 std::cout << "FAIL | "<< i << "\t"
//                           << "Golden: " << a << "\t"
//                           << "Observed: " << b << "\t"
//                           << "Diff: " << diff;

//                 // 2. Prevent division by zero
//                 if (a != 0.0) {
//                     double percent_error = (diff / std::abs(a)) * 100.0;
//                     std::cout << "\tError: " << percent_error << "%";
//                 }
//                 std::cout << std::endl;
//             }
//         }
//     }

//     std::cout << "----------------------------------" << std::endl;
//     std::cout << "PASSED: " << pass << std::endl;
//     std::cout << "FAILED: " << fail << std::endl;
//     std::cout << "----------------------------------" << std::endl;
// }

template<typename T, typename S>
void parse_results (const std::vector<T> &gold, const std::vector<T> &obs, double rel_tol = 0.03, double abs_tol = 1e-4){
    int pass = 0;
    int fail = 0;
    int print_limit = 200; // Prevent console spam

    // Metrics for AI/LLM evaluation
    double mse_sum = 0.0;
    double dot_product = 0.0;
    double norm_gold = 0.0;
    double norm_obs = 0.0;
    int total_elements = 0;

    for (size_t i = 0; i < gold.size(); i++) {
        for (size_t j = 0; j < gold[i].size(); j++) {
            S a = gold[i][j];
            S b = obs[i][j];
            total_elements++;

            // 1. Calculate tolerance using Absolute + Relative Epsilon
            double diff = std::abs(a - b);
            double tolerance = abs_tol + (rel_tol * std::abs(a));

            // Accumulate metrics for MSE and Cosine Similarity
            mse_sum += (diff * diff);
            dot_product += (a * b);
            norm_gold += (a * a);
            norm_obs += (b * b);

            if (diff <= tolerance) { 
                pass++;
            } else {
                fail++;
                
                // 2. Suppress excessive print statements
                if (fail <= print_limit) {
                    std::cout << "FAIL | [" << i << "][" << j << "]\t"
                              << "Golden: " << a << "\t"
                              << "Observed: " << b << "\t"
                              << "Diff: " << diff;

                    if (a != 0.0) {
                        double percent_error = (diff / std::abs(a)) * 100.0;
                        std::cout << "\tError: " << percent_error << "%";
                    }
                    std::cout << std::endl;
                } else if (fail == print_limit + 1) {
                    std::cout << "... further individual failures suppressed ..." << std::endl;
                }
            }
        }
    }

    // 3. Calculate final LLM metrics
    double mse = mse_sum / total_elements;
    double cos_sim = 0.0;
    if (norm_gold > 0.0 && norm_obs > 0.0) {
        cos_sim = dot_product / (std::sqrt(norm_gold) * std::sqrt(norm_obs));
    }

    std::cout << "==================================" << std::endl;
    std::cout << "PASSED: " << pass << std::endl;
    std::cout << "FAILED: " << fail << std::endl;
    std::cout << "----------------------------------" << std::endl;
    std::cout << "MEAN SQUARED ERROR: " << mse << std::endl;
    std::cout << "COSINE SIMILARITY:  " << cos_sim << std::endl;
    std::cout << "==================================" << std::endl;
}

template<typename T, typename S>
void parse_cache_results (std::vector<T> &gold, std::vector<T> &obs){
    int pass = 0;
    int fail = 0;
		int elements = 0, tokens = 0, heads = 0, layers = 0;

    // Use .size() for robust loop bounds
    for (size_t i = 0; i < gold.size(); i++) {
        for (size_t j = 0; j < gold[i].size(); j++) {
            S a = gold[i][j];
            S b = obs[i][j];
						if (elements == 64) {
							elements = 0;
							tokens++;
							if (tokens == 1024) {
								tokens = 0;
								heads++;
								if (heads == 12) {
									heads = 0;
									layers++;
								}
							}
						}
						elements++;

            // 1. Calculate tolerance based on ABSOLUTE value
            S tolerance = std::abs(a) * 0.001;
            double diff = std::abs(a - b);

            if (diff <= tolerance) { // Use <= to include exact matches
                pass++;
            } else {
                fail++;
                std::cout << "FAIL | "<< "\tL: "<<layers<<"\tH: " <<heads<<"\tT: "<<tokens<<"\tE: "<<elements
                          << "\tGolden: " << a << "\t"
                          << "Observed: " << b << "\t"
                          << "Diff: " << diff;

                // 2. Prevent division by zero
                if (a != 0.0) {
                    double percent_error = (diff / std::abs(a)) * 100.0;
                    std::cout << "\tError: " << percent_error << "%";
                }
                std::cout << std::endl;
            }
        }
    }

    std::cout << "----------------------------------" << std::endl;
    std::cout << "PASSED: " << pass << std::endl;
    std::cout << "FAILED: " << fail << std::endl;
    std::cout << "----------------------------------" << std::endl;
}

#endif