
#ifndef MARK_TB_H
#define MARK_TB_H
#include "../matmult.h"
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

template<typename T, typename S>
void parse_results (std::vector<T> &gold, std::vector<T> &obs){
    int pass = 0;
    int fail = 0;
		int size = gold.size();

    // Use .size() for robust loop bounds
    for (size_t i = 0; i < gold.size(); i++) {
        for (size_t j = 0; j < gold[i].size(); j++) {
            S a = gold[i][j];
            S b = obs[i][j];

            // 1. Calculate tolerance based on ABSOLUTE value
            S tolerance = std::abs(a) * 0.001;
            double diff = std::abs(a - b);

            if (diff <= tolerance) { // Use <= to include exact matches
                pass++;
            } else {
                fail++;
                std::cout << "FAIL | "<< i << "\t"
                          << "Golden: " << a << "\t"
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
