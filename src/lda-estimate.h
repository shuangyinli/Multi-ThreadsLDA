#ifndef LDA_ESTIMATE_H
#define LDA_ESTIMATE_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#include "lda.h"
#include "lda-data.h"
#include "lda-inference.h"
#include "lda-model.h"
#include "lda-alpha.h"
#include "utils.h"
#include "cokus.h"
#include <pthread.h>

int LAG = 5;

float EM_CONVERGED;
int EM_MAX_ITER;
int ESTIMATE_ALPHA;
double INITIAL_ALPHA;
int NTOPICS;
int NUM_THREADS;


double doc_e_step(document* doc,
                  lda_model* model);

void save_gamma(char* filename, corpus* corpus, int num_topics);

void run_em(char* start,
            char* directory,
            corpus* corpus);

void read_settings(char* filename);
void write_word_assignment(FILE* f, corpus* corpus, lda_model* model);


#endif


