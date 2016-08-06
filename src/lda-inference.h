#ifndef LDA_INFERENCE_H
#define LDA_INFERENCE_H

#include <math.h>
#include <float.h>
#include <assert.h>
#include "lda.h"
#include "utils.h"

float VAR_CONVERGED;
int VAR_MAX_ITER;

double lda_inference(document* doc, lda_model* model, double* var_gamma, double** phi);
double compute_likelihood(document* doc, lda_model* model, double** phi, double* var_gamma);

#endif
