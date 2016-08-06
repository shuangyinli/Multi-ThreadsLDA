// (C) Copyright 2004, David M. Blei (blei [at] cs [dot] cmu [dot] edu)

// This file is part of LDA-C.

// LDA-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// LDA-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "lda-estimate.h"

/*
 * perform inference on a document and update sufficient statistics
 *
 */

double doc_e_step(document* doc,
                  lda_model* model)
{
    double likelihood;

    double * docgamma = doc->doc_gamma;
    double ** docphi = doc->doc_phi;
    // posterior inference

    likelihood = lda_inference(doc, model, docgamma, docphi);

    // update sufficient statistics
    doc->likelihood = likelihood;
    return(likelihood);
}


/*
 * writes the word assignments line for a document to a file
 *
 */


/*
 * saves the gamma parameters of the current dataset
 *
 */

void save_gamma(char* filename, corpus* corpus, int num_topics)
{
    FILE* fileptr;
    int num_docs = corpus->num_docs;
    int d, k;
    fileptr = fopen(filename, "w");

    for (d = 0; d < num_docs; d++)
    {
	fprintf(fileptr, "%5.10f", corpus->docs[d].doc_gamma[0]);
	for (k = 1; k < num_topics; k++)
	{
	    fprintf(fileptr, " %5.10f", corpus->docs[d].doc_gamma[k]);
	}
	fprintf(fileptr, "\n");
    }
    fclose(fileptr);
}

 void* thread_inference(void* thread_data){
    pthread_t tid;
    tid = pthread_self();
    Thread_Data* thread_data_ptr = (Thread_Data*) thread_data;
    corpus* corpus = thread_data_ptr->corpus;
    int start = thread_data_ptr->start;
    int end = thread_data_ptr->end;
    lda_model *model = thread_data_ptr->model;

    for (int i = start; i < end; i++) {
         doc_e_step(&corpus->docs[i], model);
    }
    printf("thread %u is over with %d documents. \n", (unsigned int)tid, end - start);
    return NULL;
 }

  void run_thread_inference(corpus* corpus, lda_model *model){
    int i;
    int num_threads = NUM_THREADS;
    pthread_t pthread_ts[num_threads];
    int num_docs = corpus->num_docs;
    int num_per_threads = num_docs/num_threads;
    Thread_Data ** thread_datas = malloc(sizeof(Thread_Data)*num_threads);
    for(i=0; i<num_threads-1; i++){
        thread_datas[i] = malloc(sizeof(Thread_Data));
        thread_datas[i]->corpus = corpus;
        thread_datas[i]->model = model;
        thread_datas[i]->start = i * num_per_threads;
        thread_datas[i]->end = (i+1)*num_per_threads;
        pthread_create(&pthread_ts[i], NULL, thread_inference, (void* )thread_datas[i]); 
    }

    thread_datas[i] = malloc(sizeof(Thread_Data));
    thread_datas[i]->corpus = corpus;
    thread_datas[i]->model = model;
    thread_datas[i]->start = i * num_per_threads;
    thread_datas[i]->end = num_docs;
    pthread_create(&pthread_ts[i], NULL, thread_inference, (void* )thread_datas[i]); 
        
    for (i = 0; i < num_threads; i++) pthread_join(pthread_ts[i],NULL);

    for (i = 0; i < num_threads; i++) free(thread_datas[i]);
    free(thread_datas);
}



void run_em(char* start, char* directory, corpus* corpus)
{

    int d, n,k;
    lda_model *model = NULL;

    char filename[100];

    lda_suffstats* ss = NULL;
    if (strcmp(start, "seeded")==0)
    {
        model = new_lda_model(corpus->num_terms, NTOPICS);
        ss = new_lda_suffstats(model);
        corpus_initialize_ss(ss, model, corpus);
        lda_mle(model, ss, 0);
        model->alpha = INITIAL_ALPHA;
    }
    else if (strcmp(start, "random")==0)
    {
        model = new_lda_model(corpus->num_terms, NTOPICS);
        ss = new_lda_suffstats(model);
        random_initialize_ss(ss, model);
        lda_mle(model, ss, 0);
        model->alpha = INITIAL_ALPHA;
    }
    else
    {
        model = load_lda_model(start);
        ss = new_lda_suffstats(model);
    }

    sprintf(filename,"%s/000",directory);
    save_lda_model(model, filename);

    // run expectation maximization

    int i = 0;
    double likelihood, likelihood_old = 0, converged = 1;
    sprintf(filename, "%s/likelihood.dat", directory);
    FILE* likelihood_file = fopen(filename, "w");

    while (((converged < 0) || (converged > EM_CONVERGED) || (i <= 2)) && (i <= EM_MAX_ITER))
    {
        i++; printf("**** em iteration %d ****\n", i);
        likelihood = 0;
        zero_initialize_ss(ss, model);

        // e-step
        run_thread_inference(corpus, model);

        /*
        for (d = 0; d < corpus->num_docs; d++)
        {
            if ((d % 1000) == 0) printf("document %d\n",d);
            likelihood2 += doc_e_step(&corpus->docs[i], model);
        }*/
        


        for (d = 0; d<corpus->num_docs; d++){
            document * doc = &(corpus->docs[d]);
            double * gamma = doc->doc_gamma;
            double ** phi = doc->doc_phi;
            double gamma_sum = 0;
            for (k = 0; k < model->num_topics; k++) {
                gamma_sum += gamma[k];
                ss->alpha_suffstats += digamma(gamma[k]);
                }
            ss->alpha_suffstats -= model->num_topics * digamma(gamma_sum);

            for (n = 0; n < doc->length; n++){
                for (k = 0; k < model->num_topics; k++){
                ss->class_word[k][doc->words[n]] += doc->counts[n]*phi[n][k];
                ss->class_total[k] += doc->counts[n]*phi[n][k];
                }
            }
        likelihood += doc->likelihood;
        ss->num_docs = ss->num_docs + 1;

        }

        // m-step

        lda_mle(model, ss, ESTIMATE_ALPHA);

        // check for convergence

        converged = (likelihood_old - likelihood) / (likelihood_old);
        if (converged < 0) VAR_MAX_ITER = VAR_MAX_ITER * 2;
        likelihood_old = likelihood;

        // output model and likelihood

        fprintf(likelihood_file, "%10.10f\t %5.5e\n", likelihood, converged);
        fflush(likelihood_file);
        if ((i % LAG) == 0)
        {
            sprintf(filename,"%s/%03d",directory, i);
            save_lda_model(model, filename);
            sprintf(filename,"%s/%03d.gamma",directory, i);
            save_gamma(filename, corpus, model->num_topics);
        }
    }

    // output the final model

    sprintf(filename,"%s/final",directory);
    save_lda_model(model, filename);
    sprintf(filename,"%s/final.gamma",directory);
    save_gamma(filename, corpus, model->num_topics);

    // output the word assignments (for visualization)

    sprintf(filename, "%s/word-assignments.dat", directory);
    FILE* w_asgn_file = fopen(filename, "w");
    run_thread_inference(corpus, model);

    write_word_assignment(w_asgn_file, corpus, model);

    fclose(w_asgn_file);
    fclose(likelihood_file);

    free(corpus);
    free(model);

}


void write_word_assignment(FILE* f, corpus* corpus, lda_model* model)
{
    int d;
    for (d = 0; d < corpus->num_docs; d++)
        {
            int n;
            document * doc = &(corpus->docs[d]);
            double ** phi = doc->doc_phi;
            fprintf(f, "%03d", doc->length);
            for (n = 0; n < doc->length; n++)
            {
                fprintf(f, " %04d:%02d",
                        doc->words[n], argmax(phi[n], model->num_topics));
            }
            fprintf(f, "\n");
            fflush(f);

        }
}

/*
 * read settings.
 *
 */

void read_settings(char* filename)
{
    FILE* fileptr;
    char alpha_action[100];
    fileptr = fopen(filename, "r");
    fscanf(fileptr, "var max iter %d\n", &VAR_MAX_ITER);
    fscanf(fileptr, "var convergence %f\n", &VAR_CONVERGED);
    fscanf(fileptr, "em max iter %d\n", &EM_MAX_ITER);
    fscanf(fileptr, "em convergence %f\n", &EM_CONVERGED);
    fscanf(fileptr, "num of threads %d \n", &NUM_THREADS);
    fscanf(fileptr, "alpha %s", alpha_action);
    if (strcmp(alpha_action, "fixed")==0)
    {
	ESTIMATE_ALPHA = 0;
    }
    else
    {
	ESTIMATE_ALPHA = 1;
    }
    fclose(fileptr);
}


/*
 * main
 *
 */

int main(int argc, char* argv[])
{
    // (est / inf) alpha k settings data (random / seed/ model) (directory / out)

    corpus* corpus;

    long t1;
    (void) time(&t1);
    seedMT(t1);
    // seedMT(4357U);

    if (argc > 1)
    {
        if (strcmp(argv[1], "est")==0)
        {
            INITIAL_ALPHA = atof(argv[2]);
            NTOPICS = atoi(argv[3]);
            read_settings(argv[4]);
            corpus = read_data(argv[5],NTOPICS);
            make_directory(argv[7]);
            run_em(argv[6], argv[7], corpus);
        }
       
    }
    else
    {
        printf("usage : lda est [initial alpha] [k] [settings] [data] [random/seeded/*] [directory]\n");
        printf("        lda inf [settings] [model] [data] [name]\n");
    }
    return(0);
}
