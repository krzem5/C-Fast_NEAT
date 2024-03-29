#ifndef _NEAT_H_
#define _NEAT_H_ 1
#include <stdint.h>



struct _NEAT;



typedef struct _NEAT_GENOME_NODE{
	float bias;
	uint16_t activation_function;
	uint16_t enabled;
} neat_genome_node_t;



typedef struct _NEAT_GENOME_EDGE{
	float weight;
} neat_genome_edge_t;



typedef struct _NEAT_GENOME{
	unsigned int node_count;
	float fitness_score;
	neat_genome_node_t* nodes;
	neat_genome_edge_t* edges;
	unsigned int _node_count_sq;
	unsigned int _enabled_node_count;
} neat_genome_t;



typedef float (*neat_fitness_score_callback_t)(struct _NEAT*,const neat_genome_t*);



typedef struct _NEAT_PRNG{
	unsigned int data[64];
	unsigned int count;
} neat_prng_t;



typedef struct _NEAT{
	unsigned int input_count;
	unsigned int output_count;
	unsigned int population;
	float _last_average_fitness_score;
	neat_fitness_score_callback_t fitness_score_callback;
	neat_genome_t* genomes;
	neat_genome_node_t* _genome_data;
	neat_genome_node_t* _node_data;
	neat_genome_edge_t* _edge_data;
	float _fitness_score_sum;
	float _last_best_genome_fitness;
	unsigned int stale_iteration_count;
	neat_prng_t _prng_state;
} neat_t;



typedef struct _NEAT_MODEL_NODE{
	float bias;
	uint16_t activation_function;
	uint16_t enabled;
} neat_model_node_t;



typedef struct _NEAT_MODEL_EDGE{
	float weight;
} neat_model_edge_t;



typedef struct _NEAT_MODEL{
	unsigned int input_count;
	unsigned int output_count;
	unsigned int node_count;
	unsigned int edge_count;
	neat_model_node_t* nodes;
	neat_model_edge_t* edges;
} neat_model_t;



void neat_init(unsigned int input_count,unsigned int output_count,unsigned int population,neat_fitness_score_callback_t fitness_score_callback,neat_t* out);



void neat_deinit(neat_t* neat);



void neat_reset_genomes(neat_t* neat);



void neat_genome_evaluate(const neat_t* neat,const neat_genome_t* genome,const float* in1,const float* in2,float* out1,float* out2);



float neat_update(neat_t* neat);



const neat_genome_t* neat_get_best(const neat_t* neat);



void neat_extract_model(const neat_t* neat,const neat_genome_t* genome,neat_model_t* out);



void neat_deinit_model(const neat_model_t* model);



_Bool neat_save_model(const neat_model_t* model,const char* file_path);



_Bool neat_load_model(const char* file_path,neat_model_t* out);



void neat_model_evaluate(const neat_model_t* model,const float* in1,const float* in2,float* out1,float* out2);



float neat_random_float(neat_t* neat);



#endif
