#ifndef __NEAT_H__
#define __NEAT_H__
#include <stdint.h>



typedef struct _NEAT_GENOME_NODE{
	float bias;
} neat_genome_node_t;



typedef struct _NEAT_GENOME_EDGE{
	float weight;
} neat_genome_edge_t;



typedef struct _NEAT_GENOME{
	unsigned int node_count;
	float fitness_score;
	neat_genome_node_t* nodes;
	neat_genome_edge_t* edges;
} neat_genome_t;



typedef struct _NEAT{
	unsigned int input_count;
	unsigned int output_count;
	unsigned int population;
	float _last_average_fitness_score;
	neat_genome_t* genomes;
	neat_genome_node_t* _node_data;
	neat_genome_edge_t* _edge_data;
} neat_t;



typedef struct _NEAT_MODEL_NODE{
	float bias;
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



void neat_init(unsigned int input_count,unsigned int output_count,unsigned int population,neat_t* out);



void neat_deinit(const neat_t* neat);



void neat_genome_evaluate(const neat_t* neat,const neat_genome_t* genome,const float* in,float* out);



const neat_genome_t* neat_update(neat_t* neat,float (*fitness_score_callback)(const neat_t*,const neat_genome_t*));



void neat_extract_model(const neat_t* neat,const neat_genome_t* genome,neat_model_t* out);



void neat_deinit_model(const neat_model_t* model);



void neat_save_model(const neat_model_t* model,const char* file_path);



#endif
