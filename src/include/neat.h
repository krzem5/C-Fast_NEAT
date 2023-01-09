#ifndef __NEAT_H__
#define __NEAT_H__
#include <stdint.h>



#define NEAT_GENOME_EDGE_STATE_ENABLED 0
#define NEAT_GENOME_EDGE_STATE_DISABLED 1
#define NEAT_GENOME_EDGE_STATE_NOT_CREATED 2



typedef uint8_t neat_genome_edge_state_t;



typedef struct _NEAT_GENOME_EDGE{
	float weight;
	neat_genome_edge_state_t state;
} neat_genome_edge_t;



typedef struct _NEAT_GENOME_NODE{
	float bias;
	float value;
} neat_genome_node_t;



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
	unsigned int surviving_population;
	neat_genome_t* genomes;
} neat_t;



void neat_init(unsigned int input_count,unsigned int output_count,unsigned int population,unsigned int surviving_population,neat_t* out);



void neat_genome_evaluate(const neat_t* neat,const neat_genome_t* genome,const float* in,float* out);



const neat_genome_t* neat_update(const neat_t* neat,float (*fitness_score_callback)(const neat_t*,const neat_genome_t*));



#endif
