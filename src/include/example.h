#ifndef __EXAMPLE_H__
#define __EXAMPLE_H__ 1
#include <neat.h>



typedef struct _EXAMPLE{
	const char* name;
	unsigned int input_count;
	unsigned int output_count;
	unsigned int population;
	unsigned int surviving_population;
	float max_fitness_score;
	float (*fitness_score_callback)(const neat_t*,const neat_genome_t*);
	void (*end_callback)(const neat_t*,const neat_genome_t*);
} example_t;



const example_t* example_get(const char* name);



#endif
