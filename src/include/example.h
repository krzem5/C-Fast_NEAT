#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_ 1
#include <neat.h>



typedef struct _EXAMPLE{
	const char* name;
	unsigned int input_count;
	unsigned int output_count;
	unsigned int population;
	float max_fitness_score;
	float (*fitness_score_callback)(neat_t*,const neat_genome_t*);
	void (*end_callback)(const neat_t*,const neat_genome_t*);
} example_t;



const example_t* example_get(const char* name);



unsigned int example_random_below(unsigned int max);



float example_random_uniform(float min,float max);



#endif
