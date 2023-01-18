#include <example.h>
#include <neat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



#define DECLARE_EXAMPLE_FN(name) float name##_fitness_score_callback(const neat_t*,const neat_genome_t*);void name##_end_callback(const neat_t*,const neat_genome_t*)
#define DECLARE_EXAMPLE(name,input_count,output_count,population,max_fitness_score) {#name,input_count,output_count,population,max_fitness_score,name##_fitness_score_callback,name##_end_callback}



DECLARE_EXAMPLE_FN(cartpole);
DECLARE_EXAMPLE_FN(language);
DECLARE_EXAMPLE_FN(rock_paper_scissors);
DECLARE_EXAMPLE_FN(xor2);
DECLARE_EXAMPLE_FN(xor3);



static const example_t _example_data[]={
	DECLARE_EXAMPLE(cartpole,4,1,500,0.999f),
	DECLARE_EXAMPLE(language,312,2,500,0.999f),
	DECLARE_EXAMPLE(rock_paper_scissors,1,1,500,0.999f),
	DECLARE_EXAMPLE(xor2,2,1,128,0.95f),
	DECLARE_EXAMPLE(xor3,3,1,128,0.95f),
	{
		NULL
	}
};



const example_t* example_get(const char* name){
	const example_t* out=_example_data;
	while (out->name){
		if (!strcmp(out->name,name)){
			return out;
		}
		out++;
	}
	printf("Unknown example '%s'. Possible values:\n",name);
	out=_example_data;
	while (out->name){
		printf("- %s\n",out->name);
		out++;
	}
	return NULL;
}



unsigned int example_random_below(unsigned int max){
	return rand()%max;
}



float example_random_uniform(float min,float max){
	return min+(max-min)*rand()/RAND_MAX;
}
