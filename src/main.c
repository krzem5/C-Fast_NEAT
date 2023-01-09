#include <example.h>
#include <neat.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



int main(void){
	srand(time(NULL));
	const example_t* example=example_get("xor3");
	neat_t neat;
	neat_init(example->input_count,example->output_count,example->population,example->surviving_population,&neat);
	const neat_genome_t* best=NULL;
	for (unsigned int i=0;i<10000;i++){
		best=neat_update(&neat,example->fitness_score_callback);
		printf("%.3f\n",best->fitness_score);
		if (best->fitness_score>=example->max_fitness_score){
			break;
		}
	}
	example->end_callback(&neat,best);
	return 0;
}
