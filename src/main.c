#include <example.h>
#include <neat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>



int main(void){
	srand((unsigned int)time(NULL));
	const example_t* example=example_get("rock_paper_scissors");
	neat_t neat;
	neat_init(example->input_count,example->output_count,example->population,example->fitness_score_callback,&neat);
	const neat_genome_t* best=NULL;
	for (unsigned int i=0;i<10000;i++){
		best=neat_update(&neat);
		printf("%.2f%%\n",best->fitness_score*100);
		if (best->fitness_score>=example->max_fitness_score){
			break;
		}
	}
	example->end_callback(&neat,best);
	neat_model_t model;
	neat_extract_model(&neat,best,&model);
	neat_deinit(&neat);
	char path[4096]="build/";
	strcat(strcat(path,example->name),".neat");
	neat_save_model(&model,path);
	neat_deinit_model(&model);
	return 0;
}
