#include <example.h>
#include <neat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>



static unsigned long int get_time(void){
	struct timespec tm;
	clock_gettime(CLOCK_REALTIME,&tm);
	return tm.tv_sec*1000000000+tm.tv_nsec;
}



int main(void){
	srand(get_time()&0xffffffff);
	const example_t* example=example_get("xor3");
	neat_t neat;
	neat_init(example->input_count,example->output_count,example->population,example->fitness_score_callback,&neat);
	unsigned long int start=get_time();
	const neat_genome_t* best=NULL;
	unsigned int i=0;
	for (;i<10000;i++){
		best=neat_update(&neat);
		printf("%.2f%%\n",best->fitness_score*100);
		if (best->fitness_score>=example->max_fitness_score){
			break;
		}
	}
	double delta_time=(get_time()-start)*1e-9;
	printf("Time: %.3fs, Iterations: %u, Time per Iteration: %.6fs\n",delta_time,i,delta_time/i);
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
