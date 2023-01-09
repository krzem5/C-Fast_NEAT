#include <math.h>
#include <neat.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



static float fitness_score_callback(const neat_t* neat,const neat_genome_t* genome){
	float out=0.0f;
	for (unsigned int i=0;i<8;i++){
		float genome_in[3]={i&1,(i>>1)&1,i>>2};
		float genome_out;
		neat_genome_evaluate(neat,genome,genome_in,&genome_out);
		float diff=genome_out-((i&1)^((i>>1)&1)^(i>>2));
		out+=diff*diff;
	}
	return 1/(1+sqrtf(out));
}



int main(void){
	srand(time(NULL));
	neat_t neat;
	neat_init(3,1,500,250,&neat);
	const neat_genome_t* best=NULL;
	for (unsigned int i=0;i<10000;i++){
		best=neat_update(&neat,fitness_score_callback);
		printf("%.2f%%\n",best->fitness_score*100);
		if (best->fitness_score>=0.999f){
			break;
		}
	}
	for (unsigned int i=0;i<8;i++){
		float genome_in[3]={i&1,(i>>1)&1,i>>2};
		float genome_out;
		neat_genome_evaluate(&neat,best,genome_in,&genome_out);
		printf("%u^%u^%u=%.3f\n",i&1,(i>>1)&1,i>>2,genome_out);
	}
	return 0;
}
