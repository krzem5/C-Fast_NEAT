#include <math.h>
#include <neat.h>
#include <stdio.h>



float xor3_fitness_score_callback(const neat_t* neat,const neat_genome_t* genome){
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



void xor3_end_callback(const neat_t* neat,const neat_genome_t* genome){
	for (unsigned int i=0;i<8;i++){
		float genome_in[3]={i&1,(i>>1)&1,i>>2};
		float genome_out;
		neat_genome_evaluate(neat,genome,genome_in,&genome_out);
		printf("%u^%u^%u=%.3f\n",i&1,(i>>1)&1,i>>2,genome_out);
	}
}