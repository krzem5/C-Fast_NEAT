#include <math.h>
#include <neat.h>
#include <stdio.h>



float xor2_fitness_score_callback(const neat_t* neat,const neat_genome_t* genome){
	float out=0.0f;
	for (unsigned int i=0;i<4;i++){
		float genome_in[2]={(float)(i&1),(float)(i>>1)};
		float genome_out;
		neat_genome_evaluate(neat,genome,genome_in,&genome_out);
		float diff=genome_out*0.5f+0.5f-((float)((i&1)^(i>>1)));
		out+=diff*diff;
	}
	return 1/(1+sqrtf(out));
}



void xor2_end_callback(const neat_t* neat,const neat_genome_t* genome){
	for (unsigned int i=0;i<4;i++){
		float genome_in[2]={(float)(i&1),(float)(i>>1)};
		float genome_out;
		neat_genome_evaluate(neat,genome,genome_in,&genome_out);
		printf("%u^%u=%.3f\n",i&1,i>>1,genome_out*0.5f+0.5f);
	}
}
