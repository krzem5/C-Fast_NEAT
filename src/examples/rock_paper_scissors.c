#include <math.h>
#include <neat.h>
#include <stdio.h>



float rock_paper_scissors_fitness_score_callback(const neat_t* neat,const neat_genome_t* genome){
	float out=0.0f;
	for (unsigned int i=0;i<3;i++){
		float genome_in[2]={i/2.0f,1.0f};
		float genome_out;
		neat_genome_evaluate(neat,genome,genome_in,&genome_out);
		float diff=genome_out-((i+1)%3)/2.0f;
		out+=diff*diff;
	}
	return 1/(1+sqrtf(out));
}



void rock_paper_scissors_end_callback(const neat_t* neat,const neat_genome_t* genome){
	for (unsigned int i=0;i<3;i++){
		float genome_in[2]={i/2.0f,1.0f};
		float genome_out;
		neat_genome_evaluate(neat,genome,genome_in,&genome_out);
		printf("%u -> %.2f\n",i,genome_out*2);
	}
}
