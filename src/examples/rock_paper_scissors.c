#include <math.h>
#include <neat.h>
#include <stdio.h>



static const char* names[3]={"rock","paper","scissors"};



float rock_paper_scissors_fitness_score_callback(neat_t* neat,const neat_genome_t* genome){
	float out=0.0f;
	for (unsigned int i=0;i<3;i++){
		float genome_in[8]={i/2.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
		float genome_out;
		neat_genome_evaluate(neat,genome,genome_in,genome_in,&genome_out,&genome_out);
		float diff=genome_out*0.5f+0.5f-((i+1)%3)/2.0f;
		out+=diff*diff;
	}
	return 1/(1+sqrtf(out));
}



void rock_paper_scissors_end_callback(const neat_t* neat,const neat_genome_t* genome){
	for (unsigned int i=0;i<3;i++){
		float genome_in[8]={i/2.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
		float genome_out;
		neat_genome_evaluate(neat,genome,genome_in,genome_in,&genome_out,&genome_out);
		printf("%s -> %s (%u -> %.2f)\n",names[i],names[(unsigned int)roundf(genome_out+1-0.45f)],i,genome_out+1);
	}
}
