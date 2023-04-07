#include <math.h>
#include <neat.h>
#include <stdio.h>



#define OPTIMIZE_SIZE 1



float xor3_fitness_score_callback(neat_t* neat,const neat_genome_t* genome){
	float out=0.0f;
	for (unsigned int i=0;i<4;i++){
		float genome_in1[8]={(float)(i&1)+neat_random_float(neat),(float)(i>>1)+neat_random_float(neat),0.0f+neat_random_float(neat),0.0f,0.0f,0.0f,0.0f,0.0f};
		float genome_in2[8]={(float)(i&1)+neat_random_float(neat),(float)(i>>1)+neat_random_float(neat),1.0f+neat_random_float(neat),0.0f,0.0f,0.0f,0.0f,0.0f};
		float genome_out1;
		float genome_out2;
		neat_genome_evaluate(neat,genome,genome_in1,genome_in2,&genome_out1,&genome_out2);
		float diff=genome_out1*0.5f+0.5f-((float)((i&1)^(i>>1)));
		out+=diff*diff;
		diff=genome_out2*0.5f+0.5f-((float)((i&1)^(i>>1)^1));
		out+=diff*diff;
	}
	if (OPTIMIZE_SIZE&&out<0.25f){
		return 1/(1+sqrtf(out))-0.06f*(expf(((genome->node_count>>3)-1)*0.75f)-1)-0.075f*genome->_enabled_node_count/((float)genome->node_count);
	}
	return 1/(1+sqrtf(out));
}



void xor3_end_callback(const neat_t* neat,const neat_genome_t* genome){
	for (unsigned int i=0;i<4;i++){
		float genome_in1[8]={(float)(i&1),(float)(i>>1),0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
		float genome_in2[8]={(float)(i&1),(float)(i>>1),1.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
		float genome_out1;
		float genome_out2;
		neat_genome_evaluate(neat,genome,genome_in1,genome_in2,&genome_out1,&genome_out2);
		printf("%u^%u^0=%+.3f\n%u^%u^1=%+.3f\n",i&1,i>>1,genome_out1*0.5f+0.5f,i&1,i>>1,genome_out2*0.5f+0.5f);
	}
}
