#include <math.h>
#include <neat.h>
#include <stdlib.h>



#define BREED_MUTATION_CHANCE 0.5f
#define NODE_ADD_CHANCE 0.01f
#define EDGE_ADD_CHANCE 0.09f
#define WEIGHT_ADJUST_CHANCE 0.4f
#define WEIGHT_SET_CHANCE 0.1f
#define BIAS_ADJUST_CHANCE 0.3f
#define BIAS_SET_CHANCE 0.1f



static inline float _sigmoid(float x){
	return 1/(1+exp(-x));
}



static inline float _random_uniform(void){
	return ((float)rand())/RAND_MAX;
}



static inline unsigned int _random_int(unsigned int max){
	return rand()%max;
}



static void _adjust_genome_node_count(neat_genome_t* genome,unsigned int new_node_count){
	if (genome->node_count==new_node_count){
		return;
	}
	genome->node_count=new_node_count;
	genome->nodes=realloc(genome->nodes,new_node_count*sizeof(neat_genome_node_t));
	genome->edges=realloc(genome->edges,new_node_count*new_node_count*sizeof(neat_genome_edge_t));
}



static unsigned int _get_random_edge_index(const neat_genome_t* genome){
	unsigned int end=genome->node_count*genome->node_count;
	unsigned int out=_random_int(end);
	while ((genome->edges+out)->state!=NEAT_GENOME_EDGE_STATE_ENABLED){
		out++;
		if (out==end){
			out=0;
		}
	}
	return out;
}



static void _quicksort(neat_genome_t* genomes,unsigned int low,unsigned int high,unsigned int max){
	if (low>=high||low>=max){
		return;
	}
	float pivot=(genomes+high)->fitness_score;
	unsigned int i=low;
	for (unsigned int j=low;j<high;j++){
		if ((genomes+j)->fitness_score>pivot){
			neat_genome_t tmp=genomes[j];
			genomes[j]=genomes[i];
			genomes[i]=tmp;
			i++;
		}
	}
	neat_genome_t tmp=genomes[high];
	genomes[high]=genomes[i];
	genomes[i]=tmp;
	if (i>0){
		_quicksort(genomes,low,i-1,max);
	}
	_quicksort(genomes,i+1,high,max);
}



void neat_init(unsigned int input_count,unsigned int output_count,unsigned int population,unsigned int surviving_population,neat_t* out){
	out->input_count=input_count;
	out->output_count=output_count;
	out->population=population;
	out->surviving_population=surviving_population;
	out->genomes=malloc(population*sizeof(neat_genome_t));
	unsigned int node_count=input_count+output_count;
	neat_genome_t* genome=out->genomes;
	for (unsigned int i=0;i<population;i++){
		genome->node_count=node_count;
		genome->nodes=malloc(node_count*sizeof(neat_genome_node_t));
		genome->edges=malloc(node_count*node_count*sizeof(neat_genome_edge_t));
		unsigned int l=0;
		for (unsigned int j=0;j<node_count;j++){
			(genome->nodes+j)->bias=0.0f;
			for (unsigned int k=0;k<node_count;k++){
				(genome->edges+l)->state=NEAT_GENOME_EDGE_STATE_NOT_CREATED;
				l++;
			}
		}
		for (unsigned int j=0;j<input_count;j++){
			l=j*node_count+input_count;
			for (unsigned int k=input_count;k<node_count;k++){
				(genome->edges+l)->weight=_random_uniform()*2-1;
				(genome->edges+l)->state=NEAT_GENOME_EDGE_STATE_ENABLED;
				l++;
			}
		}
		genome++;
	}
}



void neat_deinit(const neat_t* neat){
	const neat_genome_t* genome=neat->genomes;
	for (unsigned int i=0;i<neat->population;i++){
		free(genome->nodes);
		free(genome->edges);
		genome++;
	}
	free(neat->genomes);
}



void neat_genome_evaluate(const neat_t* neat,const neat_genome_t* genome,const float* in,float* out){
	for (unsigned int i=0;i<neat->input_count;i++){
		(genome->nodes+i)->value=*in;
		in++;
	}
	for (unsigned int i=neat->input_count;i<genome->node_count;i++){
		(genome->nodes+i)->value=0.0f;
	}
	for (unsigned int i=neat->input_count+neat->output_count;i<genome->node_count;i++){
		neat_genome_node_t* node=genome->nodes+i;
		float value=node->bias;
		unsigned int j=i;
		for (unsigned int k=0;k<genome->node_count;k++){
			if ((genome->edges+j)->state==NEAT_GENOME_EDGE_STATE_ENABLED){
				value+=(genome->edges+j)->weight*(genome->nodes+k)->value;
			}
			j+=genome->node_count;
		}
		node->value=_sigmoid(value);
	}
	for (unsigned int i=neat->input_count;i<neat->input_count+neat->output_count;i++){
		neat_genome_node_t* node=genome->nodes+i;
		float value=node->bias;
		unsigned int j=i;
		for (unsigned int k=0;k<genome->node_count;k++){
			if ((genome->edges+j)->state==NEAT_GENOME_EDGE_STATE_ENABLED){
				value+=(genome->edges+j)->weight*(genome->nodes+k)->value;
			}
			j+=genome->node_count;
		}
		node->value=_sigmoid(value);
		*out=node->value;
		out++;
	}
}



const neat_genome_t* neat_update(const neat_t* neat,float (*fitness_score_callback)(const neat_t*,const neat_genome_t*)){
	const neat_genome_t* best=NULL;
	neat_genome_t* genome=neat->genomes;
	for (unsigned int i=0;i<neat->population;i++){
		genome->fitness_score=fitness_score_callback(neat,genome);
		if (!best||genome->fitness_score>best->fitness_score){
			best=genome;
		}
		genome++;
	}
	_quicksort(neat->genomes,0,neat->population-1,neat->surviving_population);
	neat_genome_t* child=neat->genomes+neat->surviving_population;
	for (unsigned int idx=neat->surviving_population;idx<neat->population;idx++){
		const neat_genome_t* random_genome=neat->genomes+_random_int(neat->surviving_population);
		if (_random_uniform()<=BREED_MUTATION_CHANCE){
			float value=_random_uniform()*(NODE_ADD_CHANCE+EDGE_ADD_CHANCE+WEIGHT_ADJUST_CHANCE+WEIGHT_SET_CHANCE+BIAS_ADJUST_CHANCE+BIAS_SET_CHANCE);
			_adjust_genome_node_count(child,random_genome->node_count+(value<=NODE_ADD_CHANCE));
			unsigned int k=0;
			unsigned int l=0;
			for (unsigned int i=0;i<random_genome->node_count;i++){
				for (unsigned int j=0;j<random_genome->node_count;j++){
					*(child->edges+l)=*(random_genome->edges+k);
					k++;
					l++;
				}
				if (value<=NODE_ADD_CHANCE){
					l++;
				}
				(child->nodes+i)->bias=(random_genome->nodes+i)->bias;
			}
			if (value<=NODE_ADD_CHANCE){
				unsigned int idx=_get_random_edge_index(random_genome);
				unsigned int i=idx/random_genome->node_count;
				unsigned int j=idx%random_genome->node_count;
				neat_genome_edge_t* edge=child->edges+idx;
				edge->state=NEAT_GENOME_EDGE_STATE_DISABLED;
				(child->edges+i*child->node_count+random_genome->node_count)->weight=1.0f;
				(child->edges+i*child->node_count+random_genome->node_count)->state=NEAT_GENOME_EDGE_STATE_ENABLED;
				(child->edges+random_genome->node_count*child->node_count+j)->weight=edge->weight;
				(child->edges+random_genome->node_count*child->node_count+j)->state=NEAT_GENOME_EDGE_STATE_ENABLED;
				(child->nodes+random_genome->node_count)->bias=0.0f;
			}
			else if (value<=NODE_ADD_CHANCE+EDGE_ADD_CHANCE){
				unsigned int i=_random_int(random_genome->node_count-neat->output_count);
				if (i>=neat->input_count){
					i+=neat->output_count;
				}
				unsigned int j=_random_int(random_genome->node_count-neat->input_count)+neat->input_count;
				if (i==j){
					j++;
					if (j==random_genome->node_count){
						j-=2;
					}
				}
				unsigned int k=i*random_genome->node_count+j;
				if ((child->edges+k)->state==NEAT_GENOME_EDGE_STATE_NOT_CREATED){
					(child->edges+k)->weight=_random_uniform()*2-1;
				}
				(child->edges+k)->state=NEAT_GENOME_EDGE_STATE_ENABLED;
			}
			else if (value<=NODE_ADD_CHANCE+EDGE_ADD_CHANCE+WEIGHT_ADJUST_CHANCE){
				(child->edges+_get_random_edge_index(random_genome))->weight+=_random_uniform()*2-1;
			}
			else if (value<=NODE_ADD_CHANCE+EDGE_ADD_CHANCE+WEIGHT_ADJUST_CHANCE+WEIGHT_SET_CHANCE){
				(child->edges+_get_random_edge_index(random_genome))->weight=_random_uniform()*2-1;
			}
			else if (value<=NODE_ADD_CHANCE+EDGE_ADD_CHANCE+WEIGHT_ADJUST_CHANCE+WEIGHT_SET_CHANCE+BIAS_ADJUST_CHANCE){
				(child->nodes+_random_int(random_genome->node_count-neat->input_count)+neat->input_count)->bias+=_random_uniform()*2-1;
			}
			else{
				(child->nodes+_random_int(random_genome->node_count-neat->input_count)+neat->input_count)->bias=_random_uniform()*2-1;
			}
		}
		else{
			const neat_genome_t* second_random_genome=neat->genomes+_random_int(neat->surviving_population);
			if (second_random_genome->fitness_score>random_genome->fitness_score){
				const neat_genome_t* tmp=random_genome;
				random_genome=second_random_genome;
				second_random_genome=tmp;
			}
			_adjust_genome_node_count(child,random_genome->node_count);
			int last_disabled_edge=0;
			unsigned int k=0;
			for (unsigned int i=0;i<random_genome->node_count;i++){
				for (unsigned int j=0;j<random_genome->node_count;j++){
					const neat_genome_edge_t* edge=random_genome->edges+k;
					if (i<second_random_genome->node_count&&j<second_random_genome->node_count){
						const neat_genome_edge_t* second_edge=second_random_genome->edges+i*second_random_genome->node_count+j;
						if (second_edge->state!=NEAT_GENOME_EDGE_STATE_NOT_CREATED&&_random_uniform()<=0.5f){
							edge=second_edge;
						}
					}
					*(child->edges+k)=*edge;
					if (edge->state==NEAT_GENOME_EDGE_STATE_DISABLED&&last_disabled_edge!=-1){
						last_disabled_edge=k;
					}
					else if (edge->state==NEAT_GENOME_EDGE_STATE_ENABLED){
						last_disabled_edge=-1;
					}
					k++;
				}
				(child->nodes+i)->bias=((i<second_random_genome->node_count&&_random_uniform()<=0.5f?second_random_genome:random_genome)->nodes+i)->bias;
			}
			if (last_disabled_edge!=-1){
				(child->edges+last_disabled_edge)->state=NEAT_GENOME_EDGE_STATE_ENABLED;
			}
		}
		child++;
	}
	return best;
}
