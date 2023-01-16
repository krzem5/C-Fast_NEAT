#include <math.h>
#include <neat.h>
#include <stdio.h>
#include <stdlib.h>



#define NODE_ADD_CHANCE 2
#define WEIGHT_ADJUST_CHANCE 40
#define WEIGHT_SET_CHANCE 10
#define BIAS_ADJUST_CHANCE 30
#define BIAS_SET_CHANCE 10
#define MAX_STALE_FITNESS_DIFFERENCE 1e-6f



typedef struct _NEAT_MODEL_FILE_HEADER{
	unsigned int input_count;
	unsigned int output_count;
	unsigned int node_count;
	unsigned int edge_count;
} neat_model_file_header_t;



typedef struct _NEAT_MODEL_FILE_EDGE{
	unsigned int index;
	float weight;
} neat_model_file_edge_t;



static inline unsigned int _random_uint32(void){
	return (rand()&0xff)|((rand()&0xff)<<8)|((rand()&0xff)<<16)|((rand()&0xff)<<24);
}



static inline float _random_uniform_rescaled(void){
	return (float)(((int32_t)_random_uint32())>>7)*0x1p-24f;
}



static void _adjust_genome_node_count(neat_genome_t* genome,unsigned int new_node_count){
	if (genome->node_count==new_node_count){
		return;
	}
	genome->node_count=new_node_count;
	genome->nodes=realloc(genome->nodes,new_node_count*sizeof(neat_genome_node_t));
	genome->edges=realloc(genome->edges,new_node_count*new_node_count*sizeof(neat_genome_edge_t));
}



void neat_init(unsigned int input_count,unsigned int output_count,unsigned int population,neat_t* out){
	out->input_count=input_count;
	out->output_count=output_count;
	out->population=population;
	out->_last_average_fitness_score=-1e8f;
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
				(genome->edges+l)->weight=_random_uniform_rescaled();
				l++;
			}
		}
		genome++;
	}
	out->_evaluation_buffer=malloc(node_count*sizeof(float));
	out->_evaluation_buffer_size=node_count;
}



void neat_deinit(const neat_t* neat){
	const neat_genome_t* genome=neat->genomes;
	for (unsigned int i=0;i<neat->population;i++){
		free(genome->nodes);
		free(genome->edges);
		genome++;
	}
	free(neat->genomes);
	free(neat->_evaluation_buffer);
}



void neat_genome_evaluate(const neat_t* neat,const neat_genome_t* genome,const float* in,float* out){
	for (unsigned int i=0;i<neat->input_count;i++){
		neat->_evaluation_buffer[i]=*in;
		in++;
	}
	for (unsigned int i=neat->input_count;i<genome->node_count;i++){
		neat->_evaluation_buffer[i]=0.0f;
	}
	const neat_genome_edge_t* edge=genome->edges+neat->input_count*genome->node_count;
	for (unsigned int i=neat->input_count;i<genome->node_count;i++){
		float value=(genome->nodes+i)->bias;
		for (unsigned int k=0;k<genome->node_count;k++){
			value+=edge->weight*neat->_evaluation_buffer[k];
			edge++;
		}
		neat->_evaluation_buffer[i]=tanhf(value);
		if (i>=genome->node_count-neat->output_count){
			*out=neat->_evaluation_buffer[i];
			out++;
		}
	}
}



const neat_genome_t* neat_update(neat_t* neat,float (*fitness_score_callback)(const neat_t*,const neat_genome_t*)){
	neat_genome_t* genome=neat->genomes;
	float average=0;
	const neat_genome_t* best_genome=NULL;
	for (unsigned int i=0;i<neat->population;i++){
		genome->fitness_score=fitness_score_callback(neat,genome);
		average+=genome->fitness_score;
		if (!best_genome||genome->fitness_score>best_genome->fitness_score){
			best_genome=genome;
		}
		genome++;
	}
	average/=neat->population;
	_Bool stale=fabs(neat->_last_average_fitness_score-average)<MAX_STALE_FITNESS_DIFFERENCE;
	neat->_last_average_fitness_score=average;
	neat_genome_t* start_genome=neat->genomes;
	neat_genome_t* end_genome=genome;
	genome=start_genome;
	for (unsigned int i=0;i<neat->population;i++){
		if (genome->fitness_score>=average){
			if (start_genome==genome){
				genome++;
			}
			else{
				neat_genome_t tmp=*genome;
				*genome=*start_genome;
				*start_genome=tmp;
			}
			start_genome++;
		}
		else{
			end_genome--;
			if (end_genome==genome){
				genome--;
			}
			else{
				neat_genome_t tmp=*genome;
				*genome=*end_genome;
				*end_genome=tmp;
			}
		}
	}
	if (stale||start_genome==neat->genomes){
		start_genome=neat->genomes+1;
	}
	neat_genome_t* child=start_genome;
	for (unsigned int idx=(start_genome-neat->genomes);idx<neat->population;idx++){
		const neat_genome_t* random_genome=neat->genomes+(_random_uint32()%idx);
		if (stale||_random_uint32()&2){
			unsigned int action=_random_uint32()%(NODE_ADD_CHANCE+WEIGHT_ADJUST_CHANCE+WEIGHT_SET_CHANCE+BIAS_ADJUST_CHANCE+BIAS_SET_CHANCE);
			_Bool add_node=action<=NODE_ADD_CHANCE;
			_adjust_genome_node_count(child,random_genome->node_count+add_node);
			if (add_node){
				if (child->node_count>neat->_evaluation_buffer_size){
					neat->_evaluation_buffer_size=child->node_count;
					neat->_evaluation_buffer=realloc(neat->_evaluation_buffer,neat->_evaluation_buffer_size*sizeof(float));
				}
				unsigned int insert_index=random_genome->node_count-neat->output_count;
				const neat_genome_node_t* random_genome_node=random_genome->nodes;
				const neat_genome_edge_t* random_genome_edge=random_genome->edges;
				unsigned int k=0;
				for (unsigned int i=0;i<child->node_count;i++){
					if (i==insert_index){
						(child->nodes+i)->bias=0.0f;
					}
					else{
						(child->nodes+i)->bias=random_genome_node->bias;
						random_genome_node++;
					}
					for (unsigned int j=0;j<child->node_count;j++){
						if (i==insert_index||j==insert_index){
							(child->edges+k)->weight=_random_uniform_rescaled();
						}
						else{
							(child->edges+k)->weight=random_genome_edge->weight;
							random_genome_edge++;
						}
						k++;
					}
				}
			}
			else{
				unsigned int k=0;
				for (unsigned int i=0;i<random_genome->node_count;i++){
					(child->nodes+i)->bias=(random_genome->nodes+i)->bias;
					for (unsigned int j=0;j<random_genome->node_count;j++){
						(child->edges+k)->weight=(random_genome->edges+k)->weight;
						k++;
					}
				}
				if (action<=NODE_ADD_CHANCE+WEIGHT_ADJUST_CHANCE){
					(child->edges+(_random_uint32()%(random_genome->node_count*random_genome->node_count)))->weight+=_random_uniform_rescaled();
				}
				else if (action<=NODE_ADD_CHANCE+WEIGHT_ADJUST_CHANCE+WEIGHT_SET_CHANCE){
					(child->edges+(_random_uint32()%(random_genome->node_count*random_genome->node_count)))->weight=_random_uniform_rescaled();
				}
				else if (action<=NODE_ADD_CHANCE+WEIGHT_ADJUST_CHANCE+WEIGHT_SET_CHANCE+BIAS_ADJUST_CHANCE){
					(child->nodes+(_random_uint32()%(random_genome->node_count)))->bias+=_random_uniform_rescaled();
				}
				else{
					(child->nodes+(_random_uint32()%(random_genome->node_count)))->bias=_random_uniform_rescaled();
				}
			}
		}
		else{
			const neat_genome_t* second_random_genome=neat->genomes+(_random_uint32()%idx);
			_adjust_genome_node_count(child,random_genome->node_count);
			unsigned int k=0;
			for (unsigned int i=0;i<random_genome->node_count;i++){
				for (unsigned int j=0;j<random_genome->node_count;j++){
					(child->edges+k)->weight=(i<second_random_genome->node_count&&j<second_random_genome->node_count&&(_random_uint32()&1)?random_genome->edges+k:second_random_genome->edges+i*second_random_genome->node_count+j)->weight;
					k++;
				}
				(child->nodes+i)->bias=((i<second_random_genome->node_count&&(_random_uint32()&1)?second_random_genome:random_genome)->nodes+i)->bias;
			}
		}
		child++;
	}
	return best_genome;
}



void neat_extract_model(const neat_t* neat,const neat_genome_t* genome,neat_model_t* out){
	out->input_count=neat->input_count;
	out->output_count=neat->output_count;
	out->node_count=genome->node_count;
	out->edge_count=0;
	out->nodes=malloc(out->node_count*sizeof(neat_model_node_t));
	out->edges=malloc(out->node_count*out->node_count*sizeof(neat_model_edge_t));
	const neat_genome_edge_t* genome_edge=genome->edges;
	neat_model_edge_t* edge=out->edges;
	for (unsigned int i=0;i<out->node_count;i++){
		(out->nodes+i)->bias=(genome->nodes+i)->bias;
		for (unsigned int j=0;j<out->node_count;j++){
			edge->weight=genome_edge->weight;
			if (edge->weight!=0.0f){
				out->edge_count++;
			}
			genome_edge++;
			edge++;
		}
	}
}



void neat_deinit_model(const neat_model_t* model){
	free(model->nodes);
	free(model->edges);
}



void neat_save_model(const neat_model_t* model,const char* file_path){
	FILE* file=fopen(file_path,"wb");
	neat_model_file_header_t header={
		model->input_count,
		model->output_count,
		model->node_count,
		model->edge_count
	};
	if (fwrite(&header,sizeof(header),1,file)!=1){
		goto _error;
	}
	const neat_model_node_t* node=model->nodes+model->input_count;
	for (unsigned int i=model->input_count;i<model->node_count;i++){
		if (fwrite(&(node->bias),sizeof(float),1,file)!=1){
			goto _error;
		}
		node++;
	}
	const neat_model_edge_t* edge=model->edges;
	for (unsigned int i=0;i<model->node_count*model->node_count;i++){
		if (edge->weight!=0.0f){
			neat_model_file_edge_t edge_data={
				i,
				edge->weight
			};
			if (fwrite(&edge_data,sizeof(neat_model_file_edge_t),1,file)!=1){
				goto _error;
			}
		}
		edge++;
	}
_error:
	fclose(file);
}
