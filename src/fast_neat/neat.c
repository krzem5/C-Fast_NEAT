#include <immintrin.h>
#include <math.h>
#include <neat.h>
#include <stdio.h>
#include <stdlib.h>



#ifdef _MSC_VER
#define AVX2_ALIGN __declspec(align(256))
#else
#define AVX2_ALIGN __attribute__((aligned(256)))
#endif



#define MAX_NODE_COUNT 512

#define NODE_ADD_CHANCE 2
#define WEIGHT_ADJUST_CHANCE 400
#define WEIGHT_SET_CHANCE 100
#define BIAS_ADJUST_CHANCE 300
#define BIAS_SET_CHANCE 100
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



static inline float _vector_sum(__m256 sum256){
	__m128 sum128=_mm_add_ps(_mm256_castps256_ps128(sum256),_mm256_extractf128_ps(sum256,1));
	__m128 sum64=_mm_add_ps(sum128,_mm_movehl_ps(sum128,sum128));
	return _mm_cvtss_f32(_mm_add_ss(sum64,_mm_shuffle_ps(sum64,sum64,0x1)));
}



void neat_init(unsigned int input_count,unsigned int output_count,unsigned int population,neat_t* out){
	out->input_count=input_count;
	out->output_count=output_count;
	out->population=population;
	out->_last_average_fitness_score=-1e8f;
	out->genomes=malloc(population*sizeof(neat_genome_t));
	out->_node_data=malloc(population*MAX_NODE_COUNT*sizeof(neat_genome_node_t));
	out->_edge_data=malloc((population*MAX_NODE_COUNT*MAX_NODE_COUNT+8)*sizeof(neat_genome_edge_t));
	unsigned int node_count=(input_count+output_count+7)&0xfffffff8;
	neat_genome_t* genome=out->genomes;
	neat_genome_node_t* node_data_ptr=out->_node_data;
	neat_genome_edge_t* edge_data_ptr=(neat_genome_edge_t*)(((uintptr_t)(out->_edge_data+7))&0xffffffffffffffe0);
	for (unsigned int i=0;i<population;i++){
		genome->node_count=node_count;
		genome->nodes=node_data_ptr;
		genome->edges=edge_data_ptr;
		node_data_ptr+=MAX_NODE_COUNT;
		edge_data_ptr+=MAX_NODE_COUNT*MAX_NODE_COUNT;
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
}



void neat_deinit(const neat_t* neat){
	free(neat->genomes);
	free(neat->_edge_data);
	free(neat->_node_data);
}



void neat_genome_evaluate(const neat_t* neat,const neat_genome_t* genome,const float* in,float* out){
	AVX2_ALIGN float node_values[MAX_NODE_COUNT];
	float* values=node_values+(neat->input_count&0xfffffff8);
	__m256 zero=_mm256_setzero_ps();
	for (unsigned int i=neat->input_count>>3;i<genome->node_count>>3;i++){
		_mm256_store_ps(values,zero);
		values+=8;
	}
	for (unsigned int i=0;i<neat->input_count;i++){
		node_values[i]=*in;
		in++;
	}
	const float* weights=(const float*)(genome->edges+neat->input_count*genome->node_count);
	for (unsigned int i=neat->input_count;i<genome->node_count;i++){
		__m256 sum256=_mm256_setzero_ps();
		values=node_values;
		for (unsigned int j=0;j<genome->node_count>>3;j++){
			sum256=_mm256_fmadd_ps(_mm256_load_ps(weights),_mm256_load_ps(values),sum256);
			weights+=8;
			values+=8;
		}
		float sum=_vector_sum(sum256)+(genome->nodes+i)->bias;
		sum*=1+sum*sum;
		node_values[i]=sum/(1+fabsf(sum));
		if (i>=genome->node_count-neat->output_count){
			*out=node_values[i];
			out++;
		}
	}
}



const neat_genome_t* neat_update(neat_t* neat,float (*fitness_score_callback)(const neat_t*,const neat_genome_t*)){
	neat_genome_t* genome=neat->genomes;
	float average=0;
	for (unsigned int i=0;i<neat->population;i++){
		genome->fitness_score=fitness_score_callback(neat,genome);
		average+=genome->fitness_score;
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
	const neat_genome_t* best_genome=neat->genomes;
	neat_genome_t* child=neat->genomes+1;
	while (child<start_genome){
		if (child->fitness_score>best_genome->fitness_score){
			best_genome=child;
		}
		child++;
	}
	for (unsigned int idx=(unsigned int)(start_genome-neat->genomes);idx<neat->population;idx++){
		const neat_genome_t* random_genome=neat->genomes+(_random_uint32()%idx);
		child->node_count=random_genome->node_count;
		if (stale||(_random_uint32()&1)||1){
			unsigned int action=_random_uint32()%(NODE_ADD_CHANCE+WEIGHT_ADJUST_CHANCE+WEIGHT_SET_CHANCE+BIAS_ADJUST_CHANCE+BIAS_SET_CHANCE);
			if (action<=NODE_ADD_CHANCE&&random_genome->node_count<MAX_NODE_COUNT){
				child->node_count+=8;
				unsigned int insert_index_end=random_genome->node_count-neat->output_count;
				unsigned int insert_index_start=insert_index_end-7;
				const neat_genome_node_t* random_genome_node=random_genome->nodes;
				const neat_genome_edge_t* random_genome_edge=random_genome->edges;
				unsigned int k=0;
				for (unsigned int i=0;i<child->node_count;i++){
					_Bool inserted_i=i>=insert_index_start&&i<=insert_index_end;
					if (inserted_i){
						(child->nodes+i)->bias=0.0f;
					}
					else{
						(child->nodes+i)->bias=random_genome_node->bias;
						random_genome_node++;
					}
					for (unsigned int j=0;j<child->node_count;j++){
						if (inserted_i||(j>=insert_index_start&&j<=insert_index_end)){
							(child->edges+k)->weight=0.0f;
						}
						else{
							(child->edges+k)->weight=random_genome_edge->weight;
							random_genome_edge++;
						}
						k++;
					}
				}
				(child->edges+(_random_uint32()%(random_genome->node_count*random_genome->node_count)))->weight=_random_uniform_rescaled();
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
