#include <immintrin.h>
#include <math.h>
#include <neat.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>



#define MAX_NODE_COUNT 512

#define NODE_ADD_CHANCE 1
#define WEIGHT_ADJUST_CHANCE 454
#define WEIGHT_SET_CHANCE 114
#define BIAS_ADJUST_CHANCE 340
#define BIAS_SET_CHANCE 114
#define MUTATION_ACTION_MASK 0x3ff
#define MAX_STALE_FITNESS_DIFFERENCE 1e-6f

#if NODE_ADD_CHANCE+WEIGHT_ADJUST_CHANCE+WEIGHT_SET_CHANCE+BIAS_ADJUST_CHANCE+BIAS_SET_CHANCE!=MUTATION_ACTION_MASK
#error Sum of mutation chances must be equal to MUTATION_ACTION_MASK
#endif



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



static unsigned int _random_uint32(neat_t* neat){
	if (!neat->_prng_state.count){
		__m256i* ptr=(__m256i*)(neat->_prng_state.data);
		__m256i permute_a=_mm256_set_epi32(4,3,2,1,0,7,6,5);
		__m256i permute_b=_mm256_set_epi32(2,1,0,7,6,5,4,3);
		__m256i s0=_mm256_lddqu_si256(ptr+0);
		__m256i s1=_mm256_lddqu_si256(ptr+1);
		__m256i s2=_mm256_lddqu_si256(ptr+2);
		__m256i s3=_mm256_lddqu_si256(ptr+3);
		__m256i u0=_mm256_srli_epi64(s0,1);
		__m256i u1=_mm256_srli_epi64(s1,3);
		__m256i u2=_mm256_srli_epi64(s2,1);
		__m256i u3=_mm256_srli_epi64(s3,3);
		__m256i t0=_mm256_permutevar8x32_epi32(s0,permute_a);
		__m256i t1=_mm256_permutevar8x32_epi32(s1,permute_b);
		__m256i t2=_mm256_permutevar8x32_epi32(s2,permute_a);
		__m256i t3=_mm256_permutevar8x32_epi32(s3,permute_b);
		s0=_mm256_add_epi64(t0,u0);
		s1=_mm256_add_epi64(t1,u1);
		s2=_mm256_add_epi64(t2,u2);
		s3=_mm256_add_epi64(t3,u3);
		_mm256_storeu_si256(ptr+0,s0);
		_mm256_storeu_si256(ptr+1,s1);
		_mm256_storeu_si256(ptr+2,s2);
		_mm256_storeu_si256(ptr+3,s3);
		_mm256_storeu_si256(ptr+4,_mm256_xor_si256(u0,t1));
		_mm256_storeu_si256(ptr+5,_mm256_xor_si256(u2,t3));
		_mm256_storeu_si256(ptr+6,_mm256_xor_si256(s0,s3));
		_mm256_storeu_si256(ptr+7,_mm256_xor_si256(s2,s1));
		neat->_prng_state.count=64;
	}
	neat->_prng_state.count--;
	return neat->_prng_state.data[neat->_prng_state.count];
}



static __attribute__((noinline)) float _random_uniform(neat_t* neat){
	return (float)(((int32_t)_random_uint32(neat))>>7)*0x1p-24f;
}



static inline float _vector_sum(__m256 sum256){
	__m128 sum128=_mm_add_ps(_mm256_castps256_ps128(sum256),_mm256_extractf128_ps(sum256,1));
	__m128 sum64=_mm_add_ps(sum128,_mm_movehl_ps(sum128,sum128));
	return _mm_cvtss_f32(_mm_add_ss(sum64,_mm_shuffle_ps(sum64,sum64,0b01)));
}



static inline unsigned int _get_number_mask(unsigned int n){
	n--;
	n|=n>>1;
	n|=n>>2;
	n|=n>>4;
	n|=n>>8;
	return n|(n>>16);
}



static inline unsigned int _random_int_mask(neat_t* neat,unsigned int mask,unsigned int max){
	unsigned int out=_random_uint32(neat)&mask;
	if (out>=max){
		out-=max;
	}
	return out;
}



static inline unsigned int _random_int(neat_t* neat,unsigned int max){
	return _random_int_mask(neat,_get_number_mask(max),max);
}



void neat_init(unsigned int input_count,unsigned int output_count,unsigned int population,neat_fitness_score_callback_t fitness_score_callback,neat_t* out){
	out->input_count=input_count;
	out->output_count=output_count;
	out->population=population;
	out->_last_average_fitness_score=-1e8f;
	out->fitness_score_callback=fitness_score_callback;
	for (unsigned int i=0;i<64;i++){
		out->_prng_state.data[i]=(rand()&0xff)|((rand()&0xff)<<8)|((rand()&0xff)<<16)|((rand()&0xff)<<24);
	}
	out->_prng_state.count=64;
	out->genomes=malloc(population*sizeof(neat_genome_t));
	out->_node_data=malloc(population*MAX_NODE_COUNT*sizeof(neat_genome_node_t));
	out->_edge_data=malloc(population*MAX_NODE_COUNT*MAX_NODE_COUNT*sizeof(neat_genome_edge_t)+31);
	unsigned int node_count=(input_count+output_count+7)&0xfffffff8;
	neat_genome_t* genome=out->genomes;
	neat_genome_node_t* node_data_ptr=out->_node_data;
	neat_genome_edge_t* edge_data_ptr=(neat_genome_edge_t*)((((uintptr_t)out->_edge_data)+31)&0xffffffffffffffe0ull);
	for (unsigned int i=0;i<population;i++){
		genome->node_count=node_count;
		genome->fitness_score=0.0f;
		genome->nodes=node_data_ptr;
		genome->edges=edge_data_ptr;
		for (unsigned int j=0;j<node_count;j++){
			node_data_ptr->bias=0.0f;
			node_data_ptr++;
			for (unsigned int k=0;k<node_count;k++){
				edge_data_ptr->weight=_random_uniform(out);
				edge_data_ptr++;
			}
		}
		node_data_ptr+=MAX_NODE_COUNT-node_count;
		edge_data_ptr+=(MAX_NODE_COUNT-node_count)*(MAX_NODE_COUNT-node_count);
		genome++;
	}
	out->_fitness_score_sum=0.0f;
}



void neat_deinit(neat_t* neat){
	free(neat->genomes);
	free(neat->_edge_data);
	free(neat->_node_data);
}



void neat_genome_evaluate(const neat_t* neat,const neat_genome_t* genome,const float* in,float* out){
	__attribute__((aligned(256))) float node_values[MAX_NODE_COUNT];
	float* values=node_values;
	__m256 zero=_mm256_setzero_ps();
	for (unsigned int i=0;i<genome->node_count;i+=8){
		if (i<neat->input_count){
			_mm256_store_ps(values,_mm256_loadu_ps(in));
			in+=8;
		}
		else{
			_mm256_store_ps(values,zero);
		}
		values+=8;
	}
	if (neat->input_count&7){
		unsigned int i=8-(neat->input_count&7);
		do{
			i--;
			node_values[i+neat->input_count]=0.0f;
		} while (i);
	}
	const float* weights=(const float*)(genome->edges+neat->input_count*genome->node_count);
	for (unsigned int i=neat->input_count;i<genome->node_count;i++){
		__m256 sum256=_mm256_setzero_ps();
		values=node_values;
		for (unsigned int j=0;j<(i+7)>>3;j++){
			sum256=_mm256_fmadd_ps(_mm256_load_ps(weights),_mm256_load_ps(values),sum256);
			weights+=8;
			values+=8;
		}
		float sum=_vector_sum(sum256)+(genome->nodes+i)->bias;
		sum+=sum*sum*sum;
		node_values[i]=sum/(1+fabsf(sum));
		if (i>=genome->node_count-neat->output_count){
			*out=node_values[i];
			out++;
		}
	}
}



const neat_genome_t* neat_update(neat_t* neat){
	float average=neat->_fitness_score_sum/neat->population;
	_Bool stale=fabs(neat->_last_average_fitness_score-average)<MAX_STALE_FITNESS_DIFFERENCE;
	neat->_last_average_fitness_score=average;
	neat_genome_t* start_genome=neat->genomes;
	neat_genome_t* end_genome=neat->genomes+neat->population;
	neat_genome_t* genome=start_genome;
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
	unsigned int surviving_genome_count=(unsigned int)(start_genome-neat->genomes);
	unsigned int surviving_genome_mask=_get_number_mask(surviving_genome_count);
	for (unsigned int idx=surviving_genome_count;idx<neat->population;idx++){
		const neat_genome_t* random_genome=neat->genomes+_random_int_mask(neat,surviving_genome_mask,surviving_genome_count);
		child->node_count=random_genome->node_count;
		if (stale||(_random_uint32(neat)&1)){
			unsigned int action=_random_uint32(neat)&MUTATION_ACTION_MASK;
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
				goto _mutate_random_edge;
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
					(child->edges+_random_int(neat,random_genome->node_count*random_genome->node_count))->weight+=_random_uniform(neat);
				}
				else if (action<=NODE_ADD_CHANCE+WEIGHT_ADJUST_CHANCE+WEIGHT_SET_CHANCE){
_mutate_random_edge:
					(child->edges+_random_int(neat,random_genome->node_count*random_genome->node_count))->weight=_random_uniform(neat);
				}
				else if (action<=NODE_ADD_CHANCE+WEIGHT_ADJUST_CHANCE+WEIGHT_SET_CHANCE+BIAS_ADJUST_CHANCE){
					(child->nodes+_random_int(neat,random_genome->node_count))->bias+=_random_uniform(neat);
				}
				else{
					(child->nodes+_random_int(neat,random_genome->node_count))->bias=_random_uniform(neat);
				}
			}
		}
		else{
			const neat_genome_t* second_random_genome=neat->genomes+_random_int_mask(neat,surviving_genome_mask,surviving_genome_count);
			unsigned int k=0;
			unsigned int random_value=_random_uint32(neat);
			for (unsigned int i=0;i<random_genome->node_count;i++){
				for (unsigned int j=0;j<random_genome->node_count;j++){
					(child->edges+k)->weight=(i<second_random_genome->node_count&&j<second_random_genome->node_count&&(random_value&1)?random_genome->edges+k:second_random_genome->edges+i*second_random_genome->node_count+j)->weight;
					k++;
					random_value>>=1;
					if (!random_value){
						random_value=_random_uint32(neat);
					}
				}
				(child->nodes+i)->bias=((i<second_random_genome->node_count&&(random_value&1)?second_random_genome:random_genome)->nodes+i)->bias;
				random_value>>=1;
				if (!random_value){
					random_value=_random_uint32(neat);
				}
			}
		}
		neat->_fitness_score_sum-=child->fitness_score;
		child->fitness_score=neat->fitness_score_callback(neat,child);
		neat->_fitness_score_sum+=child->fitness_score;
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



_Bool neat_save_model(const neat_model_t* model,const char* file_path){
	FILE* file=fopen(file_path,"wb");
	if (!file){
		return 0;
	}
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
	fclose(file);
	return 1;
_error:
	fclose(file);
	return 0;
}
