#include <immintrin.h>
#include <math.h>
#include <neat.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>



#define MAX_NODE_COUNT 512

#define ACTIVATION_FUNCTION_SCALE 4.5f
#define USE_STEP_ACTIVATION_FUNCTION 0

#define MUTATION_ACTION_TYPE_ADD_NODES 1
#define MUTATION_ACTION_TYPE_ADJUST_EDGE 454
#define MUTATION_ACTION_TYPE_SET_EDGE 114
#define MUTATION_ACTION_TYPE_ADJUST_NODE 340
#define MUTATION_ACTION_TYPE_SET_NODE 114
#define MUTATION_ACTION_MASK 0x3ff
#define MAX_STALE_FITNESS_DIFFERENCE 1e-6f

#if MUTATION_ACTION_TYPE_ADD_NODES+MUTATION_ACTION_TYPE_ADJUST_EDGE+MUTATION_ACTION_TYPE_SET_EDGE+MUTATION_ACTION_TYPE_ADJUST_NODE+MUTATION_ACTION_TYPE_SET_NODE!=MUTATION_ACTION_MASK
#error Sum of mutation actions must be equal to MUTATION_ACTION_MASK
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



typedef union _FLOAT_DATA{
	float f;
	unsigned int v;
} float_data_t;



static void _random_ensure_count(neat_t* neat,unsigned int count){
	if (neat->_prng_state.count>=count){
		return;
	}
	__m256i* ptr=(__m256i*)(neat->_prng_state.data);
	__m256i permute_a=_mm256_set_epi32(4,3,2,1,0,7,6,5);
	__m256i permute_b=_mm256_set_epi32(2,1,0,7,6,5,4,3);
	__m256i s0=_mm256_lddqu_si256(ptr);
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
	_mm256_storeu_si256(ptr,s0);
	_mm256_storeu_si256(ptr+1,s1);
	_mm256_storeu_si256(ptr+2,s2);
	_mm256_storeu_si256(ptr+3,s3);
	_mm256_storeu_si256(ptr+4,_mm256_xor_si256(u0,t1));
	_mm256_storeu_si256(ptr+5,_mm256_xor_si256(u2,t3));
	_mm256_storeu_si256(ptr+6,_mm256_xor_si256(s0,s3));
	_mm256_storeu_si256(ptr+7,_mm256_xor_si256(s2,s1));
	neat->_prng_state.count=64;
}



static inline unsigned int _random_uint32(neat_t* neat){
	_random_ensure_count(neat,1);
	neat->_prng_state.count--;
	return neat->_prng_state.data[neat->_prng_state.count];
}



static inline const unsigned int* _random_uint256_ptr(neat_t* neat){
	_random_ensure_count(neat,8);
	neat->_prng_state.count-=8;
	return neat->_prng_state.data+neat->_prng_state.count;
}



static inline float _random_uniform(neat_t* neat){
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



static inline float _activation_function(float x){
	float_data_t data={
		.f=x*ACTIVATION_FUNCTION_SCALE
	};
#if USE_STEP_ACTIVATION_FUNCTION
	return 1-2.0f*(data.v>>31);
#else
	float x_sq=x*x;
	unsigned int sign_mask=data.v&0x80000000;
	data.v&=0x7fffffff;
	x=data.f;
	x+=0.13333333f*x_sq*(5+10*x+x*x_sq);
	float x2=x+1;
	data.f=x2;
	data.v=0x7ef127ea-data.v;
	float tmp=2-x2*data.f;
	data.v|=sign_mask;
	return data.f*tmp*x;
#endif
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
	out->genomes=aligned_alloc(32,population*sizeof(neat_genome_t));
	out->_node_data=aligned_alloc(32,population*MAX_NODE_COUNT*sizeof(neat_genome_node_t));
	out->_edge_data=aligned_alloc(32,population*MAX_NODE_COUNT*MAX_NODE_COUNT*sizeof(neat_genome_edge_t));
	unsigned int node_count=(input_count+output_count+7)&0xfffffff8;
	neat_genome_t* genome=out->genomes;
	neat_genome_node_t* node_data_ptr=out->_node_data;
	neat_genome_edge_t* edge_data_ptr=out->_edge_data;
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



void __attribute__((flatten,hot,no_stack_protector)) neat_genome_evaluate(const neat_t* neat,const neat_genome_t* genome,const float* in,float* out){
	__attribute__((aligned(256))) float node_values[MAX_NODE_COUNT];
	float* values=node_values;
	__m256 zero=_mm256_setzero_ps();
	for (unsigned int i=0;i<((genome->node_count+31)&0xffffffe0);i+=8){
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
		switch (neat->input_count&7){
			case 1:
				node_values[neat->input_count+6]=0.0f;
			case 2:
				node_values[neat->input_count+5]=0.0f;
			case 3:
				node_values[neat->input_count+4]=0.0f;
			case 4:
				node_values[neat->input_count+3]=0.0f;
			case 5:
				node_values[neat->input_count+2]=0.0f;
			case 6:
				node_values[neat->input_count+1]=0.0f;
			case 7:
				node_values[neat->input_count]=0.0f;
		}
	}
	const float* weights=(const float*)(genome->edges+neat->input_count*genome->node_count);
	for (unsigned int i=neat->input_count;i<genome->node_count;i++){
		values=node_values;
		__m256 sum256a=_mm256_mul_ps(_mm256_load_ps(weights),_mm256_load_ps(values));
		__m256 sum256b=_mm256_mul_ps(_mm256_load_ps(weights+8),_mm256_load_ps(values+8));
		__m256 sum256c=_mm256_mul_ps(_mm256_load_ps(weights+16),_mm256_load_ps(values+16));
		__m256 sum256d=_mm256_mul_ps(_mm256_load_ps(weights+24),_mm256_load_ps(values+24));
		weights+=32;
		unsigned int j=32;
		while (j<i){
			values+=32;
			sum256a=_mm256_fmadd_ps(_mm256_load_ps(weights),_mm256_load_ps(values),sum256a);
			sum256b=_mm256_fmadd_ps(_mm256_load_ps(weights+8),_mm256_load_ps(values+8),sum256b);
			sum256c=_mm256_fmadd_ps(_mm256_load_ps(weights+16),_mm256_load_ps(values+16),sum256c);
			sum256d=_mm256_fmadd_ps(_mm256_load_ps(weights+24),_mm256_load_ps(values+24),sum256d);
			j+=32;
			weights+=32;
		}
		weights+=((uint64_t)genome->node_count)-j;
		node_values[i]=_activation_function(_vector_sum(_mm256_add_ps(_mm256_add_ps(sum256a,sum256b),_mm256_add_ps(sum256c,sum256d)))+(genome->nodes+i)->bias);
		if (i>=genome->node_count-neat->output_count){
			*out=node_values[i];
			out++;
		}
	}
}



float neat_update(neat_t* neat){
	float average=neat->_fitness_score_sum/neat->population;
	_Bool stale=fabs(neat->_last_average_fitness_score-average)<MAX_STALE_FITNESS_DIFFERENCE;
	neat->_last_average_fitness_score=average;
	neat_genome_t* start_genome=neat->genomes;
	neat_genome_t* end_genome=neat->genomes+neat->population;
	neat_genome_t* genome=start_genome;
	float best_genome_fitness=neat->genomes->fitness_score;
	for (unsigned int i=0;i<neat->population;i++){
		if (genome->fitness_score>best_genome_fitness){
			best_genome_fitness=genome->fitness_score;
		}
		if (genome->fitness_score>=average){
			if (start_genome==genome){
				genome++;
			}
			else{
				__m256i tmp=_mm256_load_si256((const __m256i*)genome);
				_mm256_store_si256((__m256i*)genome,_mm256_load_si256((const __m256i*)start_genome));
				_mm256_store_si256((__m256i*)start_genome,tmp);
			}
			start_genome++;
		}
		else{
			end_genome--;
			if (end_genome==genome){
				genome--;
			}
			else{
				__m256i tmp=_mm256_load_si256((const __m256i*)genome);
				_mm256_store_si256((__m256i*)genome,_mm256_load_si256((const __m256i*)end_genome));
				_mm256_store_si256((__m256i*)end_genome,tmp);
			}
		}
	}
	if (stale||start_genome==neat->genomes){
		start_genome=neat->genomes+1;
	}
	unsigned int surviving_genome_count=(unsigned int)(start_genome-neat->genomes);
	unsigned int surviving_genome_mask=_get_number_mask(surviving_genome_count);
	neat_genome_t* child=neat->genomes+surviving_genome_count;
	unsigned int mutation_type=_random_uint32(neat);
	for (unsigned int idx=surviving_genome_count;idx<neat->population;idx++){
		const neat_genome_t* random_genome=neat->genomes+_random_int_mask(neat,surviving_genome_mask,surviving_genome_count);
		child->node_count=random_genome->node_count;
		if (stale||(mutation_type&1)){
			unsigned int action=_random_uint32(neat)&MUTATION_ACTION_MASK;
			if (action<=MUTATION_ACTION_TYPE_ADD_NODES&&random_genome->node_count<MAX_NODE_COUNT){
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
				const float* edges=(const float*)(random_genome->edges);
				float* child_edges=(float*)(child->edges);
				for (unsigned int i=0;i<random_genome->node_count*random_genome->node_count;i+=8){
					_mm256_store_ps(child_edges,_mm256_load_ps(edges));
					child_edges+=8;
					edges+=8;
				}
				if (action<=MUTATION_ACTION_TYPE_ADD_NODES+MUTATION_ACTION_TYPE_ADJUST_EDGE){
_mutate_random_edge:
					(child->edges+_random_int(neat,random_genome->node_count*random_genome->node_count))->weight+=_random_uniform(neat);
				}
				else if (action<=MUTATION_ACTION_TYPE_ADD_NODES+MUTATION_ACTION_TYPE_ADJUST_EDGE+MUTATION_ACTION_TYPE_SET_EDGE){
					(child->edges+_random_int(neat,random_genome->node_count*random_genome->node_count))->weight=_random_uniform(neat);
				}
				else if (action<=MUTATION_ACTION_TYPE_ADD_NODES+MUTATION_ACTION_TYPE_ADJUST_EDGE+MUTATION_ACTION_TYPE_SET_EDGE+MUTATION_ACTION_TYPE_ADJUST_NODE){
					(child->nodes+_random_int(neat,random_genome->node_count))->bias+=_random_uniform(neat);
				}
				else{
					(child->nodes+_random_int(neat,random_genome->node_count))->bias=_random_uniform(neat);
				}
			}
		}
		else{
			const neat_genome_t* second_random_genome=neat->genomes+_random_int_mask(neat,surviving_genome_mask,surviving_genome_count);
			unsigned int min_node_count=(second_random_genome->node_count<random_genome->node_count?second_random_genome:random_genome)->node_count;
			const float* first_edges=(const float*)(random_genome->edges);
			float* child_edges=(float*)(child->edges);
			for (unsigned int i=0;i<min_node_count;i++){
				const float* second_edges=(const float*)(second_random_genome->edges+i*second_random_genome->node_count);
				__m256i random_vector=_mm256_undefined_si256();
				for (unsigned int j=0;j<min_node_count;j+=8){
					if (!(j&63)){
						random_vector=_mm256_loadu_si256((const __m256i*)_random_uint256_ptr(neat));
					}
					_mm256_store_ps(child_edges,_mm256_blendv_ps(_mm256_load_ps(first_edges),_mm256_load_ps(second_edges),_mm256_castsi256_ps(random_vector)));
					child_edges+=8;
					first_edges+=8;
					second_edges+=8;
					random_vector=_mm256_slli_epi32(random_vector,1);
				}
				for (unsigned int j=min_node_count;j<random_genome->node_count;j+=8){
					_mm256_store_ps(child_edges,_mm256_load_ps(first_edges));
					child_edges+=8;
					first_edges+=8;
				}
			}
			for (unsigned int i=min_node_count;i<random_genome->node_count;i++){
				for (unsigned int j=0;j<random_genome->node_count;j+=8){
					_mm256_store_ps(child_edges,_mm256_load_ps(first_edges));
					child_edges+=8;
					first_edges+=8;
				}
				(child->nodes+i)->bias=(random_genome->nodes+i)->bias;
			}
			const float* first_nodes=(const float*)(random_genome->nodes);
			const float* second_nodes=(const float*)(second_random_genome->nodes);
			float* child_nodes=(float*)(child->nodes);
			__m256i random_vector=_mm256_undefined_si256();
			for (unsigned int i=0;i<min_node_count;i+=8){
				if (!(i&63)){
					random_vector=_mm256_loadu_si256((const __m256i*)_random_uint256_ptr(neat));
				}
				_mm256_store_ps(child_nodes,_mm256_blendv_ps(_mm256_load_ps(first_nodes),_mm256_load_ps(second_nodes),_mm256_castsi256_ps(random_vector)));
				child_nodes+=8;
				first_nodes+=8;
				second_nodes+=8;
				random_vector=_mm256_slli_epi32(random_vector,1);
			}
			for (unsigned int i=min_node_count;i<random_genome->node_count;i+=8){
				_mm256_store_ps(child_nodes,_mm256_load_ps(first_nodes));
				child_nodes+=8;
				first_nodes+=8;
			}
		}
		mutation_type=(idx&31?mutation_type>>1:_random_uint32(neat));
		neat->_fitness_score_sum-=child->fitness_score;
		child->fitness_score=neat->fitness_score_callback(neat,child);
		neat->_fitness_score_sum+=child->fitness_score;
		if (child->fitness_score>best_genome_fitness){
			best_genome_fitness=child->fitness_score;
		}
		child++;
	}
	return best_genome_fitness;
}



const neat_genome_t* neat_get_best(const neat_t* neat){
	const neat_genome_t* out=neat->genomes;
	const neat_genome_t* genome=out+1;
	for (unsigned int i=1;i<neat->population;i++){
		if (genome->fitness_score>out->fitness_score){
			genome=out;
		}
		genome++;
	}
	return out;
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
	unsigned int offset=model->node_count*model->input_count;
	neat_model_file_header_t header={
		model->input_count,
		model->output_count,
		model->node_count,
		model->edge_count-offset
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
	const neat_model_edge_t* edge=model->edges+offset;
	for (unsigned int i=offset;i<model->node_count*model->node_count;i++){
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
