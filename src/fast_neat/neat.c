#include <immintrin.h>
#include <math.h>
#include <neat.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>



#define MAX_NODE_COUNT 256
#define MAX_ALLOWED_NODE_COUNT 256

#define ENABLE_ACTIVATION_FUNCTION_SCALE 0
#define ACTIVATION_FUNCTION_SCALE 1.0f
#define ACTIVATION_FUNCTION_TYPE_TANH 0
#define ACTIVATION_FUNCTION_TYPE_STEP 1
#define ACTIVATION_FUNCTION_TYPE_LINEAR 2
#define ACTIVATION_FUNCTION_TYPE_RELU 3
#define ACTIVATION_FUNCTION_MAX_TYPE ACTIVATION_FUNCTION_TYPE_RELU

#define MUTATION_ACTION_TYPE_ADD_NODES 1
#define MUTATION_ACTION_TYPE_ADJUST_EDGE 440
#define MUTATION_ACTION_TYPE_SET_EDGE 100
#define MUTATION_ACTION_TYPE_ADJUST_NODE 333
#define MUTATION_ACTION_TYPE_SET_NODE 104
#define MUTATION_ACTION_TYPE_SET_ACTIVATION_FUNCTION 25
#define MUTATION_ACTION_TYPE_TOGGLE_NODE 20
#define MUTATION_ACTION_MASK 0x3ff
#define MAX_STALE_FITNESS_DIFFERENCE 1e-6f

#define STALENESS_BEST_SCORE_DIFFERENCE 0.001f

#if MAX_NODE_COUNT&31
#error MAX_NODE_COUNT must be a multiple of 32
#endif

#if MAX_ALLOWED_NODE_COUNT>MAX_NODE_COUNT
#error MAX_ALLOWED_NODE_COUNT must be less than or equal to MAX_NODE_COUNT
#endif

#if MUTATION_ACTION_TYPE_ADD_NODES+MUTATION_ACTION_TYPE_ADJUST_EDGE+MUTATION_ACTION_TYPE_SET_EDGE+MUTATION_ACTION_TYPE_ADJUST_NODE+MUTATION_ACTION_TYPE_SET_NODE+MUTATION_ACTION_TYPE_SET_ACTIVATION_FUNCTION+MUTATION_ACTION_TYPE_TOGGLE_NODE!=MUTATION_ACTION_MASK
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



typedef struct __attribute__((packed)) _NEAT_MODEL_FILE_NODE{
	float bias;
	uint8_t activation_function;
	uint8_t enabled;
} neat_model_file_node_t;



typedef union _FLOAT_DATA{
	float f;
	unsigned int v;
} float_data_t;



static void _random_regenerate_bits(neat_t* neat){
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



static inline void _random_ensure_count(neat_t* neat,unsigned int count){
	if (neat->_prng_state.count<count){
		_random_regenerate_bits(neat);
	}
}



static inline unsigned int _random_uint32(neat_t* neat){
	neat->_prng_state.count--;
	return neat->_prng_state.data[neat->_prng_state.count];
}



static inline const __m256i* _random_uint256_ptr(neat_t* neat){
	neat->_prng_state.count-=8;
	return (const __m256i*)(neat->_prng_state.data+neat->_prng_state.count);
}



static inline float _random_float(neat_t* neat){
	neat->_prng_state.count--;
	return ((float)(((int32_t)(neat->_prng_state.data[neat->_prng_state.count]))>>7)*0x1p-24f)*0.5f;
}



static inline unsigned int _random_uint(neat_t* neat,unsigned int max){
	unsigned int mask=max-1;
	mask|=mask>>1;
	mask|=mask>>2;
	mask|=mask>>4;
	mask|=mask>>8;
	neat->_prng_state.count--;
	unsigned int out=neat->_prng_state.data[neat->_prng_state.count]&(mask|(mask>>16));
	if (out>=max){
		out-=max;
	}
	return out;
}



static inline __m128 _vector_sum(__m256 sum256){
	__m128 sum128=_mm_add_ps(_mm256_castps256_ps128(sum256),_mm256_extractf128_ps(sum256,1));
	__m128 sum64=_mm_add_ps(sum128,_mm_movehl_ps(sum128,sum128));
	return _mm_add_ss(sum64,_mm_shuffle_ps(sum64,sum64,0b0001));
}



static inline __m128 _activation_function_tanh(__m128 x){
	__m128 x_sq=_mm_mul_ps(x,x);
	__m128 mask=_mm_castsi128_ps(_mm_set1_epi32(0x80000000));
	__m128 sign_mask=_mm_andnot_ps(x,mask);
	x=_mm_xor_ps(x,_mm_xor_ps(sign_mask,mask));
	x=_mm_fmadd_ps(
		_mm_fmadd_ps(
			_mm_set_ps1(10.0f),
			x,
			_mm_fmadd_ps(
				x,
				x_sq,
				_mm_set_ps1(5.0f)
			)
		),
		_mm_mul_ps(
			x_sq,
			_mm_set_ps1(0.13333333f)
		),
		x
	);
	__m128 x2=_mm_add_ps(x,_mm_set_ps1(1.0f));
	__m128 tmp=_mm_castsi128_ps(_mm_sub_epi32(_mm_set1_epi32(0x7ef127ea),_mm_castps_si128(x2)));
	return _mm_xor_ps(
		_mm_mul_ps(
			_mm_mul_ps(
				_mm_fmsub_ps(
					tmp,
					x2,
					_mm_set_ps1(2.0f)
				),
				tmp
			),
			x
		),
		sign_mask
	);
}



static inline __m128 _activation_function_step(__m128 x){
	return _mm_add_ps(
		_mm_set_ps1(1.0f),
		_mm_castsi128_ps(_mm_and_si128(
			_mm_castps_si128(_mm_set_ps1(-2.0f)),
			_mm_srai_epi32(
				_mm_castps_si128(x),
				31
			)
		))
	);
}



static inline __m128 _activation_function_linear(__m128 x){
	return x;
}



static inline __m128 _activation_function_relu(__m128 x){
	return _mm_and_ps(
		x,
		_mm_castsi128_ps(_mm_sub_epi32(
			_mm_srli_epi32(
				_mm_castps_si128(x),
				31
			),
			_mm_set1_epi32(1)
		))
	);
}



static inline void _evaluate_weights_and_biases(unsigned int input_count,unsigned int output_count,unsigned int node_count,const float* weights,const neat_genome_node_t* nodes,const float* in1,const float* in2,float* out1,float* out2){
	__attribute__((aligned(256))) float node_values[MAX_NODE_COUNT<<1];
	float* values=node_values;
	__m256 zero=_mm256_setzero_ps();
	for (unsigned int i=0;i<((node_count+31)&0xffffffe0);i+=8){
		if (i<input_count){
			_mm256_store_ps(values,_mm256_loadu_ps(in1));
			_mm256_store_ps(values+8,_mm256_loadu_ps(in2));
			in1+=8;
			in2+=8;
		}
		else{
			_mm256_store_ps(values,zero);
			_mm256_store_ps(values+8,zero);
		}
		values+=16;
	}
	weights+=input_count*node_count;
	for (unsigned int i=input_count;i<node_count;i++){
		if (!(nodes+i)->enabled&&i<node_count-output_count){
			weights+=node_count;
			continue;
		}
		values=node_values;
		__m256 weight_vector_a=_mm256_load_ps(weights);
		__m256 weight_vector_b=_mm256_load_ps(weights+8);
		__m256 weight_vector_c=_mm256_load_ps(weights+16);
		__m256 weight_vector_d=_mm256_load_ps(weights+24);
		__m256 sum256a=_mm256_mul_ps(weight_vector_a,_mm256_load_ps(values));
		__m256 sum256b=_mm256_mul_ps(weight_vector_a,_mm256_load_ps(values+8));
		__m256 sum256c=_mm256_mul_ps(weight_vector_b,_mm256_load_ps(values+16));
		__m256 sum256d=_mm256_mul_ps(weight_vector_b,_mm256_load_ps(values+24));
		__m256 sum256e=_mm256_mul_ps(weight_vector_c,_mm256_load_ps(values+32));
		__m256 sum256f=_mm256_mul_ps(weight_vector_c,_mm256_load_ps(values+40));
		__m256 sum256g=_mm256_mul_ps(weight_vector_d,_mm256_load_ps(values+48));
		__m256 sum256h=_mm256_mul_ps(weight_vector_d,_mm256_load_ps(values+56));
		weights+=32;
		unsigned int j=32;
		while (j<i){
			values+=64;
			weight_vector_a=_mm256_load_ps(weights);
			weight_vector_b=_mm256_load_ps(weights+8);
			weight_vector_c=_mm256_load_ps(weights+16);
			weight_vector_d=_mm256_load_ps(weights+24);
			sum256a=_mm256_fmadd_ps(weight_vector_a,_mm256_load_ps(values),sum256a);
			sum256b=_mm256_fmadd_ps(weight_vector_a,_mm256_load_ps(values+8),sum256b);
			sum256c=_mm256_fmadd_ps(weight_vector_b,_mm256_load_ps(values+16),sum256c);
			sum256d=_mm256_fmadd_ps(weight_vector_b,_mm256_load_ps(values+24),sum256d);
			sum256e=_mm256_fmadd_ps(weight_vector_c,_mm256_load_ps(values+32),sum256e);
			sum256f=_mm256_fmadd_ps(weight_vector_c,_mm256_load_ps(values+40),sum256f);
			sum256g=_mm256_fmadd_ps(weight_vector_d,_mm256_load_ps(values+48),sum256g);
			sum256h=_mm256_fmadd_ps(weight_vector_d,_mm256_load_ps(values+56),sum256h);
			j+=32;
			weights+=32;
		}
		weights+=((uint64_t)node_count)-j;
		__m128 value_a=_vector_sum(_mm256_add_ps(_mm256_add_ps(sum256a,sum256c),_mm256_add_ps(sum256e,sum256g)));
		__m128 value_b=_vector_sum(_mm256_add_ps(_mm256_add_ps(sum256b,sum256d),_mm256_add_ps(sum256f,sum256h)));
#if ENABLE_ACTIVATION_FUNCTION_SCALE
		__m128 combined_value=_mm_fmadd_ps(_mm_blend_ps(value_a,_mm_shuffle_ps(value_b,value_b,0b0000),0b10),_mm_set_ps1(ACTIVATION_FUNCTION_SCALE),_mm_set_ps1((nodes+i)->bias*ACTIVATION_FUNCTION_SCALE));
#else
		__m128 combined_value=_mm_add_ps(_mm_blend_ps(value_a,_mm_shuffle_ps(value_b,value_b,0b0000),0b10),_mm_set_ps1((nodes+i)->bias));
#endif
		switch ((nodes+i)->activation_function){
			case ACTIVATION_FUNCTION_TYPE_TANH:
				combined_value=_activation_function_tanh(combined_value);
				break;
			case ACTIVATION_FUNCTION_TYPE_STEP:
				combined_value=_activation_function_step(combined_value);
				break;
			case ACTIVATION_FUNCTION_TYPE_LINEAR:
				combined_value=_activation_function_linear(combined_value);
				break;
			case ACTIVATION_FUNCTION_TYPE_RELU:
				combined_value=_activation_function_relu(combined_value);
				break;
		}
		float processed_value_a=_mm_cvtss_f32(combined_value);
		float processed_value_b=_mm_cvtss_f32(_mm_shuffle_ps(combined_value,combined_value,1));
		node_values[i+(i&0xfffffff8)]=processed_value_a;
		node_values[i+(i&0xfffffff8)+8]=processed_value_b;
		if (i>=node_count-output_count){
			*out1=processed_value_a;
			out1++;
			*out2=processed_value_b;
			out2++;
		}
	}
}



void neat_init(unsigned int input_count,unsigned int output_count,unsigned int population,neat_fitness_score_callback_t fitness_score_callback,neat_t* out){
	out->input_count=input_count;
	out->output_count=output_count;
	out->population=population;
	out->_last_average_fitness_score=0.0f;
	out->fitness_score_callback=fitness_score_callback;
	for (unsigned int i=0;i<64;i++){
		out->_prng_state.data[i]=(rand()&0xff)|((rand()&0xff)<<8)|((rand()&0xff)<<16)|((rand()&0xff)<<24);
	}
	out->_prng_state.count=64;
	out->genomes=aligned_alloc(32,population*sizeof(neat_genome_t));
	out->_node_data=aligned_alloc(32,population*MAX_NODE_COUNT*sizeof(neat_genome_node_t));
	out->_edge_data=aligned_alloc(32,population*MAX_NODE_COUNT*MAX_NODE_COUNT*sizeof(neat_genome_edge_t));
	neat_reset_genomes(out);
}



void neat_deinit(neat_t* neat){
	free(neat->genomes);
	free(neat->_edge_data);
	free(neat->_node_data);
}



void neat_reset_genomes(neat_t* neat){
	unsigned int node_count=(neat->input_count+neat->output_count+7)&0xfffffff8;
	unsigned int node_count_sq=node_count*node_count;
	neat_genome_t* genome=neat->genomes;
	neat_genome_node_t* node_data_ptr=neat->_node_data;
	neat_genome_edge_t* edge_data_ptr=neat->_edge_data;
	neat->_fitness_score_sum=0.0f;
	neat->_last_best_genome_fitness=0.0f;
	for (unsigned int i=0;i<neat->population;i++){
		genome->node_count=node_count;
		genome->fitness_score=0.0f;
		genome->nodes=node_data_ptr;
		genome->edges=edge_data_ptr;
		genome->_node_count_sq=node_count_sq;
		genome->_enabled_node_count=neat->input_count+neat->output_count;
		for (unsigned int j=0;j<node_count;j++){
			node_data_ptr->bias=0.0f;
			node_data_ptr->activation_function=ACTIVATION_FUNCTION_TYPE_TANH;
			node_data_ptr->enabled=(j<neat->input_count||j>=node_count-neat->output_count);
			node_data_ptr++;
			for (unsigned int k=0;k<node_count;k++){
				_random_ensure_count(neat,1);
				edge_data_ptr->weight=_random_float(neat);
				edge_data_ptr++;
			}
		}
		node_data_ptr+=MAX_NODE_COUNT-node_count;
		edge_data_ptr+=(MAX_NODE_COUNT-node_count)*(MAX_NODE_COUNT-node_count);
		genome->fitness_score=neat->fitness_score_callback(neat,genome);
		neat->_fitness_score_sum+=genome->fitness_score;
		if (!i||neat->_last_best_genome_fitness<genome->fitness_score){
			neat->_last_best_genome_fitness=genome->fitness_score;
		}
		genome++;
	}
	neat->stale_iteration_count=0;
}



void __attribute__((flatten,hot,no_stack_protector)) neat_genome_evaluate(const neat_t* neat,const neat_genome_t* genome,const float* in1,const float* in2,float* out1,float* out2){
	_evaluate_weights_and_biases(neat->input_count,neat->output_count,genome->node_count,(const float*)(genome->edges),(const neat_genome_node_t*)(genome->nodes),in1,in2,out1,out2);
}



float neat_update(neat_t* neat){
	float average=neat->_fitness_score_sum/neat->population;
	_Bool stale=fabs(neat->_last_average_fitness_score-average)<MAX_STALE_FITNESS_DIFFERENCE;
	neat->_last_average_fitness_score=average;
	neat_genome_t* start_genome=neat->genomes;
	neat_genome_t* end_genome=neat->genomes+neat->population;
	neat_genome_t* genome=start_genome;
	float best_genome_fitness=neat->genomes->fitness_score;
	unsigned int best_genome_index=0xffffffff;
	for (unsigned int i=0;i<neat->population;i++){
		if (genome->fitness_score>best_genome_fitness){
			best_genome_fitness=genome->fitness_score;
			best_genome_index=0xffffffff;
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
			if (best_genome_index==0xffffffff){
				best_genome_index=start_genome-neat->genomes;
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
			if (best_genome_index==0xffffffff){
				best_genome_index=end_genome-neat->genomes;
			}
		}
	}
	if (stale||start_genome==neat->genomes){
		__m256i tmp=_mm256_load_si256((const __m256i*)(neat->genomes+best_genome_index));
		_mm256_store_si256((__m256i*)(neat->genomes+best_genome_index),_mm256_load_si256((const __m256i*)(neat->genomes)));
		_mm256_store_si256((__m256i*)(neat->genomes),tmp);
		start_genome=neat->genomes+1;
	}
	unsigned int surviving_genome_count=(unsigned int)(start_genome-neat->genomes);
	neat_genome_t* child=neat->genomes+surviving_genome_count;
	_random_ensure_count(neat,4);
	unsigned int first_crossover_index=(stale?neat->population:surviving_genome_count+_random_uint(neat,neat->population-surviving_genome_count));
	for (unsigned int idx=surviving_genome_count;idx<neat->population;idx++){
		_random_ensure_count(neat,4);
		const neat_genome_t* random_genome=neat->genomes+_random_uint(neat,surviving_genome_count);
		child->node_count=random_genome->node_count;
		child->_node_count_sq=random_genome->_node_count_sq;
		if (idx<first_crossover_index){
			child->_enabled_node_count=random_genome->_enabled_node_count;
			unsigned int action=_random_uint32(neat)&MUTATION_ACTION_MASK;
			if (action<=MUTATION_ACTION_TYPE_ADD_NODES&&random_genome->node_count<MAX_ALLOWED_NODE_COUNT){
				child->node_count+=8;
				child->_node_count_sq=child->node_count*child->node_count;
				unsigned int insert_index_end=random_genome->node_count-neat->output_count;
				unsigned int insert_index_start=insert_index_end-7;
				const neat_genome_node_t* random_genome_node=random_genome->nodes;
				const neat_genome_edge_t* random_genome_edge=random_genome->edges;
				neat_genome_node_t* nodes=child->nodes;
				neat_genome_edge_t* edges=child->edges;
				for (unsigned int i=0;i<child->node_count;i++){
					_Bool inserted_i=(i>=insert_index_start&&i<=insert_index_end);
					if (inserted_i){
						nodes->bias=0.0f;
						nodes->activation_function=ACTIVATION_FUNCTION_TYPE_LINEAR;
						nodes->enabled=(i==insert_index_start);
					}
					else{
						nodes->bias=random_genome_node->bias;
						nodes->activation_function=random_genome_node->activation_function;
						nodes->enabled=random_genome_node->enabled;
						random_genome_node++;
					}
					nodes++;
					for (unsigned int j=0;j<child->node_count;j++){
						if (inserted_i||(j>=insert_index_start&&j<=insert_index_end)){
							edges->weight=0.0f;
						}
						else{
							edges->weight=random_genome_edge->weight;
							random_genome_edge++;
						}
						edges++;
					}
				}
				child->_enabled_node_count++;
				goto _mutate_random_edge;
			}
			else{
				const float* edges=(const float*)(random_genome->edges);
				float* child_edges=(float*)(child->edges);
				for (unsigned int i=0;i<random_genome->_node_count_sq;i+=8){
					_mm256_store_ps(child_edges,_mm256_load_ps(edges));
					child_edges+=8;
					edges+=8;
				}
				const double* nodes=(const double*)(random_genome->nodes);
				double* child_nodes=(double*)(child->nodes);
				for (unsigned int i=0;i<random_genome->node_count;i+=4){
					_mm256_store_pd(child_nodes,_mm256_load_pd(nodes));
					child_nodes+=4;
					nodes+=4;
				}
				if (action<=MUTATION_ACTION_TYPE_ADD_NODES+MUTATION_ACTION_TYPE_ADJUST_EDGE){
_mutate_random_edge:
					(child->edges+_random_uint(neat,random_genome->_node_count_sq))->weight+=_random_float(neat);
				}
				else if (action<=MUTATION_ACTION_TYPE_ADD_NODES+MUTATION_ACTION_TYPE_ADJUST_EDGE+MUTATION_ACTION_TYPE_SET_EDGE){
					(child->edges+_random_uint(neat,random_genome->_node_count_sq))->weight=_random_float(neat);
				}
				else if (action<=MUTATION_ACTION_TYPE_ADD_NODES+MUTATION_ACTION_TYPE_ADJUST_EDGE+MUTATION_ACTION_TYPE_SET_EDGE+MUTATION_ACTION_TYPE_ADJUST_NODE){
					(child->nodes+_random_uint(neat,random_genome->node_count))->bias+=_random_float(neat);
				}
				else if (action<=MUTATION_ACTION_TYPE_ADD_NODES+MUTATION_ACTION_TYPE_ADJUST_EDGE+MUTATION_ACTION_TYPE_SET_EDGE+MUTATION_ACTION_TYPE_ADJUST_NODE+MUTATION_ACTION_TYPE_SET_NODE){
					(child->nodes+_random_uint(neat,random_genome->node_count))->bias=_random_float(neat);
				}
				else if (action<=MUTATION_ACTION_TYPE_ADD_NODES+MUTATION_ACTION_TYPE_ADJUST_EDGE+MUTATION_ACTION_TYPE_SET_EDGE+MUTATION_ACTION_TYPE_ADJUST_NODE+MUTATION_ACTION_TYPE_SET_NODE+MUTATION_ACTION_TYPE_SET_ACTIVATION_FUNCTION){
					(child->nodes+_random_uint(neat,random_genome->node_count))->activation_function=_random_uint(neat,ACTIVATION_FUNCTION_MAX_TYPE);
				}
				else{
					uint16_t* enabled=&((child->nodes+_random_uint(neat,random_genome->node_count))->enabled);
					child->_enabled_node_count+=(*enabled?-1:1);
					(*enabled)^=1;
				}
			}
		}
		else{
			const neat_genome_t* second_random_genome=neat->genomes+_random_uint(neat,surviving_genome_count);
			unsigned int min_node_count=(second_random_genome->node_count<random_genome->node_count?second_random_genome:random_genome)->node_count;
			const double* first_nodes=(const double*)(random_genome->nodes);
			const double* second_nodes=(const double*)(second_random_genome->nodes);
			double* child_nodes=(double*)(child->nodes);
			__m256i random_vector=_mm256_undefined_si256();
			for (unsigned int i=0;i<min_node_count;i+=4){
				if (!(i&255)){
					_random_ensure_count(neat,8);
					random_vector=_mm256_loadu_si256(_random_uint256_ptr(neat));
				}
				_mm256_store_pd(child_nodes,_mm256_blendv_pd(_mm256_load_pd(first_nodes),_mm256_load_pd(second_nodes),_mm256_castsi256_pd(random_vector)));
				child_nodes+=4;
				first_nodes+=4;
				second_nodes+=4;
				random_vector=_mm256_slli_epi64(random_vector,1);
			}
			for (unsigned int i=min_node_count;i<random_genome->node_count;i+=4){
				_mm256_store_pd(child_nodes,_mm256_load_pd(first_nodes));
				child_nodes+=4;
				first_nodes+=4;
			}
			child->_enabled_node_count=0;
			const float* first_edges=(const float*)(random_genome->edges);
			float* child_edges=(float*)(child->edges);
			for (unsigned int i=0;i<min_node_count;i++){
				const float* second_edges=(const float*)(second_random_genome->edges+i*second_random_genome->node_count);
				for (unsigned int j=0;j<min_node_count;j+=8){
					if (!(j&255)){
						_random_ensure_count(neat,8);
						random_vector=_mm256_loadu_si256(_random_uint256_ptr(neat));
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
				child->_enabled_node_count+=(child->nodes+i)->enabled;
			}
			for (unsigned int i=min_node_count;i<random_genome->node_count;i++){
				for (unsigned int j=0;j<random_genome->node_count;j+=8){
					_mm256_store_ps(child_edges,_mm256_load_ps(first_edges));
					child_edges+=8;
					first_edges+=8;
				}
				child->_enabled_node_count+=(child->nodes+i)->enabled;
			}
		}
		neat->_fitness_score_sum-=child->fitness_score;
		child->fitness_score=neat->fitness_score_callback(neat,child);
		neat->_fitness_score_sum+=child->fitness_score;
		if (child->fitness_score>best_genome_fitness){
			best_genome_fitness=child->fitness_score;
		}
		child++;
	}
	float best_genome_fitness_diff=neat->_last_best_genome_fitness-best_genome_fitness;
	if (fabs(best_genome_fitness_diff)<STALENESS_BEST_SCORE_DIFFERENCE){
		neat->stale_iteration_count++;
	}
	else{
		neat->stale_iteration_count=0;
		neat->_last_best_genome_fitness=best_genome_fitness;
	}
	for (unsigned int i=0;i<neat->population;i++){
		if ((neat->genomes+i)->fitness_score>best_genome_fitness){
			best_genome_fitness=(neat->genomes+i)->fitness_score;
		}
	}
	return best_genome_fitness;
}



const neat_genome_t* neat_get_best(const neat_t* neat){
	const neat_genome_t* out=neat->genomes;
	const neat_genome_t* genome=out+1;
	for (unsigned int i=1;i<neat->population;i++){
		if (genome->fitness_score>out->fitness_score){
			out=genome;
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
	out->edges=malloc(genome->_node_count_sq*sizeof(neat_model_edge_t));
	unsigned int offset=genome->node_count*neat->input_count;
	const neat_genome_edge_t* genome_edge=genome->edges+offset;
	neat_model_edge_t* edge=out->edges+offset;
	for (unsigned int i=0;i<out->node_count;i++){
		(out->nodes+i)->bias=(genome->nodes+i)->bias;
		(out->nodes+i)->activation_function=(genome->nodes+i)->activation_function;
		(out->nodes+i)->enabled=(genome->nodes+i)->enabled;
		if (i<neat->input_count){
			continue;
		}
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
		model->edge_count
	};
	if (fwrite(&header,sizeof(header),1,file)!=1){
		goto _error;
	}
	const neat_model_node_t* node=model->nodes+model->input_count;
	for (unsigned int i=model->input_count;i<model->node_count;i++){
		neat_model_file_node_t node_data={
			node->bias,
			node->activation_function,
			node->enabled
		};
		if (fwrite(&node_data,sizeof(neat_model_file_node_t),1,file)!=1){
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



_Bool neat_load_model(const char* file_path,neat_model_t* out){
	FILE* file=fopen(file_path,"rb");
	if (!file){
		return 0;
	}
	out->nodes=NULL;
	out->edges=NULL;
	neat_model_file_header_t header;
	if (fread(&header,sizeof(header),1,file)!=1){
		goto _error;
	}
	out->input_count=header.input_count;
	out->output_count=header.output_count;
	out->node_count=header.node_count;
	out->edge_count=header.edge_count;
	out->nodes=malloc(out->node_count*sizeof(neat_model_node_t));
	out->edges=malloc(out->node_count*out->node_count*sizeof(neat_model_edge_t));
	neat_model_node_t* node=out->nodes;
	neat_model_edge_t* edge=out->edges;
	for (unsigned int i=0;i<out->node_count;i++){
		for (unsigned int j=0;j<out->node_count;j++){
			edge->weight=0.0f;
			edge++;
		}
		node->bias=0.0f;
		node->activation_function=ACTIVATION_FUNCTION_TYPE_TANH;
		node->enabled=1;
		node++;
	}
	node=out->nodes+out->input_count;
	for (unsigned int i=out->input_count;i<out->node_count;i++){
		neat_model_file_node_t node_data;
		if (fread(&node_data,sizeof(neat_model_file_node_t),1,file)!=1){
			goto _error;
		}
		node->bias=node_data.bias;
		node->activation_function=node_data.activation_function;
		node->enabled=node_data.enabled;
		node++;
	}
	for (unsigned int i=0;i<out->edge_count;i++){
		neat_model_file_edge_t edge_data;
		if (fread(&edge_data,sizeof(neat_model_file_edge_t),1,file)!=1){
			goto _error;
		}
		(out->edges+edge_data.index)->weight=edge_data.weight;
	}
	fclose(file);
	return 1;
_error:
	free(out->nodes);
	free(out->edges);
	out->nodes=NULL;
	out->edges=NULL;
	fclose(file);
	return 0;
}



void __attribute__((flatten,hot,no_stack_protector)) neat_model_evaluate(const neat_model_t* model,const float* in1,const float* in2,float* out1,float* out2){
	_evaluate_weights_and_biases(model->input_count,model->output_count,model->node_count,(const float*)(model->edges),(const neat_genome_node_t*)(model->nodes),in1,in2,out1,out2);
}



float neat_random_float(neat_t* neat){
	if (!neat->_prng_state.count){
		_random_regenerate_bits(neat);
	}
	neat->_prng_state.count--;
	return ((float)(((int32_t)(neat->_prng_state.data[neat->_prng_state.count]))>>7)*0x1p-24f)*0.03125f;
}
