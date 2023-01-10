#include <math.h>
#include <neat.h>
#include <stdio.h>
#include <string.h>



#define MAX_LETTER_COUNT 12
#define LANGUAGE_COUNT 2
#define WORD_COUNT 1000



static const char* languages[LANGUAGE_COUNT]={"en","pl"};
static char words[MAX_LETTER_COUNT*WORD_COUNT*LANGUAGE_COUNT];
static _Bool loaded=0;



static void _load_data(void){
	char read_buffer[(MAX_LETTER_COUNT+1)*WORD_COUNT];
	loaded=1;
	char* data=words;
	for (unsigned int i=0;i<LANGUAGE_COUNT;i++){
		char path[4096]="data/";
		strcat(strcat(path,languages[i]),".txt");
		FILE* file=fopen(path,"rb");
		read_buffer[fread(read_buffer,1,(MAX_LETTER_COUNT+1)*WORD_COUNT,file)]=0;
		fclose(file);
		const char* src=read_buffer;
		while (*src){
			unsigned int j=0;
			while (*src&&*src!='\n'){
				*data=*src;
				data++;
				src++;
				j++;
			}
			while (j<MAX_LETTER_COUNT){
				*data=0;
				data++;
				j++;
			}
			if (*src=='\n'){
				src++;
			}
		}
		while (data-words<(i+1)*MAX_LETTER_COUNT*WORD_COUNT){
			*data=0;
			data++;
		}
	}
}



static void _encode_word(const char* word,float* out){
	for (unsigned int i=0;i<MAX_LETTER_COUNT;i++){
		unsigned int k=*word-97;
		word++;
		for (unsigned int j=0;j<26;j++){
			*out=(j==k?1.0f:0.0f);
			out++;
		}
	}
}



static unsigned int _get_max_language(const float* genome_out){
	unsigned int out=0;
	float max=*genome_out;
	genome_out++;
	for (unsigned int i=1;i<LANGUAGE_COUNT;i++){
		if (*genome_out>max){
			max=*genome_out;
			out=i;
		}
		genome_out++;
	}
	return out;
}



float language_fitness_score_callback(const neat_t* neat,const neat_genome_t* genome){
	if (!loaded){
		_load_data();
	}
	unsigned int word_language_index=0;
	float genome_in[26*MAX_LETTER_COUNT];
	float genome_out[LANGUAGE_COUNT];
	float out=0.0f;
	for (unsigned int i=0;i<WORD_COUNT*LANGUAGE_COUNT;i++){
		_encode_word(words+i*MAX_LETTER_COUNT,genome_in);
		neat_genome_evaluate(neat,genome,genome_in,genome_out);
		word_language_index=i/WORD_COUNT;
		for (unsigned int j=0;j<LANGUAGE_COUNT;j++){
			float diff=genome_out[j]-(j==word_language_index?1.0f:0.0f);
			out+=diff*diff;
		}
	}
	return 1/(1+sqrtf(out));
}



void language_end_callback(const neat_t* neat,const neat_genome_t* genome){
	const char* word="example";
	float genome_in[26*MAX_LETTER_COUNT];
	float genome_out[LANGUAGE_COUNT];
	_encode_word(word,genome_in);
	neat_genome_evaluate(neat,genome,genome_in,genome_out);
	printf("%s -> %s (%.2f %.2f)\n",word,languages[_get_max_language(genome_out)],genome_out[0],genome_out[1]);
}
