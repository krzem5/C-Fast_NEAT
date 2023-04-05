#include <example.h>
#include <math.h>
#include <neat.h>
#include <stdio.h>



#define SIMULATION_COUNT 1000
#define MAX_SIMULATION_STEPS 1000
#define FORCE 10.0f
#define GRAVITY 9.81f
#define CART_MASS 1.0f
#define POLE_MASS 0.1f
#define POLE_LENGTH 0.5f
#define TAU 0.08f
#define MAX_X_VALUE 2.4f
#define MAX_ANGLE_VALUE 0.3f
#define MIN_START_VALUE 0.1f
#define MAX_START_VALUE 0.2f



typedef struct _STATE{
	union{
		struct{
			float x;
			float velocity;
			float angle;
			float angular_velocity;
			float zero[4];
		};
		float raw[8];
	};
} state_t;




static void _init_state(state_t* out){
	for (unsigned int i=0;i<4;i++){
		out->raw[i]=example_random_uniform(MIN_START_VALUE-MAX_START_VALUE,MAX_START_VALUE-MIN_START_VALUE);
		out->raw[i]+=MIN_START_VALUE*(out->raw[i]<0?-1:1);
		out->raw[i+4]=0.0f;
	}
}



static void _update_state(state_t* state,float force){
	float angle_cos=cosf(state->angle);
	float angle_sin=sinf(state->angle);
	float tmp=(force+state->angular_velocity*state->angular_velocity*angle_sin*(POLE_MASS+POLE_LENGTH))/(POLE_MASS+CART_MASS);
	float angular_acceletation=(angle_sin*GRAVITY-angle_cos*tmp)/(POLE_LENGTH*(4/3.0f-angle_cos*angle_cos*POLE_MASS/(POLE_MASS+CART_MASS)));
	float acceleration=tmp-angular_acceletation*angle_cos*(POLE_MASS+POLE_LENGTH)/(POLE_MASS+CART_MASS);
	state->x+=TAU*state->velocity;
	state->velocity+=TAU*acceleration;
	state->angle+=TAU*state->angular_velocity;
	state->angular_velocity+=TAU*angular_acceletation;
}



float cartpole_fitness_score_callback(const neat_t* neat,const neat_genome_t* genome){
	unsigned int out=0;
	for (unsigned int i=0;i<SIMULATION_COUNT;i+=2){
		state_t states[2];
		_init_state(states);
		_init_state(states+1);
		unsigned int state_time[2]={0,0};
		unsigned int state_count;
		unsigned int j=0;
		do{
			j++;
			float force_directions[2];
			neat_genome_evaluate(neat,genome,states[0].raw,states[1].raw,force_directions,force_directions+1);
			state_count=2;
			for (unsigned int k=0;k<2;k++){
				if (fabs(states[k].x)<=MAX_X_VALUE&&fabs(states[k].angle)<=MAX_ANGLE_VALUE&&j<MAX_SIMULATION_STEPS){
					_update_state(states+k,FORCE*((force_directions[k]>0)*2-1));
				}
				else{
					state_count--;
					if (!state_time[k]){
						state_time[k]=j;
					}
				}
			}
		} while (state_count);
		out+=state_time[0]+state_time[1];
	}
	return ((float)out)/(SIMULATION_COUNT*MAX_SIMULATION_STEPS);
}



void cartpole_end_callback(const neat_t* neat,const neat_genome_t* genome){
	state_t state;
	_init_state(&state);
	for (unsigned int i=0;fabs(state.x)<=MAX_X_VALUE&&fabs(state.angle)<=MAX_ANGLE_VALUE&&i<MAX_SIMULATION_STEPS;i++){
		float force_direction;
		neat_genome_evaluate(neat,genome,state.raw,state.raw,&force_direction,&force_direction);
		_update_state(&state,FORCE*((force_direction>0)*2-1));
		printf("%f %f\n",state.x,state.angle);
	}
}
