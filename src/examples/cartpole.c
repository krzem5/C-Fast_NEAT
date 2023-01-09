#include <example.h>
#include <math.h>
#include <neat.h>
#include <stdio.h>



#define SIMULATION_COUNT 100
#define MAX_SIMULATION_STEPS 1000
#define FORCE 10.0f
#define GRAVITY 9.81f
#define CART_MASS 1.0f
#define POLE_MASS 0.1f
#define POLE_LENGTH 0.5f
#define TAU 0.08f
#define MAX_X_VALUE 2.4f
#define MAX_THETA_VALUE 0.2094f
#define MAX_START_VALUE 0.1f



typedef struct _STATE{
	union{
		struct{
			float x;
			float x_dot;
			float theta;
			float theta_dot;
		};
		struct{
			float raw[4];
		};
	};
} state_t;




static void _init_state(state_t* out){
	for (unsigned int i=0;i<4;i++){
		out->raw[i]=example_random_uniform(-MAX_START_VALUE,MAX_START_VALUE);
	}
}



static void _update_state(state_t* state,float force){
	float theta_cos=cos(state->theta);
	float theta_sin=sin(state->theta);
	float tmp=(force+state->theta_dot*state->theta_dot*theta_sin*(POLE_MASS+POLE_LENGTH))/(POLE_MASS+CART_MASS);
	float theta_dot_dot=(theta_sin*GRAVITY-theta_cos*tmp)/(POLE_LENGTH*(4/3.0f-theta_cos*theta_cos*POLE_MASS/(POLE_MASS+CART_MASS)));
	float x_dot_dot=tmp-theta_dot_dot*theta_cos*(POLE_MASS+POLE_LENGTH)/(POLE_MASS+CART_MASS);
	state->x+=TAU*state->x_dot;
	state->x_dot+=TAU*x_dot_dot;
	state->theta+=TAU*state->theta_dot;
	state->theta_dot+=TAU*theta_dot_dot;
}



float cartpole_fitness_score_callback(const neat_t* neat,const neat_genome_t* genome){
	unsigned int out=0;
	for (unsigned int i=0;i<SIMULATION_COUNT;i++){
		state_t state;
		_init_state(&state);
		unsigned int j=0;
		while (fabs(state.x)<=MAX_X_VALUE&&fabs(state.theta)<=MAX_THETA_VALUE&&j<MAX_SIMULATION_STEPS){
			float force_direction;
			neat_genome_evaluate(neat,genome,state.raw,&force_direction);
			_update_state(&state,FORCE*((force_direction>0.5f)*2-1));
			j++;
		}
		out+=j;
	}
	return ((float)out)/(SIMULATION_COUNT*MAX_SIMULATION_STEPS);
}



void cartpole_end_callback(const neat_t* neat,const neat_genome_t* genome){
	state_t state;
	_init_state(&state);
	for (unsigned int i=0;fabs(state.x)<=MAX_X_VALUE&&fabs(state.theta)<=MAX_THETA_VALUE&&i<MAX_SIMULATION_STEPS;i++){
		float force_direction;
		neat_genome_evaluate(neat,genome,state.raw,&force_direction);
		_update_state(&state,FORCE*((force_direction>0.5f)*2-1));
		printf("%f %f\n",state.x,state.theta);
	}
}
