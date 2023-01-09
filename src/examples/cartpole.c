#include <example.h>
#include <math.h>
#include <neat.h>
#include <stdio.h>



#define MAX_SIMULATION_STEPS 10000
#define FORCE 10.0f
#define GRAVITY 9.81f
#define CART_MASS 1.0f
#define POLE_MASS 0.1f
#define POLE_LENGTH 0.5f
#define TAU 0.02f
#define MAX_X_VALUE 2.4f
#define MAX_THETA_VALUE 0.2094f
#define MAX_START_VALUE 0.05f



float cartpole_fitness_score_callback(const neat_t* neat,const neat_genome_t* genome){
	float x=example_random_uniform(-MAX_START_VALUE,MAX_START_VALUE);
	float x_dot=example_random_uniform(-MAX_START_VALUE,MAX_START_VALUE);
	float theta=example_random_uniform(-MAX_START_VALUE,MAX_START_VALUE);
	float theta_dot=example_random_uniform(-MAX_START_VALUE,MAX_START_VALUE);
	float genome_in[4];
	float genome_out;
	unsigned int i=0;
	while (fabs(x)<=MAX_X_VALUE&&fabs(theta)<=MAX_THETA_VALUE&&i<MAX_SIMULATION_STEPS){
		genome_in[0]=x;
		genome_in[1]=x_dot;
		genome_in[2]=theta;
		genome_in[3]=theta_dot;
		neat_genome_evaluate(neat,genome,genome_in,&genome_out);
		float force=FORCE*((genome_out>0.5f)*2-1);
		float theta_cos=cos(theta);
		float theta_sin=sin(theta);
		float tmp=(force+theta_dot*theta_dot*theta_sin*(POLE_MASS+POLE_LENGTH))/(POLE_MASS+CART_MASS);
		float theta_dot_dot=(theta_sin*GRAVITY-theta_cos*tmp)/(POLE_LENGTH*(4/3.0f-theta_cos*theta_cos*POLE_MASS/(POLE_MASS+CART_MASS)));
		float x_dot_dot=tmp-theta_dot_dot*theta_cos*(POLE_MASS+POLE_LENGTH)/(POLE_MASS+CART_MASS);
		x+=TAU*x_dot;
		x_dot+=TAU*x_dot_dot;
		theta+=TAU*theta_dot;
		theta_dot+=TAU*theta_dot_dot;
		i++;
	}
	return ((float)i)/MAX_SIMULATION_STEPS;
}



void cartpole_end_callback(const neat_t* neat,const neat_genome_t* genome){
	float x=example_random_uniform(-MAX_START_VALUE,MAX_START_VALUE);
	float x_dot=example_random_uniform(-MAX_START_VALUE,MAX_START_VALUE);
	float theta=example_random_uniform(-MAX_START_VALUE,MAX_START_VALUE);
	float theta_dot=example_random_uniform(-MAX_START_VALUE,MAX_START_VALUE);
	float genome_in[4];
	float genome_out;
	unsigned int i=0;
	while (fabs(x)<=MAX_X_VALUE&&fabs(theta)<=MAX_THETA_VALUE&&i<MAX_SIMULATION_STEPS){
		genome_in[0]=x;
		genome_in[1]=x_dot;
		genome_in[2]=theta;
		genome_in[3]=theta_dot;
		neat_genome_evaluate(neat,genome,genome_in,&genome_out);
		float force=FORCE*((genome_out>0.5f)*2-1);
		float theta_cos=cos(theta);
		float theta_sin=sin(theta);
		float tmp=(force+theta_dot*theta_dot*theta_sin*(POLE_MASS+POLE_LENGTH))/(POLE_MASS+CART_MASS);
		float theta_dot_dot=(theta_sin*GRAVITY-theta_cos*tmp)/(POLE_LENGTH*(4/3.0f-theta_cos*theta_cos*POLE_MASS/(POLE_MASS+CART_MASS)));
		float x_dot_dot=tmp-theta_dot_dot*theta_cos*(POLE_MASS+POLE_LENGTH)/(POLE_MASS+CART_MASS);
		x+=TAU*x_dot;
		x_dot+=TAU*x_dot_dot;
		theta+=TAU*theta_dot;
		theta_dot+=TAU*theta_dot_dot;
		printf("%f %f %f %f\n",x,x_dot,theta,theta_dot);
		i++;
	}
}
