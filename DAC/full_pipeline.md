1) we establish a connection using mavlink with the rov. No issue here.
2) We create a fake goal using the Fake joystick class. No issue here. 
3) We start the imu_listener that will collect data from the rov. No issue here. 
4) We create the Rov_gym_environment. No issue here. 
5) During environment resets, we set "self.state_history = [state] * self.history_length". This should not be an issue, it is as if the rov was stationnary for a while. 
6) We create the agent using DeterministicGCAgent. This then creates an actor and a Critic, both based on a GRU Network. They hae the same sizes appart from inputs and outputs. 
7) If needed, we switch the goal (during exploration and first learning phase)
then, inside the step loop : {
8) We select an action based on the phase (random or actor choice)
8.1) select_action takes a state, and outputs the action, after feeding the state to the actor. 
9) Environmnent step, where we apply an action for 0.1s, then compute the next state_sequence and the reward for our action.
10) We convert observations and action to an appropriate format before pushing them in the replay_buffer. 
11) Once in every "update_every", we update either the critic only (exploration-specific) or the critic and the actor. 
}
and we repeat 8-11 until convergence while saving occasionally. 