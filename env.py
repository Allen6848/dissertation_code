from util_RL import *

import config
import numpy as np
import matplotlib.pyplot as plt

import openmm
import openmm.unit as units # Standardized unit import
from openmm.app.topology import Topology
from openmm.app.element import Element
import mdtraj
import csv

class Propagate_1D:
    def __init__(self,
                 N=config.num_bins,
                 kT=0.5981,
                 sim_time = 10000, # Keeping this parameter if used elsewhere
                 window_size = 100
                 # Removed max_traj_length from __init__ parameters, it will be derived
                 ):
        self.N = N
        self.kT = kT
        self.sim_time = sim_time
        self.window_size = window_size

        # --- FIX START ---
        # Calculate the actual expected length of a single trajectory from propagate function.
        # This is the number of frames saved per RL propagation step.
        # Ensure config.dcdfreq_mfpt is not zero to prevent division by zero.
        if config.dcdfreq_mfpt == 0:
            raise ValueError("config.dcdfreq_mfpt cannot be zero for trajectory length calculation.")

        self.actual_traj_frames_per_step = config.propagation_step // config.dcdfreq_mfpt

        # Initialize pos_traj to store 'config.propagation_step' number of these trajectories.
        # Each stored trajectory will now have a length of 'self.actual_traj_frames_per_step',
        # ensuring it matches the shape of coor_x from propagate.
        self.pos_traj = np.zeros((config.propagation_step, self.actual_traj_frames_per_step))

        # Update the max_traj_length attribute to reflect this calculated value,
        # ensuring internal consistency. This is used by pad_trajectory.
        self.max_traj_length = self.actual_traj_frames_per_step
        # --- FIX END ---


        self.nodes = np.arange(self.N)
        self.cumulative_bias = np.zeros(self.N)
        self.current_propagation_index = 0

        # Initialize global_gaussian_params to hold parameters for each propagation step
        # It should have a size equal to the maximum number of propagation steps in an episode.
        self.global_gaussian_params = np.zeros((config.propagation_step, 3))


        # Set up topology, system, integrator, etc.
        elem = Element(0, "X", "X", 1.0)
        top = Topology()
        top.addChain()
        top.addResidue("xxx", top._chains[0])
        top.addAtom("X", elem, top._chains[0]._residues[0])

        mass = 12.0 * units.amu # Using units.amu
        self.system = openmm.System()
        self.system.addParticle(mass)

        # 1. Add y, z constraints forces (static, always present)
        y_pot = openmm.CustomExternalForce("1e3 * y^2")
        y_pot.addParticle(0)
        z_pot = openmm.CustomExternalForce("1e3 * z^2")
        z_pot.addParticle(0)
        self.system.addForce(y_pot)
        self.system.addForce(z_pot)

        # 2. Create and add the static FES force
        self.static_fes_force, _ = apply_fes(
                                          particle_idx=0,
                                          gaussian_param=None,
                                          pbc=config.pbc,
                                          amp=config.amp,
                                          name="FES",
                                          mode=config.fes_mode,
                                          plot=False)
        self.static_fes_force_index = self.system.addForce(self.static_fes_force)
        # Store the initial FES expression (useful for debugging/logging, not directly used in force update now)
        self.initial_fes_expression = self.static_fes_force.getEnergyFunction()


        # 3. Initialize dynamic bias force (initially None or an empty force)
        self.dynamic_bias_force = None
        self.dynamic_bias_force_index = -1 # Index in the system's force list, -1 indicates not added yet


        # Integrator
        self.integrator = openmm.LangevinIntegrator(300*units.kelvin, # Using units.kelvin
                                                    1.0/units.picoseconds, # Using units.picoseconds
                                                    0.002*units.picoseconds) # Using units.picoseconds

        # Create simulation object
        self.simulation = openmm.app.Simulation(top, self.system, self.integrator, config.platform)

        # Initialize context after all initial forces are added
        self.simulation.context.setPositions(config.start_state)
        self.simulation.context.setVelocitiesToTemperature(300*units.kelvin) # Using units.kelvin
        self.simulation.minimizeEnergy() # This also implicitly reinitializes the context

        # Store anything else needed
        self.config = config
        # Initialize other variables like pos_traj, reach, etc.


    #initialize the K as state
    def define_states(self):
        states = create_K_1D(self.N)
        return states

    def reward(self, K, action):
        # Convert RL action to bias params
        # action is assumed to be array-like [a, b, c] where b is the x0 of the Gaussian

        # --- FIX START: Scale the 'b' component (x0) of the action ---
        # Assuming action[1] (the 'b' parameter) is a value that needs to be scaled
        # to the range of the x-coordinate (0 to 2*pi).
        # If your agent produces values between, say, 0 and 1, multiply by 2*np.pi.
        # If your agent produces values between -1 and 1, you might need (action[1] + 1) / 2 * 2*np.pi.
        # For this fix, we assume action[1] is a raw value that needs to be scaled
        # to the range of 0 to 2*pi if it's currently limited to a smaller range.
        # A simple multiplicative scaling is applied here. If the agent outputs normalized values (0 to 1),
        # this will map them correctly. If the agent outputs unnormalized values, this might need adjustment
        # in networks.py or by knowing the agent's output range.

        print(f"DEBUG: Raw action received by reward function: {action}")

        scaled_action = np.array(action, dtype=np.float64) # Create a copy to modify, ensure float type
        scaled_action[1] = action[1] * (2 * np.pi) # Scale the x0 component to 0-2*pi

        # Ensure that the scaled x0 stays within the domain boundaries [0, 2*pi]
        scaled_action[1] = np.clip(scaled_action[1], 0.0, 2.0 * np.pi)

        print(f"DEBUG: Scaled x0 (action[1]) for bias: {scaled_action[1]}")

        # --- FIX END ---

        # Ensure current_propagation_index is within bounds before assigning
        if self.current_propagation_index >= config.propagation_step:
            print(f"Warning: Attempted to store action at index {self.current_propagation_index}, "
                  f"which is out of bounds for global_gaussian_params of size {config.propagation_step}. "
                  "This indicates an episode might have run longer than expected without reset.")
            idx_to_store = config.propagation_step - 1
        else:
            idx_to_store = self.current_propagation_index

        self.global_gaussian_params[idx_to_store,:] = scaled_action # Store the scaled action

        # Call propagate_ld_wrapper
        (steps_run, new_dynamic_bias_force, new_dynamic_bias_force_index), traj, reached_flag, coor_x = propagate_ld_wrapper(
            simulation=self.simulation,
            prop_index=self.current_propagation_index,
            pos_traj=self.pos_traj,
            action_bias_params=scaled_action, # Keeping this parameter, though `scaled_action` is used for bias params
            dynamic_bias_force=self.dynamic_bias_force,
            dynamic_bias_force_index=self.dynamic_bias_force_index,
            config=self.config,
            global_gaussian_params=self.global_gaussian_params # Pass the full history of gaussians
        )

        # Update the env's stored dynamic bias force and its index
        self.dynamic_bias_force = new_dynamic_bias_force
        self.dynamic_bias_force_index = new_dynamic_bias_force_index

        # Compute reward as before, but using continuous traj `traj`
        distances = [abs(config.end_x_pos - s)**2 for s in traj]  # adjust if traj continuous
        ewa_distance = exponentially_weighted_average(distances, alpha=0.1, window_size=self.window_size)
        reached_bonus = reached_flag * 1000
        reward = reached_bonus - ewa_distance

        return reward, reached_flag, traj, coor_x


    def step(self, action):
        reward, reached_flag, traj, coor_x= self.reward(self.K, action)
        self.current_propagation_index += 1 # Increment index after processing step

        # Determine if the episode is done
        done_by_max_steps = (self.current_propagation_index >= config.propagation_step)
        done_by_reaching_target = (reached_flag == 1) # If reached_flag is 1, target is reached

        done = done_by_max_steps or done_by_reaching_target

        if done_by_max_steps and not done_by_reaching_target:
            print(f"Episode reached {config.propagation_step} steps without reaching target. Ending episode.")
        elif done_by_reaching_target:
            print(f"Target state reached at step {self.current_propagation_index}. Ending episode.")


        next_state = pad_trajectory(coor_x.flatten(), self.max_traj_length)

        return next_state, reward, reached_flag, traj, coor_x, done # Now returns `done` flag

    def reset(self):
        # Reset positions and velocities
        self.simulation.context.setPositions(config.start_state)
        self.simulation.context.setVelocitiesToTemperature(300*units.kelvin) # Using units.kelvin
        self.simulation.minimizeEnergy() # Re-minimize after position reset

        self.K = self.define_states()  # Store K as an attribute

        # Reset global_gaussian_params to all zeros for a new episode
        # Ensure consistent size with config.propagation_step
        self.global_gaussian_params = np.zeros((config.propagation_step, 3))


        # IMPORTANT: When resetting, we also need to clear or reset the dynamic bias force.
        # The simplest way is to remove the existing dynamic bias force and set its reference to None.
        if self.dynamic_bias_force_index != -1:
            try:
                self.system.removeForce(self.dynamic_bias_force_index)
            except Exception as e:
                print(f"Warning: Could not remove dynamic bias force during reset: {e}")
                # Fallback: try to find and remove if index was stale
                for i in range(self.system.getNumForces()):
                    if self.system.getForce(i) is self.dynamic_bias_force:
                        self.system.removeForce(i)
                        print(f"Removed dynamic bias force by reference during reset at index {i}")
                        break

            self.dynamic_bias_force = None
            self.dynamic_bias_force_index = -1
            # Reinitialize context after removing the force to ensure force list is updated
            self.simulation.context.reinitialize(preserveState=True)


        # Reset current propagation index
        self.current_propagation_index = 0

        initial_traj = [config.start_x_pos] # config.start_x_pos is a float, pad_trajectory expects array/list
        state = pad_trajectory(initial_traj, self.max_traj_length)
        return state


