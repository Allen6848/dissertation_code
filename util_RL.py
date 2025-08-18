import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.linalg import expm, logm, eig
import openmm
import openmm.unit as units # Standardized unit import
# Removed: from openmm import unit
# Removed: from openmm.unit import * # Removed to avoid conflict
from openmm import Vec3
from openmm.app.topology import Topology
from openmm.app.element import Element
import config
from tqdm import tqdm
import mdtraj
import os # Added import for os module to create directories


# this is utility function for main.py


def gaussian(x, a, b, c): #self-defined gaussian function
    return a * np.exp(-(x - b)**2 / (2*(c**2))) # Fixed potential division by zero if c is zero, also 2*c^2

def create_K_1D(N=100, kT=0.5981):
    #create the K matrix for 1D model potential
    #K is a N*N matrix, representing the transition rate between states
    #The diagonal elements are the summation of the other elements in the same row, i.e. the overall outflow rate from state i
    #The off-diagonal elements are the transition rate from state i to state j (or from j to i???)"
    x = np.linspace(0, 5*np.pi, N) #create a grid of x values
    y1 = np.sin((x-np.pi))
    y2 = np.sin((x-np.pi)/2)
    amplitude = 10
    xtilt = 0.5
    y = (xtilt*y1 + (1-xtilt)*y2) * 3


    K = np.zeros((N,N))
    for i in range(N-1):
        K[i, i + 1] = amplitude * np.exp((y[i+1] - y[i]) / 2 / kT)
        K[i + 1, i] = amplitude * np.exp((y[i] - y[i+1]) / 2 / kT) #where does this formula come from?
    for i in range(N):
        K[i, i] = 0
        K[i, i] = -np.sum(K[i, :])
    return K


#define a function calculating the mean first passage time
def mfpt_calc(peq, K):
    """
    peq is the probability distribution at equilibrium.
    K is the transition matrix.
    N is the number of states.
    """
    N = K.shape[0] #K is a square matrix.
    onevec = np.ones((N, 1))
    Qinv = np.linalg.inv(peq.T * onevec - K.T)

    mfpt = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            #to avoid devided by zero error:
            if peq[j] == 0:
                mfpt[i, j] = 0
            else:
                mfpt[i, j] = 1 / peq[j] * (Qinv[j, j] - Qinv[i, j])

    return mfpt


#here we define a function, transform the unperturbed K matrix,
#with given biasing potential, into a perturbed K matrix K_biased.

def bias_K_1D(K, total_bias, kT=0.596):
    """
    K is the unperturbed transition matrix.
    total_bias is the total biasing potential.
    kT is the thermal energy.
    This function returns the perturbed transition matrix K_biased.
    """
    N = K.shape[0]
    K_biased = np.zeros([N, N])

    for i in range(N-1):
        u_ij = total_bias[i+1] - total_bias[i]

        K_biased[i, i+1] = K[i, i+1] * np.exp(u_ij /(2*kT))
        K_biased[i+1, i] = K[i+1, i] * np.exp(-u_ij /(2*kT))

    for i in range(N):
        K_biased[i,i] = -np.sum(K_biased[:,i])
    return K_biased


def compute_free_energy(K, kT=0.596):
    """
    K is the transition matrix
    kT is the thermal energy
    peq is the stationary distribution #note this was defined as pi in Simian's code.
    F is the free energy
    eigenvectors are the eigenvectors of K

    first we calculate the eigenvalues and eigenvectors of K
    then we use the eigenvalues to calculate the equilibrium distribution: peq.
    then we use the equilibrium distribution to calculate the free energy: F = -kT * ln(peq)
    """
    evalues, evectors = np.linalg.eig(K)

    #sort the eigenvalues and eigenvectors
    index = np.argsort(evalues) #sort the eigenvalues, the largest eigenvalue is at the end of the list
    evalues_sorted = evalues[index] #sort the eigenvalues based on index

    #calculate the equilibrium distribution
    peq = evectors[:, index[-1]].T/np.sum(evectors[:, index[-1]]) #normalize the eigenvector
    #take the real, positive part of the eigenvector i.e. the probability distribution at equilibrium.
    peq = np.real(peq)

    peq = np.maximum(peq, 0) #take the positive part of the eigenvector

    #calculate the free energy
    F = -kT * np.log(peq + 1e-6) #add a small number to avoid log(0))

    return [peq, F, evectors, evalues, evalues_sorted, index]

def get_M(K, t):
    """
    Get Markov Matrix from Rate Matrix.
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("Input K must be a 2D square matrix.")

    K_norm = K.copy()
    N = K_norm.shape[0]


    #check this bit of code
    for i in range(N):
        # Sum of the entire i-th row
        row_sum = np.sum(K_norm[i, :])
        # Adjust the diagonal element to ensure row-sum = 0
        K_norm[i, i] -= row_sum

    # expm
    return expm(K_norm * t)

def get_K(M, t):
    """
    Get Rate Matric from Markov Matrix.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("Input M must be a 2D square matrix.")

    if t <= 0:
        raise ValueError("Time step t must be positive.")

    # Compute matrix logarithm
    K = logm(M) / t

    return K

def update_bias_potential(simulation, global_gaussian_params, prop_index, dynamic_bias_force, dynamic_bias_force_index, amp = 7, plot=False, time_tag=""):
    """
    Creates or updates a CustomExternalForce representing the sum of all accumulated Gaussian biases.
    This force is *separate* from the static FES force. It removes the old bias force and adds a new one.

    Args:
        simulation: OpenMM Simulation object.
        global_gaussian_params: A numpy array storing all (a, b, c) parameters for all propagations.
                                Shape: (max_propagations, 3).
        prop_index: The current propagation index (0-based).
        dynamic_bias_force: The *current* OpenMM CustomExternalForce object for dynamic biases (can be None).
        dynamic_bias_force_index: The index of the current dynamic bias force in the system (can be -1).
        amp: Amplitude factor for the bias.
        plot: Boolean, whether to plot the potential energy.
        time_tag: String, used for plot filename.

    Returns:
        new_dynamic_bias_force: The newly created (or updated) CustomExternalForce object for biases.
        new_dynamic_bias_force_index: The index of this new force in the system.
    """
    system = simulation.system
    context = simulation.context # Get context for plotting
    particle_idx = 0 # Assuming the particle being biased is always at index 0

    # 1. Construct the new combined bias expression
    # This expression represents the sum of ALL Gaussians up to the current prop_index.
    bias_expressions = []
    # Loop from 0 up to and including prop_index
    for i in range(prop_index + 1):
        # Ensure parameters for plotting are handled carefully, especially if prop_index is 0 initially.
        if i < global_gaussian_params.shape[0]: # Check if index is within bounds of global_gaussian_params
            bias_expressions.append(f"bias_A{i}*exp(-(x-bias_x0{i})^2/(2*bias_sigma_x{i}^2))")

    combined_bias_expression = " + ".join(bias_expressions) if bias_expressions else "0" # Default to "0" if no biases yet

    # 2. Create the NEW dynamic bias force object
    new_dynamic_bias_force = openmm.CustomExternalForce(combined_bias_expression)
    new_dynamic_bias_force.addParticle(particle_idx)

    # 3. Add all parameters for the new dynamic bias force
    # These parameters are for all gaussians from index 0 to prop_index
    for i in range(prop_index + 1):
        if i < global_gaussian_params.shape[0]: # Check if index is within bounds
            a_val = global_gaussian_params[i, 0] * 4.184 * amp
            b_val = global_gaussian_params[i, 1]
            c_val = global_gaussian_params[i, 2]
            # Handle potential division by zero for c_val in the expression if c_val is zero
            # OpenMM's expression parser should handle division by zero with a float, but avoid if possible.
            # A common approach is to add a small epsilon if c_val is too small, or clamp it.
            # For now, relying on OpenMM's handling, but if errors persist, check this.
            new_dynamic_bias_force.addGlobalParameter(f"bias_A{i}", a_val)
            new_dynamic_bias_force.addGlobalParameter(f"bias_x0{i}", b_val)
            new_dynamic_bias_force.addGlobalParameter(f"bias_sigma_x{i}", c_val)
        else:
            # If global_gaussian_params is not fully populated yet for this index
            # (e.g., first step of a new episode, but `prop_index` means a historical reference),
            # provide default parameters or handle as an error.
            # Given prop_index is the current step, this usually means i == prop_index
            # so this else block might not be hit if global_gaussian_params size is correct.
            pass


    # 4. Remove old dynamic bias force if it exists
    if dynamic_bias_force_index != -1 and dynamic_bias_force is not None:
        try:
            system.removeForce(dynamic_bias_force_index)
        except Exception as e:
            # Handle cases where the force might have been removed already or index is invalid
            print(f"Warning: Could not remove old bias force at index {dynamic_bias_force_index}: {e}")
            # Fallback: try to find and remove if index was stale but force still present
            for i in range(system.getNumForces()):
                if system.getForce(i) is dynamic_bias_force:
                    system.removeForce(i)
                    print(f"Removed old bias force by reference at new index {i}")
                    break


    # 5. Add the new dynamic bias force to the system
    new_dynamic_bias_force_index = system.addForce(new_dynamic_bias_force)

    # 6. Reinitialize the context (critical after force modifications)
    # This ensures the simulation's context reflects the updated forces.
    simulation.context.reinitialize(preserveState=True)

    # 7. Plotting the total potential if requested
    if plot:
        # Ensure plot directory exists
        plot_dir = "./plots"
        os.makedirs(plot_dir, exist_ok=True)

        x_vals = np.linspace(0, 2*np.pi, 200)  # from 0 to 2pi nm, higher resolution
        energies = []

        # >>> CRUCIAL STEP: SAVE THE CURRENT, ORIGINAL POSITIONS OF THE PARTICLE <<<
        original_positions = context.getState(getPositions=True).getPositions(asNumpy=True)

        for x_coord in x_vals:
            # Create a temporary position array for evaluation.
            # It's important to make a *copy* of original_positions so we don't
            # accidentally modify the array that holds the state of the simulation.
            temp_pos = original_positions.copy()
            # Set the particle's x-position to the current x_coord for evaluation,
            # keeping y and z at their original values (likely 0 due to constraints).
            temp_pos[particle_idx, 0] = x_coord # Using the 'units' alias
            
            # Set the positions in the OpenMM Context to these temporary positions
            context.setPositions(temp_pos)
            
            # Get the potential energy at these temporary positions
            state = context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(units.kilojoule_per_mole) # Using the 'units' alias
            energies.append(energy)

        # >>> CRUCIAL STEP: RESTORE THE ORIGINAL POSITIONS AFTER PLOTTING <<<
        # This ensures the simulation continues from where it left off,
        # unaffected by the position changes made for plotting.
        context.setPositions(original_positions)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, energies, label="Total Potential Energy (FES + Bias)")
        plt.xlabel("x-position (nm)")
        plt.ylabel("Potential Energy (kJ/mol)")
        plt.title(f"Prop {prop_index}: Total Potential Energy")
        plt.grid(True)
        plt.legend()
        plot_path_potential = os.path.join(plot_dir, f"{time_tag}_potential_prop_{prop_index}.png")
        plt.savefig(plot_path_potential)
        plt.close()
        print(f"Saved potential plot to {plot_path_potential}")


    # Return the new force object and its new index
    return new_dynamic_bias_force, new_dynamic_bias_force_index

def plot_trajectory(coor_x, prop_index, time_tag="", plot_dir="./plots"):
    """
    Plots the x-coordinate trajectory for a given propagation.

    Args:
        coor_x: A numpy array of x-coordinates for the trajectory.
        prop_index: The current propagation index.
        time_tag: String, used for plot filename.
        plot_dir: Directory to save the plot.
    """
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    # Time steps calculation must use the correct unit from 'units' alias
    time_steps = np.arange(len(coor_x)) * config.dcdfreq_mfpt * config.stepsize.value_in_unit(units.picoseconds)
    plt.plot(time_steps, coor_x, label="x-coordinate trajectory")
    plt.xlabel("Time (ps)")
    plt.ylabel("x-position (nm)")
    plt.title(f"Prop {prop_index}: x-coordinate Trajectory")
    plt.grid(True)
    plt.legend()
    plot_path_traj = os.path.join(plot_dir, f"{time_tag}_trajectory_prop_{prop_index}.png")
    plt.savefig(plot_path_traj)
    plt.close()
    print(f"Saved trajectory plot to {plot_path_traj}")


def propagate(simulation,
              prop_index,
              pos_traj,   #this records the trajectory of the particle. in shape: [prop_index, sim_steps, 3]
              steps=config.propagation_step,
              dcdfreq=config.dcdfreq_mfpt,
              stepsize=config.stepsize,
              num_bins=config.num_bins,
              pbc=config.pbc,
              time_tag=None,
              top=None,
              reach=None,
              global_gaussian_params=None,
              ):
    """
    from langevin_langevin_sim_mfpt_opt.py
    here we use the openmm context object to propagate the system.
    save the CV and append it into the CV_total.
    use the DHAM_it to process CV_total, get the partially observed Markov matrix from trajectory.
    return the current position, the CV_total, and the partially observed Markov matrix.
    """

    # Ensure trajectory directory exists
    os.makedirs(f"./trajectory/explore/", exist_ok=True)

    file_handle = open(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.dcd", 'bw')
    dcd_file = openmm.app.dcdfile.DCDFile(file_handle, top, dt = stepsize) #note top is no longer a global pararm, we need pass this.
    for _ in tqdm(range(int(steps/dcdfreq)), desc=f"Propagation {prop_index}"):
        simulation.integrator.step(dcdfreq)
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=pbc)
        dcd_file.writeModel(state.getPositions(asNumpy=True))
    file_handle.close()

    #save the top to pdb.
    with open(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.pdb", 'w') as f:
        openmm.app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)

    #we load the pdb and pass it to mdtraj_top
    mdtraj_top = mdtraj.load(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.pdb")

    #use mdtraj to get the coordinate of the particle.
    traj = mdtraj.load_dcd(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.dcd", top = mdtraj_top)#top = mdtraj.Topology.from_openmm(top)) #this will yield error because we using imaginary element X.
    coor = traj.xyz[:,0,:] #[all_frames,particle_idx,xyz] # we grep the particle 0.

    #we digitize the x, y coordinate into meshgrid (0, 2pi, num_bins)
    x_bins_range = np.linspace(0, 2*np.pi, num_bins) #hardcoded.

    #we digitize the coor into the meshgrid.
    coor_x = coor.squeeze()[:,:1] #we only take the xcoordinate.
    #we append the coor_xy_digitized into the pos_traj.
    if coor_x.squeeze().ndim > 1:
        pos_traj[prop_index,:] = coor_x.squeeze().flatten()
    else:
        pos_traj[prop_index,:] = coor_x.squeeze()


    #we take all previous digitized x and feed it into DHAM.
    coor_x_total = pos_traj[:prop_index+1,:].squeeze() #note this is in coordinate space np.linspace(0, 2*np.pi, num_bins)
    print("coor_x_total shape: ", coor_x_total.shape)
    #we now reshape it to [cur_propagation+1, num_bins, 1]
    # This reshaping might be problematic if num_bins is not the actual number of frames per trajectory.
    # coor_x_total represents the accumulated trajectory *segments*.
    # If it's used for DHAM, its shape might depend on DHAM's input requirements.
    # Assuming DHAM expects a flat list of all positions, or segments of positions.
    # Re-evaluating this part if DHAM fails. For now, matching previous logic.
    if coor_x_total.ndim == 1: # If only one propagation, it's 1D, make it 2D (1, length)
        coor_x_total = coor_x_total.reshape(1, -1, 1)
    else: # If multiple propagations, it's 2D (num_prop, length), reshape to (num_prop, length, 1)
        coor_x_total = coor_x_total.reshape(prop_index+1, -1, 1) # Reshape to (num_propagations, frames_per_prop, 1)

    print("coor_x_total shape: ", coor_x_total.shape)

    #here we load all the gaussian_params from previous propagations.
    #size of gaussian_params: [num_propagation, num_gaussian, 3] (a,b,c),
    # note for 2D this would be [num_propagation, num_gaussian, 5] (a,bx,by,cx,cy)
    """gaussian_params = np.zeros([prop_index+1, config.num_gaussian, 3])
    for i in range(prop_index+1):
        gaussian_params[i,:,:] = np.loadtxt(f"./params/{time_tag}_gaussian_fes_param_{i}.txt").reshape(-1,3)
        print(f"gaussian_params for propagation {i} loaded.")
    """
    #note: gaussian_params will be passed global now, as a numpy array, in shape [num_propagation, num_gaussian, 3]

    cur_pos = coor_x_total[-1] #the current position of the particle, in ravelled 1D form.

    #determine if the particle has reached the target state.
    reach = None # Initialize reach to None
    end_state_xyz = config.end_state.value_in_unit_system(units.md_unit_system)[0] # Using the 'units' alias
    end_state_x = end_state_xyz[:1]
    for index_d, d in enumerate(coor_x):
        #if the distance of current pos is the config.target_state, we set reach to index_d.
        target_distance = np.linalg.norm(d - end_state_x)
        if target_distance < 0.1: # Check if within a small tolerance
            reach = index_d * config.dcdfreq_mfpt
            break # Exit loop once reached

    return cur_pos, pos_traj, reach, coor_x


def propagate_ld_wrapper(simulation,
                         prop_index,
                         pos_traj,
                         action_bias_params,
                         dynamic_bias_force, # Pass the current dynamic bias force object
                         dynamic_bias_force_index, # Pass its index
                         config,
                         global_gaussian_params=None,
                         end_state=None):
    """
    Wrapper to run Langevin dynamics propagation for one RL step, consistent with RL env format.
    Handles applying and updating the dynamic bias force.

    Returns:
        (steps, new_dynamic_bias_force, new_dynamic_bias_force_index): Tuple containing steps run and updated bias force info
        trajectory: 1D array, the trajectory from this propagation
        not_reached: int, 1 if end state was reached, 0 otherwise
        coor_x: The raw x-coordinates from the trajectory.
    """
    print(global_gaussian_params[prop_index,:])
    # 1. Apply RL-generated bias potential (updates/replaces the dynamic bias force)
    # Pass time_tag and plot=True for plotting
    new_dynamic_bias_force, new_dynamic_bias_force_index = update_bias_potential(
        simulation=simulation,
        global_gaussian_params=global_gaussian_params,
        prop_index=prop_index,
        dynamic_bias_force=dynamic_bias_force,
        dynamic_bias_force_index=dynamic_bias_force_index,
        amp=config.amp, # Pass amp from config
        plot=True, # Enable plotting potential
        time_tag=config.time_tag
    )

    # 2. Run propagation
    steps = config.propagation_step
    cur_pos, pos_traj, reach_flag_from_propagate, coor_x = propagate(simulation=simulation,
                                                  prop_index=prop_index,
                                                  pos_traj=pos_traj,
                                                  steps=steps,
                                                  dcdfreq=config.dcdfreq_mfpt,
                                                  stepsize=config.stepsize,
                                                  num_bins=config.num_bins,
                                                  pbc=config.pbc,
                                                  time_tag=config.time_tag,
                                                  top=simulation.topology,
                                                  reach=None, # The internal `reach` from propagate is just a check for return.
                                                  global_gaussian_params=global_gaussian_params)

    # 3. Plot trajectory after propagation
    # Accumulate all trajectories up to the current propagation index (prop_index)
    # pos_traj has shape (config.propagation_step, self.actual_traj_frames_per_step)
    # The relevant part is pos_traj[:prop_index + 1, :]
    all_current_episode_trajectories = pos_traj[:prop_index + 1, :]
    # Flatten this into a single 1D array for plotting
    accumulated_coor_x = all_current_episode_trajectories.flatten()
    
    # Pass the accumulated trajectory to the plot function
    plot_trajectory(accumulated_coor_x, prop_index, time_tag=config.time_tag)

    # 4. Compare final position with end state
    if end_state is None:
        end_state = config.end_state.value_in_unit_system(units.md_unit_system)[0][:1] # Using the 'units' alias

    # Check if the final state is equal (within tolerance) to end state
    # Use reach_flag_from_propagate for immediate check, otherwise check final position
    not_reached = 1 if reach_flag_from_propagate is not None else 0
    # Also double check if the very last position is close to the end state, in case `reach` missed it
    if not_reached == 0 and np.allclose(coor_x[-1], end_state, atol=0.1): # Check last recorded x-coordinate
        not_reached = 1

    # 5. Output the trajectory for this RL step and Bin it
    bins = np.linspace(0, 2 * np.pi, config.num_bins + 1) # Ensure bins are dimensionless
    binned_x = np.digitize(coor_x.squeeze(), bins) - 1  # bin index starts from 0
    binned_x = np.clip(binned_x, 0, config.num_bins - 1)  # safety clipping

    # Store binned trajectory with shape (steps,) into pos_traj[prop_index, :]
    trajectory = binned_x
    #trajectory = pos_traj[prop_index, :len(binned_x)] # Return the actual binned trajectory portion


    return (steps, new_dynamic_bias_force, new_dynamic_bias_force_index), trajectory, not_reached, coor_x

# def bin_trajectory(positions, num_bins, low=0, high=2*np.pi):
#    bins = np.linspace(low, high, num_bins+1)
#    binned = np.digitize(positions, bins) - 1
#    binned = np.clip(binned, 0, num_bins-1)
#    return binned


def exponentially_weighted_average(arr, alpha=0.1, window_size=1000):
    """
    Computes an exponentially weighted average, starting from a given point before the last.
    """
    if len(arr) < window_size:
        # If array is shorter than window_size, just compute simple average or handle as appropriate.
        # For now, let's just return the average of the available data.
        if len(arr) == 0:
            return 0
        # If the array is very short, the concept of "window_size" past values doesn't fully apply.
        # Let's adapt this for shorter arrays or raise a more specific error if needed.
        # For now, we'll use the available length.
        start_idx = 0
        subset = arr
    else:
        start_idx = max(0, len(arr) - window_size)  # Ensure it doesn't go out of bounds
        subset = arr[start_idx:]  # Extract the last `window_size` values

    if len(subset) == 0: # Handle empty subset case
        return 0

    ewa = subset[0]  # Initialize with the first value in the subset
    for x in subset[1:]:
        ewa = alpha * x + (1 - alpha) * ewa  # Update rule

    return ewa

#learning rate decay
def get_learning_rate(episode, num_episodes, initial_lr=0.001, final_lr=0.0001):
    return initial_lr + (final_lr - initial_lr) * (episode / num_episodes)

def pad_trajectory(trajectory, fixed_length):
    """
    Pads the trajectory with zeros at the beginning so that its length is fixed_length.
    If the trajectory is longer than fixed_length, only the last fixed_length elements are returned.
    """
    traj_length = len(trajectory)
    if traj_length < fixed_length:
        padded_traj = np.pad(trajectory, (fixed_length - traj_length, 0), 'constant') # Use np.pad for efficiency
    else:
        padded_traj = trajectory[-fixed_length:]
    return padded_traj

def apply_fes(particle_idx, gaussian_param=None, pbc=False, name="FES", amp=7, mode="gaussian", plot=False, plot_path="./fes_visualization.png"):
    """
    Creates an OpenMM CustomExternalForce object for the static Free Energy Surface (FES).
    This function *does not* add the force to a system; it only returns the force object itself.

    Returns:
        fes_potential_force: The created CustomExternalForce object.
        initial_expression: The string expression used for this force.
    """
    pi = np.pi
    k = 5
    max_barrier = '1e2'
    offset = 0.4

    # Base expression for boundary potentials
    main_energy_expression = f"{max_barrier} * (1 / (1 + exp({k} * (x - (1-{offset}))))) + {max_barrier} * (1 / (1 + exp(-{k} * (x - (2 * {pi} + {offset})))))"

    fes_potential_force = None
    initial_expression = main_energy_expression # Default expression

    if mode == "multiwell":
        if pbc:
            raise NotImplementedError("pbc not implemented for multi-well potential.")
        else:
            num_hills = 9
            fesA_i = np.array([0.9, 0.3, 0.7, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp
            fesx0_i = [1.12, 1, 3, 4.15, 4, 5.27, 4.75, 6, 1]
            fessigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]

            multiwell_terms = []
            for i in range(num_hills):
                multiwell_terms.append(f"fesA{i}*exp(-(x-fesx0{i})^2/(2*fessigma_x{i}^2))")

            initial_expression = main_energy_expression + " + " + " + ".join(multiwell_terms) # Update expression

            fes_potential_force = openmm.CustomExternalForce(initial_expression)
            fes_potential_force.addParticle(particle_idx)

            # Add global parameters specific to the multiwell potential
            for i in range(num_hills):
                fes_potential_force.addGlobalParameter(f"fesA{i}", fesA_i[i] * 4.184)
                fes_potential_force.addGlobalParameter(f"fesx0{i}", fesx0_i[i])
                fes_potential_force.addGlobalParameter(f"fessigma_x{i}", fessigma_x_i[i])
    else: # mode is not "multiwell", use simple barriers
        fes_potential_force = openmm.CustomExternalForce(main_energy_expression)
        fes_potential_force.addParticle(particle_idx)

    # fes_potential_force should now always be created.
    # It is NOT added to a system here; it is returned for addition in env.py.
    return fes_potential_force, initial_expression


