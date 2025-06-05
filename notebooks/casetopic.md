## Case Topic Determination Summary

Case topics are assigned to arriving cases based on a pre-simulation step:

1.  **Pre-simulation**: Topics for potential cases are generated *before* their arrival times are simulated.
2.  **Batch Generation**: A fixed array of 100,000 topics (`sim_topics`) is created at the start of the arrival generation process.
3.  **Method**: Topics are selected using `numpy.random.choice`, which allows for weighted random sampling.
4.  **Inputs for Generation**:
    *   `topics`: A list of possible topic identifiers (e.g., integers 0-9).
    *   `P_topics`: A hardcoded list of probabilities, where each probability corresponds to a topic in the `topics` list. This defines the distribution of case topics. The specific probabilities are:
        *   Topic 0: 0.318670
        *   Topic 1: 0.243781
        *   Topic 2: 0.138256
        *   Topic 3: 0.066510
        *   Topic 4: 0.064415
        *   Topic 5: 0.063891
        *   Topic 6: 0.039277
        *   Topic 7: 0.037968
        *   Topic 8: 0.014925
        *   Topic 9: 0.012307
5.  **Assignment**: When a new case arrival is simulated, it is assigned the next topic from the pre-generated `sim_topics` array in sequential order.

In essence, the statistical distribution of topics is determined by `P_topics`, and a long sequence of these topics is generated upfront to be assigned to cases as they arrive in the simulation.
