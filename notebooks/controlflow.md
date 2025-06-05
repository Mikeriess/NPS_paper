## Activity Transition Simulation Summary (Concise)

Case activity transitions are simulated in the `CaseActivities` function:

*   **Agent-Driven**: Agents process cases they are assigned to.
*   **Work Hours**: Activity start times are automatically delayed if they fall outside standard business hours (Mon-Fri, 8 AM - 6 PM). End-time adjustments for work hours are not currently applied.
*   **Case Completion**: Upon reaching an "END" activity, the agent is set to idle, the case is marked "closed", and its full history is logged.
*   **Reproducibility**: The use of an initial `seed` and a global `counter` ensures that all choices and random draws are deterministic and repeatable.

### Activity Sequencing

The type of the next activity (e.g., "Task-Reminder", "Interaction", "Email", "END") is determined by a Markov chain logic. However, specific choices are pre-selected from `p_vectors` (based on a `seed` and `counter`) rather than being drawn probabilistically from a transition matrix at runtime. This ensures reproducibility. The conceptual transition probabilities underlying the `p_vector` generation are shown below.


### Conceptual Activity Transition Matrix

The following table shows the conceptual probabilities for transitioning from a current activity state to the next. *Note: The simulation uses pre-determined sequences (`p_vectors`) based on these conceptual probabilities for reproducibility.*

| From State      | To Task-Reminder | To Interaction | To Email | To END |
|-----------------|------------------|----------------|----------|--------|
| **Start**       | 0.00             | 0.92           | 0.08     | 0.00   |
| **Task-Reminder** | 0.08             | 0.00           | 0.67     | 0.25   |
| **Interaction** | 0.00             | 0.02           | 0.96     | 0.02   |
| **Email**       | 0.00             | 0.02           | 0.45     | 0.53   |
| **END**         | 0.00             | 0.00           | 0.00     | 1.00   |
