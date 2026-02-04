Here is a revised technical specification for an **Agentic AlphaEvolve Workflow**.

This specification refactors the monolithic controller into a decentralized multi-agent system. Each function (Search, Mutation, Scoring) is handled by distinct, ephemeral agents that are spun up on demand and terminated after task completion, preserving the system's "embarrassingly parallel"  nature.

### **Technical Specification: Agentic AlphaEvolve**

#### **I. Core Architecture: The Orchestrator & The Swarm**

Instead of a single loop, the system consists of a persistent **Orchestrator** that manages a swarm of ephemeral agents. The state is externalized to a shared **Program Database** to allow agents to operate independently.

1. **Master Orchestrator (Persistent):** The "manager" agent. It monitors the budget and spawns Search Agents. It does not execute code or query LLMs itself.  
2. **Program Database (Shared State):** The "memory" accessible by all agents. It stores programs, scores, and lineage information.

3. **Ephemeral Agents (The Swarm):** Short-lived agents created for a single generation cycle.

#### ---

**II. Agent Definitions**

**1\. Search Agent (The "Explorer")**

* **Lifecycle:** Created by the Orchestrator. Runs for *one* evolutionary step (selection → mutation → scoring → storage). Terminates upon completion.  
* **Parallelism:** The Orchestrator can spawn $N$ Search Agents simultaneously to run in parallel.

* **Responsibilities:**  
  * **Selection:** Queries the ProgramDatabase to select a parent program using MAP-Elites or Island Model logic.

  * **Context Construction:** Builds the prompt using the parent code, past feedback, and system instructions.

  * **Delegation:** It acts as a supervisor for the Mutation and Scoring agents, coordinating their inputs and outputs.  
  * **Commit:** If the cycle is successful, it writes the new program and its score back to the ProgramDatabase.

**2\. Mutation Agent (The "Coder")**

* **Lifecycle:** Spawned by a Search Agent. Terminates after generating valid code.  
* **Responsibilities:**  
  * **LLM Querying:** Queries the LLM ensemble (Gemini Flash/Pro)  with the prompt provided by the Search Agent.

  * **Diff Generation:** Requests changes in the \<\<\<\<\<\<\< SEARCH / \>\>\>\>\>\>\> REPLACE format for targeted updates.

  * **Patching:** Applies the diff to the parent program to produce the Child Program.

  * **Syntax Check:** Runs a basic syntax validation. If the code is invalid, it can self-correct (retry) or report failure to the Search Agent.

**3\. Scoring Agent (The "Tester")**

* **Lifecycle:** Spawned by a Search Agent. Terminates after returning a score.  
* **Responsibilities:**  
  * **Sandbox Creation:** Sets up an isolated environment for execution.  
  * **Cascade Evaluation:** Implements the hypothesis testing cascade:

    * *Phase 1:* Runs fast, small-scale tests. Fails fast if unsuccessful.  
    * *Phase 2:* Runs full-scale, expensive tests (only if Phase 1 passes).  
  * **LLM Grading:** Optionally invokes a separate LLM call to grade qualitative metrics (e.g., simplicity).

  * **Metric Return:** Returns a dictionary of scalars (metrics) and execution logs to the Search Agent.

#### ---

**III. Workflow Sequence (Parallel Track)**

For every parallel slot available (e.g., 50 parallel slots), the Orchestrator initiates this sequence:

1. **Orchestrator** creates **Search Agent \#N**.  
2. **Search Agent** reads DB, selects Parent Code, and compiles Context Prompt.  
3. **Search Agent** spawns **Mutation Agent**.  
   * *Mutation Agent* calls LLM → gets Diffs → applies Patch → returns Child Code.  
   * *Mutation Agent* terminates.  
4. **Search Agent** spawns **Scoring Agent** (passing Child Code).  
   * *Scoring Agent* runs evaluate() (Cascade Phase 1 & 2).  
   * *Scoring Agent* returns Score Dict.  
   * *Scoring Agent* terminates.  
5. **Search Agent** saves Child Code \+ Score Dict to DB.  
6. **Search Agent** terminates.

#### ---

**IV. Advantages of this Design**

* **Scalability:** Search Agents can be deployed across different machines or cloud functions, allowing the system to scale evaluation horizontally.

* **Fault Tolerance:** If a Scoring Agent crashes (common with experimental code) or hangs, it only kills that single ephemeral instance, not the main loop.  
* **Resource Efficiency:** Agents consume resources only when active. Heavy resources (like GPUs for scoring) are requested only by the Scoring Agent during its brief lifecycle.

#### **V. Advanced Feature: Meta-Prompt Agent**

To implement the "Meta Prompt Evolution" mentioned in the text, the Orchestrator can periodically spawn a specialized **Meta-Prompt Agent**.

* **Task:** It reads the ProgramDatabase to analyze which prompts led to high-scoring programs.  
* **Action:** It evolves the system instructions (the "meta prompt") stored in the configuration, effectively allowing the system to learn *how* to prompt itself better over time.
