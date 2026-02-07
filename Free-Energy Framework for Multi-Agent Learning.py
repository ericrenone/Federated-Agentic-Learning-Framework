#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Simulates multi-agent LLM nodes minimizing Free Energy under
Information Geometry + Rational Inattention + Nash-Pareto constraints.

Fully self-contained Python single-file implementation.
No external dependencies beyond standard library.

"""

import math
import random
import tkinter as tk
from itertools import product

# ---------------- Configuration ----------------
class Config:
    NUM_AGENTS = 5         # Number of federated LLM nodes
    NUM_STATES = 10        # Discrete state space for simulation
    STEPS = 100            # Simulation steps
    ETA = 0.1              # Learning rate
    LAMBDA = 0.5           # Attention budget (Rational Inattention)
    SEED = 42              # Random seed for reproducibility

random.seed(Config.SEED)

# ---------------- Utility Functions ----------------
def kl_divergence(Q, P):
    """Discrete KL divergence D_KL(Q||P)."""
    return sum(Q[i] * math.log(Q[i]/P[i]) if Q[i]>0 else 0 for i in range(len(Q)))

def mutual_information(Q, P_joint):
    """Compute mutual information I(S;O) from joint distribution P(S,O)."""
    num_states = len(Q)
    P_s = [sum(P_joint[s][o] for o in range(num_states)) for s in range(num_states)]
    P_o = [sum(P_joint[s][o] for s in range(num_states)) for o in range(num_states)]
    I = 0.0
    for s in range(num_states):
        for o in range(num_states):
            if P_joint[s][o] > 0:
                I += P_joint[s][o] * math.log(P_joint[s][o]/(P_s[s]*P_o[o]))
    return I

def normalize(dist):
    """Normalize a distribution to sum to 1."""
    total = sum(dist)
    if total == 0:
        return [1.0/len(dist)]*len(dist)
    return [d/total for d in dist]

def fisher_metric(Q):
    """Simple diagonal approximation of Fisher Information for discrete states."""
    return [1.0/max(q,1e-6) for q in Q]

# ---------------- Agent Class ----------------
class Agent:
    def __init__(self, num_states, attention_budget):
        self.num_states = num_states
        self.Q = normalize([random.random() for _ in range(num_states)])
        self.attention = attention_budget
        # True environment distribution (can be stochastic, LLM-inspired)
        self.P = normalize([random.random() for _ in range(num_states)])
        # Joint distribution for mutual information
        self.P_joint = [[self.P[s]*self.P[o] for o in range(num_states)] for s in range(num_states)]
        self.history = []

    def free_energy(self):
        F = kl_divergence(self.Q, self.P) + self.attention * mutual_information(self.Q, self.P_joint)
        return F

    def natural_gradient_step(self, eta):
        g_inv = fisher_metric(self.Q)
        # Approximate gradient: (Q - P) + lambda * MI derivative ~ simplified
        grad = [(self.Q[i]-self.P[i]) + self.attention*0.01 for i in range(self.num_states)]
        self.Q = normalize([self.Q[i] - eta * g_inv[i]*grad[i] for i in range(self.num_states)])
        self.history.append(self.Q.copy())

# ---------------- Simulation ----------------
class Simulation:
    def __init__(self, num_agents, num_states, steps, eta, attention):
        self.agents = [Agent(num_states, attention) for _ in range(num_agents)]
        self.steps = steps
        self.eta = eta

    def run(self):
        for t in range(self.steps):
            # Step 1: Each agent updates via natural gradient
            for agent in self.agents:
                agent.natural_gradient_step(self.eta)
            # Step 2: Nash-Pareto enforcement (simplified averaging)
            avg_Q = [sum(agent.Q[i] for agent in self.agents)/len(self.agents) for i in range(self.agents[0].num_states)]
            for agent in self.agents:
                agent.Q = normalize([(agent.Q[i]+avg_Q[i])/2 for i in range(len(agent.Q))])
        return self.agents

# ---------------- Visualization ----------------
class Visualizer:
    def __init__(self, agents):
        self.agents = agents
        self.root = tk.Tk()
        self.root.title("Canonical LLM Free Energy Simulation")
        self.canvas = tk.Canvas(self.root, width=800, height=400)
        self.canvas.pack()
        self.colors = ["red","green","blue","orange","purple","brown","cyan","magenta","yellow","gray"]

    def draw(self, step):
        self.canvas.delete("all")
        width = 700
        height = 300
        margin = 50
        num_states = len(self.agents[0].Q)
        for idx, agent in enumerate(self.agents):
            Q_hist = agent.history[min(step,len(agent.history)-1)]
            for s, val in enumerate(Q_hist):
                x = margin + s*(width/num_states)
                y = height - val*height
                self.canvas.create_rectangle(x, height, x+width/num_states*0.8, y, fill=self.colors[s%len(self.colors)])
        self.root.update()

    def animate(self):
        for t in range(len(self.agents[0].history)):
            self.draw(t)
            self.root.after(50)
        self.root.mainloop()

# ---------------- Main ----------------
if __name__ == "__main__":
    sim = Simulation(
        num_agents=Config.NUM_AGENTS,
        num_states=Config.NUM_STATES,
        steps=Config.STEPS,
        eta=Config.ETA,
        attention=Config.LAMBDA
    )
    agents = sim.run()
    # Print final Free Energy
    for idx, agent in enumerate(agents):
        print(f"Agent {idx+1} final Free Energy: {agent.free_energy():.6f}")
    # Visualize evolution
    viz = Visualizer(agents)
    viz.animate()
