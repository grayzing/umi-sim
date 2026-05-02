import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np

from src.simulation import Simulation

def visualize_network(simulation):
    """
    Visualizes the current positions of UEs and Sectors (gNBs).
    """
    plt.figure(figsize=(10, 10))
    
    # 1. Plot UEs
    ue_x = [ue.position.x for ue in simulation.ues.values()]
    ue_y = [ue.position.y for ue in simulation.ues.values()]
    plt.scatter(ue_x, ue_y, c='blue', s=20, label='UE', alpha=0.6)
    
    # 2. Plot Sectors
    for gnb in simulation.gnbs.values():
        gnb_x, gnb_y = gnb.position.x, gnb.position.y
        plt.scatter(gnb_x, gnb_y, c='red', marker='^', s=100, label='gNB' if gnb.cell_id == 0 else "")
        
        # Draw sector boresight (direction) lines
        for ru in gnb.sectors:
            # We use the panel_angle_rad to draw the sector orientation
            angle = ru.panel_angle_rad
            length = 50  # Length of the direction line in meters
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            
            plt.arrow(gnb_x, gnb_y, dx, dy, head_width=10, head_length=10, fc='red', ec='red', alpha=0.5)

    plt.title(f"Network Topology at Time: {simulation.time}ms")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axis('equal')
    plt.show()

sim = Simulation()
sim.initialize_network(19, 100)
visualize_network(sim)