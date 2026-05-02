from simulation import Simulation
import plotly.graph_objects as go
import numpy as np

from utils import fetch_buildings, get_building_meshes, bbox_dtr, bbox_dtla, bbox_manhattan

def visualize_network_3d(simulation):
    fig = go.Figure()

    # 1. Plot UEs
    ue_x = [ue.position.x for ue in simulation.ues.values()]
    ue_y = [ue.position.y for ue in simulation.ues.values()]
    ue_z = [ue.position.z for ue in simulation.ues.values()]
    
    fig.add_trace(go.Scatter3d(
        x=ue_x, y=ue_y, z=ue_z,
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.8),
        name='UE'
    ))

    # 2. Draw Association Lines (Serving Links)
    # Using the None-separator trick to plot multiple lines in one trace
    x_lines, y_lines, z_lines = [], [], []
    for ue in simulation.ues.values():
        if ue.serving_gnb:
            x_lines.extend([ue.position.x, ue.serving_gnb.position.x, None])
            y_lines.extend([ue.position.y, ue.serving_gnb.position.y, None])
            z_lines.extend([ue.position.z, ue.serving_gnb.position.z, None])
    
    fig.add_trace(go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color='green', width=2),
        name='Serving Link'
    ))

    # 3. Plot Sectors and Orientation Vectors
    for gnb in simulation.gnbs.values():
        gnb_x, gnb_y, gnb_z = gnb.position.x, gnb.position.y, gnb.position.z
        
        # Add gNB point
        fig.add_trace(go.Scatter3d(
            x=[gnb_x], y=[gnb_y], z=[gnb_z],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name='gNB'
        ))

        # Add vectors for each sector
        for ru in gnb.sectors:
            downtilt_rad = np.radians(ru.panel_downtilt)
            azimuth_rad = ru.panel_angle_rad
            length = 20
            dx = length * np.cos(azimuth_rad) * np.cos(downtilt_rad)
            dy = length * np.sin(azimuth_rad) * np.cos(downtilt_rad)
            dz = -length * np.sin(downtilt_rad)

            fig.add_trace(go.Scatter3d(
                x=[gnb_x, gnb_x + dx],
                y=[gnb_y, gnb_y + dy],
                z=[gnb_z, gnb_z + dz],
                mode='lines',
                line=dict(color='red', width=5),
                name=f'Sector {ru.id}'
            ))

    buildings_gdf = fetch_buildings(simulation.bbox)
    mesh_data = get_building_meshes(buildings_gdf)

    fig.add_trace(go.Mesh3d(
        **mesh_data,
        color='lightblue',
        opacity=0.6,
        name='Buildings'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data'
        ),
        title="UMi Scenario in Downtown LA"
    )
    fig.show()

sim = Simulation(logging=True)
sim.initialize_network(19, 400)
sim.run(5)
visualize_network_3d(sim)