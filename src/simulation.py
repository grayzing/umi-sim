from enum import Enum
from time import time
from collections import deque
import numpy as np
import heapq
import h5py
import torch

from nr import NrUe, NrGnb, AdvancedSleepMode, AsmTransitionState, NrRadioUnit
from utils import Vector, euclidean_distance, propagation_delay, path_loss, spectral_efficiency, bbox_dtla, bbox_dtr, bbox_manhattan, fetch_buildings, convert_to_local_origin, generate_hex_grid, nudge_to_outdoor
from shapely import Point, LineString
from torch_geometric.data import Data, Batch


time_difference_tolerance: float = 1

def AdvancedSleepModeIntMapping(x) -> AdvancedSleepMode:
    assert x >= 0 and x <= 3

    if x == 0:
        return AdvancedSleepMode.ACTIVE
    
    if x == 1:
        return AdvancedSleepMode.SM1
    
    if x == 2:
        return AdvancedSleepMode.SM2
    
    if x == 3:
        return AdvancedSleepMode.SM3
    
    else:
        raise IndexError

class SimulationLogger:
    def __init__(self, filename="results.h5") -> None:
        
        self.file = h5py.File(filename, 'w')

        self.user_equipment_kpis = self.file.create_dataset(
            "user_equipment_kpis", (0,),
            maxshape=(None,),
            dtype=[
                ('time', 'f8'),
                ('ue_id', 'i4'),
                ('position_x', 'f4'),
                ('position_y', 'f4'),
                ('position_z', 'f4'),
                ('indoor', 'i4'),
                ('velocity_x', 'f8'),
                ('velocity_y', 'f8'),
                ('throughput', 'f4'),
                ('assoc_sector_id', 'i4'),
                ('sinr', 'f4'),
                ('rsrp', 'f4'),
                ('assigned_prbs','i4')
            ]
        )
        self.user_equipment_kpi_buffer: deque = deque([])

        self.sector_kpis = self.file.create_dataset(
            "sector_kpis", (0, ),
            maxshape=(None, ),
            dtype=[
                ('time', 'f8'),
                ('gnb_id', 'i4'),
                ('position_x', 'f4'),
                ('position_y', 'f4'),
                ('position_z', 'f4'),
                ('downtilt_angle_deg', 'f4'),
                ('sector_angle_deg', 'f4'),
                ('prb_utilization', 'f4'),
                ('average_throughput_mbps', 'f4'),
                ('total_throughput_mbps', 'f4'),
                ('num_ues', 'i4'),
                
            ]
        )
        self.sector_kpi_buffer: deque = deque([])
        
        # Ptrs
        self.user_equipment_kpi_buffer_ptr = 0
        self.sector_kpi_buffer_ptr = 0

    def update_kpi_buffers(self, ues: dict[int, NrUe], sectors: list[NrRadioUnit], timestamp: float):
        """
        Update KPI buffers given UEs and sectors.

        Args:
            ues (list[NrUe]): List of UEs
            sectors (list[NrRadioUnit]): List of sectors
            timestamp (float): Time (take time from simulation.time)
        """
        for ue in ues.values():
            self.user_equipment_kpi_buffer.appendleft((
                timestamp,
                ue.id,
                ue.position.x,
                ue.position.y,
                ue.position.z,
                int(ue.indoor == True),
                ue.velocity.x,
                ue.velocity.y,
                ue.max_throughput,
                ue.sector.id if ue.sector else -1,
                ue.sinr,
                ue.rsrp,
                ue.assigned_prbs
            ))

        for sector in sectors:
            total_connected_ues = len(sector.connected_ues)
            prb_utilization = 0.0
            total_throughput = sum([ue.max_throughput for ue in sector.connected_ues])
            average_throughput = 0.0
            if total_connected_ues > 0:
                average_throughput = total_throughput / total_connected_ues
                prb_utilization = sum([ue.assigned_prbs for ue in sector.connected_ues]) / total_connected_ues

            self.sector_kpi_buffer.appendleft((
                timestamp,
                sector.id,
                sector.parent.position.x,
                sector.parent.position.y,
                sector.parent.position.z,
                sector.panel_downtilt,
                np.rad2deg(sector.panel_angle_rad),
                prb_utilization,
                average_throughput,
                total_throughput,
                total_connected_ues
            ))

    def log(self):
        # Log everything buffered for the UE KPIs
        for kpi in self.user_equipment_kpi_buffer:
            self.user_equipment_kpis.resize((self.user_equipment_kpi_buffer_ptr + 1,) )
            self.user_equipment_kpis[self.user_equipment_kpi_buffer_ptr] = kpi
            self.user_equipment_kpi_buffer_ptr += 1
        self.user_equipment_kpi_buffer.clear()

        # Log everything buffered for the sector KPIs
        for kpi in self.sector_kpi_buffer:
            self.sector_kpis.resize((self.sector_kpi_buffer_ptr + 1,) )
            self.sector_kpis[self.sector_kpi_buffer_ptr] = kpi
            self.sector_kpi_buffer_ptr += 1
        self.sector_kpi_buffer.clear()

    def close(self):
        self.file.close()

class EmbbTrafficGenerator:
    def __init__(self, simulation: 'Simulation', packet_size: int, rate: int = 1) -> None:
        self.packet_size = packet_size
        self.rate = rate
        self.simulation: Simulation = simulation

    def generate_packet(self, source_radio_unit: NrRadioUnit, target_ue: NrUe):
        """
        Generate packet, calculate propagation delay + transmission delay based on channel conditions
        """
        if not target_ue.serving_gnb:
            return
        
        if target_ue.sinr < -5 or target_ue.max_throughput <= 0:
            # Drop the packet
            return
        
        source_gnb = source_radio_unit.parent
        p_delay = propagation_delay(target_ue.position, source_gnb.position)
        effective_rate = 0.8 * target_ue.max_throughput
        t_delay = self.packet_size / effective_rate
        # Calculate queuing delay
        q_delay = 0

        capacity = sum([ue.max_throughput for ue in source_radio_unit.connected_ues])
        if capacity > 0:
            demand_per_ue = (self.rate * self.packet_size) / 1000  # Mbps
            total_demand = len(source_radio_unit.connected_ues) * demand_per_ue
            utilization = total_demand / max(1e-6, capacity)
            if utilization >= 1.0:
                return
            # mean waiting time in M/M/1 queue (in ms)
            q_delay = t_delay * (utilization / (1 - utilization))
        else:
            q_delay = 0
        total_delay = (p_delay + t_delay + q_delay)

        self.simulation.schedule_event(total_delay, ActionType.FtpPacketReceive, args={
            "packet_size": self.packet_size,
            "source_gnb": source_gnb,
            "target_ue": target_ue,
            "total_delay": total_delay,

        })

class ActionType(Enum):
    FtpPacketSend = "FtpPacketSend"
    FtpPacketReceive = "FtpPacketReceive"
    AdvancedSleepModeAdjust = "AdvancedSleepModeAdjust"
    AdvancedSleepModeTransitionEnd = "AdvancedSleepModeTransitionEnd"

class GnbSleepModeStrategy(Enum):
    Control = 0
    Naive = 1
    RASM = 2
    Random = 3

class Event:
    def __init__(self, execution_time: float, action: ActionType, parent: 'Simulation', args: dict | None = None) -> None:
        self.execution_time = execution_time
        self.args: dict | None = args
        self.action: ActionType = action
        self.parent: 'Simulation' = parent

    def __lt__(self, other: 'Event'):
        return self.execution_time < other.execution_time
    
    def __le__(self, other: 'Event'):
        return self.execution_time <= other.execution_time
    
    def __gt__(self, other: 'Event'):
        return self.execution_time > other.execution_time
    
    def __ge__(self, other: 'Event'):
        return self.execution_time >= other.execution_time

    def do(self) -> None:
        """
        Perform whatever action is specified, using args as arguments.
        """
        if self.action == ActionType.FtpPacketReceive:
            assert self.args

            assert self.args["target_ue"]
            assert self.args["source_gnb"]
            assert self.args["packet_size"]    

            #packet: Packet = self.args["packet"]
            target_ue: NrUe = self.args["target_ue"]
            packet_size: int = self.args["packet_size"]
            total_delay: float = self.args["total_delay"]
            if not target_ue.sector:
                # Drop the packet
                return
            # Update UE instantaneous throughput
            target_ue.instantaneous_throughput = packet_size / total_delay
        elif self.action == ActionType.FtpPacketSend:
            assert self.args

            assert self.args["target_ue"]
            assert self.args["source_radio_unit"]

            source_radio_unit: NrRadioUnit = self.args["source_radio_unit"]
            target_ue: NrUe = self.args["target_ue"]

            if source_radio_unit == target_ue.sector:
                self.parent.traffic_generator.generate_packet(source_radio_unit, target_ue)
                return
            
            if target_ue.sector:
                self.parent.traffic_generator.generate_packet(target_ue.sector, target_ue)
                return

        elif self.action == ActionType.AdvancedSleepModeAdjust:
            assert self.args

            assert self.args["advanced_sleep_mode"]
            assert self.args["target_sector"]

            advanced_sleep_mode: AdvancedSleepMode = self.args["advanced_sleep_mode"]
            target_sector: NrRadioUnit = self.args["target_sector"]

            target_sector.set_advanced_sleep_mode(advanced_sleep_mode)
            target_sector.asm_transition_state = AsmTransitionState.DEBO

            self.parent.schedule_event(advanced_sleep_mode.value[1], ActionType.AdvancedSleepModeTransitionEnd, args={
                "target_sector": target_sector
            })

        elif self.action == ActionType.AdvancedSleepModeTransitionEnd:
            assert self.args
            assert self.args["target_sector"]
            target_sector: NrRadioUnit = self.args["target_sector"]
            target_sector.asm_transition_state = AsmTransitionState.NONE

class Simulation:
    def __init__(self, logging=False, delta: float = 1, energy_saving_strategy: GnbSleepModeStrategy = GnbSleepModeStrategy.Control) -> None:
        self.event_queue: list[Event] = []
        self.delta: float = delta
        self.gnbs: dict[int, NrGnb] = {}
        self.sectors: list[NrRadioUnit] = []
        self.ues: dict[int, NrUe] = {}

        self.traffic_generator: EmbbTrafficGenerator = EmbbTrafficGenerator(self, 800, 10) # 1600 kiB per sexond
        self.time = 0
        self.energy_saving_strategy: GnbSleepModeStrategy = energy_saving_strategy

        self.rsrp_handover_threshold = -100 #dBm
        self.thermal_noise_floor = -88 #dBm
        self.intersite_distance = 200 #meter
        self.neighbor_sector_breakpoint = -110 #meter
        self.bounds = (-500, 500) #meter x meter
        self.penetration_loss_std = 6.5 #dB
        self.bbox = bbox_dtla
        self.gdf = convert_to_local_origin(fetch_buildings(self.bbox))

        self.logger: SimulationLogger | None = None
        if logging == True:
            self.logger = SimulationLogger()
        
        self.logging = logging
        self.rng = np.random.default_rng() # Default fallback

    def schedule_event(self, time_to_execute: float, action: ActionType, args: dict | None = None) -> None:
        event = Event(self.time + time_to_execute, action, self, args)
        heapq.heappush(self.event_queue, event)

    def set_advanced_sleep_mode(self, target_sector: NrRadioUnit, advanced_sleep_mode: AdvancedSleepMode):
        if target_sector.asm_transition_state != AsmTransitionState.NONE:
            # warn("Tried to change ASM of gNB while it was in ASM transition")
            return
        tte = advanced_sleep_mode.value[0] + target_sector.advanced_sleep_mode.value[0] # Time To Execute should include the time it takes to transition to the desired ASM as well as leave the gNB's previous ASM.
        target_sector.asm_transition_state = AsmTransitionState.TRAN
        for ue in target_sector.connected_ues:
            target_sector.remove_ue(ue)
        self.schedule_event(tte, ActionType.AdvancedSleepModeAdjust, args={
            "advanced_sleep_mode": advanced_sleep_mode,
            "target_sector": target_sector
        })

    def initialize_network(self, n: int, m: int):
        """
        Generate n gnbs, m ues
        
        :param n: Number of gNBs
        :type n: int
        :param m: Number of UEs
        :type m: int
        """
        bounds = self.gdf.total_bounds
        hex_points = generate_hex_grid(bounds, self.intersite_distance)
        
        k = 0
        for x, y in hex_points:
            p = Point(x, y)
            
            # 1. Try to place at the grid point
            # 2. If blocked, nudge it
            valid_pos = nudge_to_outdoor(p, self.gdf, rng=self.rng)
            
            if valid_pos:
                # Spawn your gNB here
                self.gnbs[k] = NrGnb(valid_pos[0],valid_pos[1],10,10,k,self)
                self.gnbs[k].parent_scheduler = self

                for j, ru in enumerate(self.gnbs[k].sectors):
                    self.sectors.append(ru)
                    ru.id = 3*k + j

                k += 1
            else:
                # Fallback: Skip this gNB or place at map center
                # (If it's stuck inside a massive building block)
                continue

        for i in range(m):
            self.ues[i] = NrUe(0,0,1.5,0, 1.5, i+n, self)
            ue_building = self.find_building_for_ue(self.ues[i])
            if ue_building is not None:
                self.ues[i].indoor = True
            else:
                self.ues[i].indoor = False

            if self.rng.integers(0,2) == 0:
                self.ues[i].velocity.x *= -1
            
            if self.rng.integers(0,2) == 0:
                self.ues[i].velocity.y *= -1
            
            self.ues[i].position.x = self.rng.uniform(-500,501)
            self.ues[i].position.y = self.rng.uniform(-500,501)

        dummy_ue = NrUe(0,0,1.5,9999,1.5,9999,self)
        # Add best beam cards
        for sector in self.sectors:
            for other_sector in self.sectors:
                if other_sector == sector:
                    continue
                dummy_ue.position = sector.parent.position
                rsrp = self.calculate_o2i_penetration_loss(dummy_ue,other_sector)
                if rsrp >= self.neighbor_sector_breakpoint:
                    sector.adjacent_sectors.append(other_sector)
        del dummy_ue
        self.step()

    def find_building_for_ue(self, ue: NrUe):
        if self.gdf is None:
            raise Exception("GDF was not initialized")
            return
        ue_point = Point(ue.position.x, ue.position.y)
        
        # This uses the spatial index of the GeoDataFrame (very fast)
        # It returns a boolean mask of all buildings containing the point
        mask = self.gdf.contains(ue_point)
        
        # Get the rows where mask is True
        containing_buildings = self.gdf[mask]
        
        if not containing_buildings.empty:
            # Return the first building found (if UEs aren't overlapping buildings)
            return containing_buildings.iloc[0]
        return None

    def get_best_beam(self, ue: NrUe, blacklist: set[NrRadioUnit] | None = None) -> NrRadioUnit | None:
        # Hysteresis in dB to prevent excessive handovers
        HYSTERESIS = 5.0 
        
        active_sectors = [sector for sector in self.sectors if sector.advanced_sleep_mode == AdvancedSleepMode.ACTIVE]

        if not active_sectors:
            return None

        best_sector: NrRadioUnit | None = ue.sector
        sector: NrRadioUnit
        for sector in active_sectors:
            if len(sector.connected_ues) >= 66:
                continue
            r_candidate = self.calculate_rsrp_wrt_sector(ue, sector)
            if best_sector is None:
                best_sector = sector
            else:
                r_current = self.calculate_rsrp_wrt_sector(ue, best_sector)
                if r_candidate > (r_current + HYSTERESIS):
                    best_sector = sector
        return best_sector

    def generate_packet_arrival_times(self, window_duration: int) -> np.ndarray:
        inter_arrivals = self.rng.exponential(scale=1/self.traffic_generator.rate, size=int(self.traffic_generator.rate * window_duration * 2))
        arrivals = np.cumsum(inter_arrivals)
        arrivals = arrivals[arrivals <= window_duration]
    
        return arrivals
    
    def send_packets_at_times(self)->None:
        # Keep only arrivals within duration
        for gnb in self.gnbs.values():
            for ru in gnb.sectors:
                for ue in ru.connected_ues:
                    arrival_times = self.generate_packet_arrival_times(1) # Generate arrivals for 1 second
                    for t in arrival_times:
                        m_t = t * 1e3
                        self.schedule_event(
                            m_t,
                            ActionType.FtpPacketSend,
                            args={
                                "source_radio_unit": ru,
                                "target_ue": ue
                            }
                        )
    

        
    
    def total_energy_usage(self) -> float:
        return sum([sum([ru.get_power_consumption() for ru in gnb.sectors]) for gnb in self.gnbs.values()])

    def step(self):
        candidate_event: Event
        while self.event_queue:
            time_difference = abs(self.event_queue[0].execution_time - self.time)
            if time_difference <= 2:
                candidate_event = heapq.heappop(self.event_queue)
                candidate_event.do()
            else:
                break

        # Change UE position based on mobility model
        for ue in self.ues.values():
            # Stop UEs from leaving the bounds of the sim-- if they are near the edge, have them turn back
            if ue.position.x <= self.bounds[0] or ue.position.x >= self.bounds[1]:
                ue.velocity.x *= -1
            
            if ue.position.y <= self.bounds[0] or ue.position.y >= self.bounds[1]:
                ue.velocity.y *= -1

            new_position: Vector = Vector(0,0,ue.position.z)
            new_position_dx = ue.velocity.x * self.delta
            new_position_dy = ue.velocity.y * self.delta

            new_position.x = ue.position.x + new_position_dx
            new_position.y = ue.position.y + new_position_dy
                
            ue.set_position(new_position)

        # Handover to gNB with best RSRP, if RSRP is below threshold, update instantaneous rate
        best_beam: NrRadioUnit | None = self.sectors[0]
        if self.energy_saving_strategy == GnbSleepModeStrategy.Control:
            for ue in self.ues.values():
                if ue.rsrp <= self.rsrp_handover_threshold:
                    best_beam = self.get_best_beam(ue)

                    if best_beam:
                        if best_beam != ue.sector:
                            if ue.sector:
                                ue.sector.remove_ue(ue)
                            ue.sector = best_beam
                            ue.serving_gnb = best_beam.parent

                            best_beam.connected_ues.append(ue)
        else:
            """
            for ue in self.ues:
                    best_beam = self.gnbs[0]
                    for gnb in self.gnbs:
                        r = rsrp(gnb.position, ue.position, gnb.tx_freq, gnb.tx_power)
                        if r > rsrp(best_gnb.position, ue.position, best_gnb.tx_freq, best_gnb.tx_power):
                            best_gnb = gnb

                    if rsrp(best_gnb.position, ue.position, best_gnb.tx_freq, best_gnb.tx_power) < -100:
                        if ue.serving_gnb:
                            ue.serving_gnb.remove_ue(ue)
                        ue.serving_gnb = None
                        continue
                    
                    if best_gnb.radio_unit.advanced_sleep_mode != AdvancedSleepMode.ACTIVE:
                        # wake up O-RU and then in next few steps the UE will connect to it, since it should still be the best O-RU to connect to
                        self.set_advanced_sleep_mode(
                            best_gnb,
                            AdvancedSleepMode.ACTIVE
                        )
                    else:
                        if ue.serving_gnb:
                            ue.serving_gnb.remove_ue(ue)
                        ue.serving_gnb = best_gnb
                        best_gnb.connected_ues.append(ue)
            """
                        
        # Send FTP packet downlink to UEs
        # Homogeneous Poisson Arrival process with rate 1600 KiB/sec
        # Generate arrival times once per second
        if self.time % 1e3 == 0 or self.time == 0:
            self.send_packets_at_times()

        # Allocate PRBs for attached UEs
        for gnb in self.gnbs.values():
            for ru in gnb.sectors:
                ru.allocate()

        # Update total interference, RSRP, SINR, max throughput
        for ue in self.ues.values():
            if ue.sector and ue.serving_gnb:
                ue.total_interference = self.calculate_total_interference(ue)
                ue.rsrp = self.calculate_rsrp_wrt_sector(ue, ue.sector)
                ue.sinr = self.calculate_sinr(ue)
                ue.max_throughput = self.calculate_max_throughput(ue)

        if self.time > 0 and self.energy_saving_strategy == GnbSleepModeStrategy.Naive:
            if self.time % 50 == 0:
                for gnb in self.gnbs.values():
                    for ru in gnb.sectors:
                        if len(ru.connected_ues) == 0 and ru.asm_transition_state == AsmTransitionState.NONE and ru.advanced_sleep_mode == AdvancedSleepMode.ACTIVE:
                            self.set_advanced_sleep_mode(ru, AdvancedSleepMode.SM1)

        # Log KPIs
        if self.logging and self.logger:
            self.logger.update_kpi_buffers(self.ues, self.sectors, self.time)

            if self.time % 100 == 0:
                self.logger.log()

        self.time += self.delta

    def reset(self, rng=None):
        if rng is not None:
            self.rng = rng
        
        self.ues.clear()
        self.gnbs.clear()
        self.sectors.clear()

        if self.logging and self.logger:
            self.logger.sector_kpi_buffer.clear()
            self.logger.user_equipment_kpi_buffer.clear()

        self.initialize_network(19, self.rng.integers(150,400))

    def run(self, stop_time: float) -> None:
        assert stop_time > 0
        time: float = self.time
        while time < stop_time:   
            self.step()
            time += self.delta
            self.time = time
        if self.logging and self.logger:
            self.logger.close()

    def calculate_total_interference(self, ue: NrUe) -> float:
        """
        Calculate total interference experienced by UE ue.
        Used for SINR calculation
        
        :param ue: UE to calculate interference for
        :type u: NrUe
        :return: Total interference experience by UE ue.
        :rtype: float
        """
        if not ue.serving_gnb or not ue.sector:
            return -np.inf
        interference = 0.0
        for sector in ue.sector.adjacent_sectors:
            rsrp_dbm = self.calculate_rsrp_wrt_sector(ue, sector)

            load_factor = min(len(sector.connected_ues)/sector.num_prbs, 1.0)
            interference += load_factor * 10**(rsrp_dbm/10)
        return interference
    
    def calculate_rsrp_wrt_sector(self, ue: NrUe, sector: NrRadioUnit):
        """
        Calculate the received signal power of the UE WRT a given NrRadioUnit

        Args:

            ue (NrUe): The UE of interest
            sector (NrRadioUnit): The NrRadioUnit of interest

        Returns:
            float: RSRP wrt the sector
        """
        rsrp_with_o2i = sector.tx_power - self.calculate_o2i_penetration_loss(ue,sector) - path_loss(sector.parent.position, ue.position, sector.tx_freq, self.gdf, self.rng) + sector.calculate_directional_gain(ue)
        return rsrp_with_o2i 
    
    def calculate_sinr(self, ue: NrUe):
        """
        Calculate SINR of UE

        Args:
            ue (NrUe): UE to calculate SINR for.

        Returns:
            float: The SINR of the UE in dB
        """
        if not ue.serving_gnb or not ue.sector:
            return -np.inf
        
        sector = ue.sector
        received_signal_power = self.calculate_rsrp_wrt_sector(ue, sector)

        # Calculate transmission delay
        s_lin = 10**(received_signal_power / 10)
        i_lin = ue.total_interference
        n_lin = 10**(-88 / 10) # Thermal noise floor

        # Calculate SINR
        sinr_linear = s_lin / (i_lin + n_lin)
        sinr_db = 10*np.log10(sinr_linear)

        return sinr_db
    
    def calculate_max_throughput(self, ue: NrUe) -> float:
        """
        Calculate the maximum achievable throughput of the UE given channel conditions

        Args:
            ue (NrUe): The UE of interest

        Returns:
            float: Maximum achievable throughput of the UE in KiB/ms
        """
        if not ue.sector or not ue.serving_gnb:
            return -np.inf
        
        bandwidth_per_prb_mhz = 1.44
        assigned_bandwidth_mhz = ue.assigned_prbs * bandwidth_per_prb_mhz

        se = spectral_efficiency(ue.sinr)
        mimo_rank = 2 if ue.sinr >= 10 else 1

        max_throughput_mbps = assigned_bandwidth_mhz * se * mimo_rank
        return max_throughput_mbps
    
    def calculate_o2i_penetration_loss(self, ue: NrUe, sector: NrRadioUnit) -> float:
        """
        Calculate O2I PL in a high loss scenario between a UE and a sector

        Args:
            ue (NrUe): The UE of interest
            sector (NrRadioUnit): The sector of interest

        Returns:
            float: O2I PL in dB
        """
        # Calculate outdoor pathloss
        penetration_loss_iir_glass = 23 + 0.3 * sector.tx_freq # dB
        penetration_loss_concrete = 5 + 4 * sector.tx_freq # dB

        # Calculate all of the pathloss terms
        if ue.indoor and ue.building:
            x1 = self.rng.uniform(0,25)
            x2 = self.rng.uniform(0,25)
            d_2d_in = min(x1,x2) # 3GPP dictated this don't ask me.
            pathloss_building_penetration_loss = 5 - 10 * np.log10(0.7*10**(-1 * penetration_loss_iir_glass / 10) + 0.3*10**(-1 * penetration_loss_concrete / 10))
            pathloss_inside_building = 0.5 * d_2d_in

            # Calculate pathloss in dB
            total_o2i_penetration_loss = pathloss_building_penetration_loss + pathloss_inside_building + self.rng.normal(0, self.penetration_loss_std) #dB

            return total_o2i_penetration_loss
        else:
            return 0
        
    def state(self):
        """
        Get current state of simulation

        Returns:
            Data: The state of simulation as a graph Data object
        """
        source: list [int] = []
        destination: list [int] = []
        w: list[float] = []
        v: list[list[float]] = []

        for gnb in self.gnbs.values():
            # Build feature vector
            total_throughput = 0
            average_throughput = 0
            average_total_interference = 0
            for ru in gnb.sectors:
                if len(ru.connected_ues) > 0:
                    average_total_interference = sum([ue.total_interference for ue in ru.connected_ues])/len(ru.connected_ues)
                    total_throughput = sum([ue.max_throughput for ue in ru.connected_ues])
                    average_throughput = total_throughput / len(ru.connected_ues)
                v.append([gnb.position.x, gnb.position.y, gnb.position.z, ru.panel_angle_rad, ru.advanced_sleep_mode.value[2], total_throughput, average_throughput, average_total_interference])

        # Build edge vector
        for ue in self.ues.values():
            for sector in self.sectors:
                r = self.calculate_rsrp_wrt_sector(ue,sector)
                source.append(sector.id)
                destination.append(ue.id)
                w.append(r)        

        for ue in self.ues.values():
            v.append([ue.position.x, ue.position.y, ue.position.z, np.pi, -1, ue.max_throughput, -1, ue.total_interference])

        feature_vector: torch.Tensor = torch.tensor(v, dtype=torch.float32).detach().cpu()
        edge_vector: torch.Tensor = torch.Tensor()
        edge_vector: torch.Tensor = torch.tensor([source, destination], dtype=torch.int).detach().cpu()
        weight_vector: torch.Tensor = torch.tensor(w, dtype=torch.float32).detach().cpu()

        return feature_vector, edge_vector, weight_vector
    
    def reward(self) -> float:
        """
        Calculate reward given the state of the simulation
        
        :return: The sum of the average RSRP, average throughput, and sum of all sleep modes in the system
        :rtype: float
        """
        total_connected_ues = 0
        
        throughput_reward = 0
        sleep_reward = 0
        sla_violation_count = 0
        for sector in self.sectors:
            if sector.connected_ues:
                total_connected_ues += len(sector.connected_ues)

                for ue in sector.connected_ues:
                    if ue.max_throughput < ue.minimum_throughput:
                        sla_violation_count += 1
                    throughput_reward += min(ue.max_throughput / ue.minimum_throughput, 4.0)

                asm = sector.advanced_sleep_mode
                if asm == AdvancedSleepMode.SM1:
                    sleep_reward += 1
                elif asm == AdvancedSleepMode.SM2:
                    sleep_reward += 2
                elif asm == AdvancedSleepMode.SM3:
                    sleep_reward += 3

        if total_connected_ues == 0: # because this can only happen if ALL of the sectors are switched off...
            return -5
        return (throughput_reward)/len(self.sectors) + (sleep_reward)/(len(self.sectors)) - sla_violation_count/total_connected_ues

    

