from utils import Vector, spectral_efficiency
from enum import Enum
from shapely import Polygon
import numpy as np

class AdvancedSleepMode(Enum):
    ACTIVE = (0, 0, 0)
    SM1 = (0.00355, 0.0071, 1)
    SM2 = (0.5, 1, 2)
    SM3 = (5, 10, 3)
    
class AsmTransitionState(Enum):
    NONE = 0
    TRAN = 1
    DEBO = 2

class NrRadioUnit:
    def __init__(self, parent: 'NrGnb', panel_angle_rad: float, id: int = 0 ) -> None:
        """
        Initialize the RU

        :param parent: The NrGnb that owns this RU
        :type parent: NrGnb
        """

        self.parent: 'NrGnb' = parent
        self.id: int = id

        self.tx_power: float = 35.0 #dBm
        self.tx_freq: float = 30 #Ghz
        self.front_to_back_ratio: float = 30.0 # dB
        self.horizontal_beamwidth: float = 65.0 # degrees
        self.vertical_beamwidth: float = 65.0 # degrees
        self.max_array_gain: float = 29.0 # dB
        self.sidelobe_attenuation: float = 30.0 # dB

        self.bandwidth: float = 100 #Mhz
        self.num_prbs: int = 66 # For numerology 3

        # Panel config
        self.num_panel_elements_horizontal: int = 4
        self.num_panel_elements_vertical: int = 8
        self.num_panels_horizontal: int = 2
        self.num_panels_vertical: int = 2
        self.panel_angle_rad: float = panel_angle_rad # Radians
        self.panel_downtilt: float = 10 # Degrees

        # Other things
        self.connected_ues: list[NrUe] = []
        self.adjacent_sectors: list[NrRadioUnit] = []

        self.advanced_sleep_mode: AdvancedSleepMode = AdvancedSleepMode.ACTIVE
        self.asm_transition_state: AsmTransitionState = AsmTransitionState.NONE

    def allocate(self):
        """
        Allocate 66 PRBs using Proportional Fair scheduling logic.
        """
        if not self.connected_ues:
            return

        # 1. Calculate Instantaneous Rate (Ri) for 1 PRB for each UE
        # We use the rate of 1 PRB as the 'channel quality' indicator
        weights = []
        for ue in self.connected_ues:
            # Get bits-per-RE for current SINR
            eff = spectral_efficiency(ue.sinr) 
            # Ri = bits per slot for 1 PRB (approximate)
            ri = 1 * 168 * eff * 8000 / 1e3 # kb/s
            
            # 2. Calculate PF Priority
            # Use a floor of 1.0 to prevent billion-point priorities
            priority = ri / max(1.0, ue.average_throughput)
            weights.append(priority)

        # 3. Distribute the 66 PRBs proportional to the weights
        total_weight = sum(weights)
        if total_weight == 0: # Everyone is in outage
            for ue in self.connected_ues:
                ue.assigned_prbs = self.num_prbs // len(self.connected_ues)
        else:
            for i, ue in enumerate(self.connected_ues):
                # Proportional allocation
                ue.assigned_prbs = int(self.num_prbs * (weights[i] / total_weight) + 0.5)
                # print(f"UE {ue.id} assigned {ue.assigned_prbs} PRBs")

        # 4. Update the Moving Average Throughput (Ti)
        # This must happen after the throughput is calculated in the simulation step
        # Use alpha=0.01 for a 100ms window
        alpha = 0.01
        for ue in self.connected_ues:
            ue.average_throughput = (1 - alpha) * ue.average_throughput + alpha * ue.instantaneous_throughput

    def get_azimuth(self, ue: 'NrUe') -> float:
        """Calculate azimuth angle between two points

        Args:
            u (Vector): First point
            v (Vector): Second point

        Returns:
            float: Azimuth angle in degrees
        """
        theta_global = np.atan2(ue.position.y - self.parent.position.y, ue.position.x - self.parent.position.x)
        phi = theta_global - self.panel_angle_rad
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        
        return np.degrees(phi)
    
    def get_elevation_angle(self, ue: 'NrUe') -> float:
        """
        Calculate downtilt toward UE

        Args:
            ue (NrUe): The UE of interest

        Returns:
            float: Downtilt in degrees
        """
        delta_z = self.parent.position.z - ue.position.z # Assuming gNB is higher
        dist_2d = np.sqrt((self.parent.position.x - ue.position.x)**2 + 
                          (self.parent.position.y - ue.position.y)**2)
        
        # arctan2 returns radians, convert to degrees
        return np.degrees(np.arctan2(delta_z, dist_2d))
    
    def calculate_directional_gain(self, ue: 'NrUe') -> float:
        """
        Calculate the directional gain of this NrRadioUnit relative to the provided UE

        Args:
            ue (NrUe): The UE of interest

        Returns:
            float: The directional gain of the RU in dB
        """
        azimuthal_angle_deg = self.get_azimuth(ue)

        elevation_angle_deg = self.get_elevation_angle(ue)

        g_h = -min(12 * (azimuthal_angle_deg / self.horizontal_beamwidth)**2, self.front_to_back_ratio)
        g_v = -min(12 * ((elevation_angle_deg - self.panel_downtilt) / self.vertical_beamwidth)**2, self.sidelobe_attenuation)
        ue_gain = 8
        directional_gain = self.max_array_gain + g_h + g_v + ue_gain # Linear combination of dB values
        return max(directional_gain, self.max_array_gain - self.front_to_back_ratio)
    
    def remove_ue(self, ue: 'NrUe') -> None:
        """
        Remove a UE from this Radio Unit's resource allocation table.

        Args:
            ue (NrUe): The UE that must be removed
        """

        if not ue in self.connected_ues:
            return
        self.connected_ues.remove(ue)
        self.allocate()

        ue.assigned_prbs = 0
        ue.max_throughput = 0
        ue.instantaneous_throughput = 0

        ue.sector = None
        ue.serving_gnb = None

    def get_power_consumption(self) -> float:
        """
        Return power consumption of 4T4R RU at different ASMs according to table on
        Power Modeling of the O-RAN O-RU & Application of Advanced Sleep Modes for
        Enhanced Energy Efficiency by Usman et. al
    
        :return: Power consumption of the RU in W.
        :rtype: float
        """
        assert self.advanced_sleep_mode in AdvancedSleepMode

        if self.advanced_sleep_mode == AdvancedSleepMode.ACTIVE:
            return 397.0
        elif self.advanced_sleep_mode == AdvancedSleepMode.SM1:
            return 88.0
        elif self.advanced_sleep_mode == AdvancedSleepMode.SM2:
            return 40.0
        elif self.advanced_sleep_mode == AdvancedSleepMode.SM3:
            return 28.0
        
        return 0.0
        

    def set_advanced_sleep_mode(self, asm: AdvancedSleepMode)->None:
        """
        Change the advanced sleep mode of the RU.
        SM1 will turn off the PA and AF, takes one frame to activate/deactivate.
        
        :param self: Description
        :param asm: Description
        :type asm: ADVANCED_SLEEP_MODE
        """
        self.advanced_sleep_mode = asm


class NrGnb:
    """
    A class for the NrGnb. Since this is a system-level simulator, there is only
    some rough simulation of resource-block allocation

    """
    def __init__(self, x: float, y: float, z: float, height: float, cell_id: int = 0, parent = None) -> None:
        """
        Initialize NrGnb in accordance to 3GPP specifications for UMi scenario.
        
        :param x: The x value of the gNB's position (in meters)
        :type x: float
        :param y: The y value of the gNB's position (in meters)
        :type y: float
        :param z: The z value of the gNB's position (in meters)
        :type z: float
        :param height: The height of the gNB's antennae (in meters)
        :type height: float
        :param cell_id: The ID of the gNB. Only handled by NrHelper usually.
        :type cell_id: int
        :param parent: The NrHelper that owns the gNB. Optional.
        :type parent: NrHelper | None
        """
        self.cell_id: int = cell_id
        self.position: Vector = Vector(x,y,z)
        self.sectors: list[NrRadioUnit] = [
            NrRadioUnit(self, 0),
            NrRadioUnit(self, 2*np.pi/3),
            NrRadioUnit(self, 4*np.pi/3)
        ]

        self.parent_scheduler = parent
        self.adjacent_gnbs: list[NrGnb] = []

        # self.initialize_turtle()

    def get_best_sector(self, ue: 'NrUe') -> NrRadioUnit | None:
        best_sector: NrRadioUnit = self.sectors[0]
        for sector in self.sectors:
            if sector.advanced_sleep_mode != AdvancedSleepMode.ACTIVE:
                continue
            tentative_gain = sector.calculate_directional_gain(ue)
            if tentative_gain > best_sector.calculate_directional_gain(ue):
                best_sector = sector
        if best_sector.advanced_sleep_mode != AdvancedSleepMode.ACTIVE:
            return None
        return best_sector

    def get_parent(self):
        return self.parent_scheduler
    
    def set_parent(self, parent):
        self.parent_scheduler = parent

    def get_position(self) -> Vector:
        return self.position

    def remove_ue(self, ue: 'NrUe'):
        for ru in self.sectors:
            ru.remove_ue(ue)
    

class NrUe:
    def __init__(self, x: float, y: float, z: float, rnti: int, antenna_height: float, id: int = 0, parent_scheduler = None, servingGnb: NrGnb | None = None) -> None:
        """
        Docstring for __init__
        
        :param x: The x value of the UE's position (in meters)
        :type x: float
        :param y: The y value of the UE's position (in meters)
        :type y: float
        :param z: The z value of the UE's position (in meters)
        :type z: float
        :param servingGnb: The gNB serving this UE.
        :param rnti: The RNTI of the UE.
        :type rnti: float
        :param antenna_height: The height of the UE antenna.
        :type antenna_height: float
        :param parent_scheduler: The sceduler that can modify this UE.
        :type parent_scheduler: Scheduler 
        :type servingGnb: NrGnb | None
        """
        self.serving_gnb: NrGnb | None = servingGnb
        self.sector: NrRadioUnit | None = None
        if self.serving_gnb:
            self.sector = self.serving_gnb.sectors[0]

        self.position: Vector = Vector(x,y,z)

        self.antenna_height: float = antenna_height # meters
        self.velocity: Vector = Vector(0.00083, 0.00083, 0) # meters/ms

        self.id: int = id
        self.indoor: bool = False
        self.building: Polygon | None = None

        self.instantaneous_throughput: float = 0
        self.average_throughput: float = 1.0
        self.max_throughput: float = 0
        self.minimum_throughput: float = 50 # Mbps or Kbpms

        self.total_interference: float = 0
        self.rsrp: float = -np.inf
        self.sinr: float = 0
        self.assigned_prbs: int = 0

        self.tx_power: int = 20 #dBm
        self.tx_freq: float = 30 #gHz

        
        
    def get_position(self) -> Vector:
        return self.position
    
    def set_position(self, position: Vector) -> None:
        self.position = position





