import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import keyboard  # Requires `pip install keyboard`
import time
import math
import random

# Simulation parameters
NUM_DRONES = 20
SIMULATION_DURATION = 30 # Number of simulated seconds
DT = 0.1 # Simulation time delta between steps in seconds
SIM_STEPS = int(SIMULATION_DURATION // DT)  # Total simulation steps
AREA_SIZE = 100.0  # 100x100 meter area
CELL_SIZE = 10.0  # 10x10 meter grid cells
COMM_RANGE = 20.0  # 2-hop communication range for STDMA
AVOIDANCE_RADIUS = 8.0  # Radius for collision avoidance
FRAME_DURATION = 1  # Frame duration in seconds
SLOTS_PER_FRAME = 5  # Number of time slots per frame
STEPS_PER_DT = 1 / DT
REPULSION_STRENGTH = 100.0  # Strength of collision avoidance force
PREDICTION_TIME = 1.0  # Time horizon for predictive avoidance
MAX_DRONE_SIZE = 2 # Max area of drones
MIN_DRONE_SIZE = 1 # Min area of drones
MAX_VELOCITY = 10
NEIGHBOR_TIMEOUT = 1 # Number of simulated seconds before a neighbor is forgotten
DRONE_ACCELERATION = 1.1 # How fast the drone accelerates

np.random.seed(3)  # Deterministic initialization

def step_to_time(step):
    return int(step * DT)

class Neighbor:
    def __init__(self, drone, last_seen):
        self.drone = drone
        self.last_seen = last_seen

# Drone class with predictive collision avoidance
class Drone:
    def __init__(self, id, position, velocity, size, slot):
        self.id = id
        self.position = np.array(position, dtype=float)  # [x, y]
        self.velocity = np.array(velocity, dtype=float)
        self.size = size  # Size of drone (for visualization and collision)
        self.slot = slot  # Assigned time slot
        self.neighbors = []  # List of (position, velocity) from neighbors
        self.collided = False
        self.radius = math.sqrt(size / np.pi)
        self.is_broadcasting = False
        self.cell_id = 0
        self.is_avoiding_collision = False

    def update_position(self):
        if self.collided:
            return

        # Always increase velocity until it reaches the cap
        if np.linalg.norm(self.velocity) < MAX_VELOCITY:
            self.velocity *= DRONE_ACCELERATION

        self.position += self.velocity * DT
        # Bounce off boundaries
        if self.position[0] < 0 or self.position[0] > AREA_SIZE:
            self.velocity[0] = -self.velocity[0]
            self.position[0] = np.clip(self.position[0], 0, AREA_SIZE)
        if self.position[1] < 0 or self.position[1] > AREA_SIZE:
            self.velocity[1] = -self.velocity[1]
            self.position[1] = np.clip(self.position[1], 0, AREA_SIZE)

    def broadcast_position(self, current_slot):
        #if self.collided:
        #    self.is_broadcasting = False
        #    return

        if self.slot == current_slot:
            self.is_broadcasting = True
        else:
            self.is_broadcasting = False

    def avoid_collisions(self):
        if not self.neighbors or self.collided:
            return

        future_self_pos = self.position + self.velocity * PREDICTION_TIME
        repulsion = np.zeros(2)

        self.is_avoiding_collision = False
        for neighbor in self.neighbors:
            neighbor_drone = neighbor.drone
            future_neighbor_pos = neighbor_drone.position + neighbor_drone.velocity * PREDICTION_TIME
            diff = future_self_pos - future_neighbor_pos
            dist = np.linalg.norm(diff)

            if dist < AVOIDANCE_RADIUS and dist > 0:
                repulsion_force = REPULSION_STRENGTH * (diff / dist) / dist
                repulsion += repulsion_force

        if np.linalg.norm(repulsion) > 0:
            # Apply repulsion as a change in velocity, capped to MAX_VELOCITY
            self.velocity += repulsion * DT
            speed = np.linalg.norm(self.velocity)
            if speed > MAX_VELOCITY:
                self.velocity = (self.velocity / speed) * MAX_VELOCITY
                self.is_avoiding_collision = True

    def align_with_neighbors(self):
        if not self.neighbors or self.collided:
            return

        # Average neighbor velocity
        avg_velocity = np.zeros(2)
        for neighbor in self.neighbors:
            avg_velocity += neighbor.drone.velocity
        avg_velocity /= len(self.neighbors)

        # Blend current velocity slightly toward the average
        alignment_strength = 0.4  # Small blending factor (tunable)
        self.velocity += alignment_strength * (avg_velocity - self.velocity)

        # Cap to max velocity
        #speed = np.linalg.norm(self.velocity)
        #if speed > MAX_VELOCITY:
        #    self.velocity = (self.velocity / speed) * MAX_VELOCITY

    def calc_cell_id(self):
        self.cell_id = int(self.position[0] // CELL_SIZE), int(self.position[1] // CELL_SIZE)

    def assign_slot(self):
        slot = hash(self.cell_id) % SLOTS_PER_FRAME
        for neighbor in self.neighbors:
            while slot == neighbor.drone.slot:
                slot = (slot + 1) % SLOTS_PER_FRAME
        self.slot = slot

    def remove_old_neighbors(self, time):
        for i, n in enumerate(self.neighbors):
            if n.last_seen + NEIGHBOR_TIMEOUT < time:
                print("removed neighbor")
                del self.neighbors[i]

# STDMA and communication

def distance_between_drones(a, b):
    return np.linalg.norm(a.position - b.position) - (a.radius + b.radius)

def get_neighbors(drones, time):
    for drone_a in drones:
        if not drone_a.is_broadcasting:
            continue
        for drone_b in drones:
            if drone_a.id == drone_b.id:
                continue
            if distance_between_drones(drone_a, drone_b) < COMM_RANGE:
                if len(drone_b.neighbors) == 0:
                    drone_b.neighbors.append(Neighbor(drone_a, time))
                else:
                    seen = False
                    for neighbor in drone_b.neighbors:
                        if neighbor.drone.id == drone_a.id:
                            neighbor.last_seen = time
                            seen = True
                            continue
                    if not seen:
                        drone_b.neighbors.append(Neighbor(drone_a, time))
                        print(f"time {time}, drone {drone_b.id} got new neighbor {drone_a.id}")

# Simulation state
class SimulationState:
    def __init__(self):
        self.drones = []
        self.step_history = []

    def initialize_drones(self):
        # Assign each drone a size between 5 and 10 (larger size increases collision chance)
        self.drones = [
            Drone(i,
                  [np.random.uniform(0, AREA_SIZE), np.random.uniform(0, AREA_SIZE)], # Position
                  [np.random.uniform(-MAX_VELOCITY, MAX_VELOCITY), np.random.uniform(-MAX_VELOCITY, MAX_VELOCITY)], # Velocity
                  np.random.uniform(MIN_DRONE_SIZE, MAX_DRONE_SIZE), # Drone size
                  int(np.random.uniform(0, SLOTS_PER_FRAME))) # Base slot
            for i in range(NUM_DRONES)
        ]

    def save_state(self, step):
        self.step_history.append(copy.deepcopy(self.drones))

    def load_state(self, step):
        self.drones = copy.deepcopy(self.step_history[step])

    def check_collisions(self):
        for drone_a in self.drones:
            for drone_b in self.drones:
                if drone_a.id == drone_b.id:
                    continue
                if drone_a.collided:
                    continue
                distance = np.linalg.norm(drone_a.position - drone_b.position)
                distance = distance - (drone_a.radius + drone_b.radius)
                # Collision occurs only if distance less than sum of sizes (markers overlap)
                if distance < 1.5:
                    drone_a.collided = True
                    drone_a.velocity = np.zeros(2)
                    drone_b.collided = True
                    drone_b.velocity = np.zeros(2)
                    continue

    def slot_from_step(self, step):
        current_time = int(step * DT)
        current_slot = int(current_time % SLOTS_PER_FRAME)
        return current_slot

    def run_step(self, step):
        current_slot = self.slot_from_step(step)
        current_time = step_to_time(step)
        get_neighbors(self.drones, current_time)
        self.check_collisions()

        # Each drone runs executes a series of tasks every simulation step.
        for drone in self.drones:
            drone.calc_cell_id()
            drone.assign_slot()
            drone.avoid_collisions()
            drone.align_with_neighbors()
            drone.update_position()
            drone.broadcast_position(current_slot)
            drone.remove_old_neighbors(current_time)

        self.save_state(step)

# Visualization

def simulate_swarm_with_navigation():

    sim = SimulationState()
    sim.initialize_drones()

    print("Calculating simulation")
    for step in range(SIM_STEPS):
        print(f"step {step} / {SIM_STEPS}")
        sim.run_step(step)

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(0, AREA_SIZE)
    ax.set_ylim(0, AREA_SIZE)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Drone Swarm with Predictive Avoidance (←/→ to navigate, Space to pause/resume)")

    scatter = ax.scatter([], [], c='blue', label="Drones", s=100)  # Default size, overridden in update
    slot_texts = [ax.text(0, 0, "", fontsize=8) for _ in range(NUM_DRONES)]
    comm_lines = []
    avoidance_circles = []

    # Draw grid
    for x in range(0, int(AREA_SIZE)+1, int(CELL_SIZE)):
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    for y in range(0, int(AREA_SIZE)+1, int(CELL_SIZE)):
        ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

    current_step = 0
    paused = False
    running = True

    def update_visualization(step):
        nonlocal comm_lines, avoidance_circles
        sim.load_state(step)
        positions = [drone.position for drone in sim.drones]
        slots = [drone.slot for drone in sim.drones]
        sizes = [drone.size * 25 for drone in sim.drones]  # Scale sizes for marker area
        velocities = [np.linalg.norm(drone.velocity) for drone in sim.drones]
        ids = [drone.id for drone in sim.drones]
        neighbors_all = [drone.neighbors for drone in sim.drones]

        colors = []
        current_slot = sim.slot_from_step(step)
        current_time = step_to_time(step)
        print(f"step {step}, current_slot {current_slot}, current_time: {current_time}")
        for drone in sim.drones:
            if drone.slot == current_slot:
                colors.append('yellow')  # Flash color for communicating drones
            elif drone.collided:
                colors.append('red')
            elif drone.is_avoiding_collision:
                colors.append('green')
            else:
                colors.append('blue')

        scatter.set_offsets(positions)
        scatter.set_color(colors)
        scatter.set_sizes(sizes)

        for i, (id, pos, slot, text, vel, neighbors) in enumerate(zip(ids, positions, slots, slot_texts, velocities, neighbors_all)):
            text.set_position((pos[0], pos[1] + 2))
            text.set_text(f"ID {id}\nSlot {slot}\nVel: {vel:0.2f}\nneighbors: {len(neighbors)}")

        # Remove old communication lines and circles
        for line in comm_lines:
            line.remove()
        comm_lines = []

        # Draw communication links
        #for i, slot in enumerate(slots):
        #    if slot == current_slot:
        #        for neighbor_id in get_neighbors(sim.drones, i, COMM_RANGE):
        #            line, = ax.plot(
        #                [positions[i][0], positions[neighbor_id][0]],
        #                [positions[i][1], positions[neighbor_id][1]],
        #                'r-', alpha=0.5
        #            )
        #            comm_lines.append(line)

        # Draw avoidance radius circles around drones
        for circle in avoidance_circles:
            circle.remove()
        avoidance_circles = []
        for drone in sim.drones:
            circle = plt.Circle(drone.position, AVOIDANCE_RADIUS, color='green', fill=False, linestyle='dotted', alpha=0.3)
            ax.add_patch(circle)
            avoidance_circles.append(circle)

        # Draw communication radius circles around drones
        for drone in sim.drones:
            circle = plt.Circle(drone.position, COMM_RANGE, color='magenta', fill=False, linestyle='dotted', alpha=0.3)
            ax.add_patch(circle)
            avoidance_circles.append(circle)

        current_time = step * DT
        ax.set_title(f"Step {step}/{SIM_STEPS-1}, Time: {int(current_time)}, Slot {current_slot}, {'Paused' if paused else 'Running'}")
        plt.draw()
        plt.pause(0.01)

    update_visualization(current_step)

    # Print info to the terminal
    print("Controls: ← (rewind), → (fast-forward), Space (pause/resume), Q (quit)")
    print(f"FRAME_DURATION: {FRAME_DURATION}")
    print(f"SLOTS_PER_FRAME: {SLOTS_PER_FRAME}")
    print(f"DT: {DT}")

    while running:
        if keyboard.is_pressed('left') and current_step > 0:
            current_step -= 1
            update_visualization(current_step)
            time.sleep(0.1)
        elif keyboard.is_pressed('right') and current_step < SIM_STEPS - 1:
            current_step += 1
            update_visualization(current_step)
            time.sleep(0.1)
        elif keyboard.is_pressed('space'):
            paused = not paused
            update_visualization(current_step)
            time.sleep(0.2)
        elif keyboard.is_pressed('q'):
            running = False
        if not paused and current_step < SIM_STEPS - 1:
            current_step += 1
            update_visualization(current_step)
        time.sleep(DT)

    plt.ioff()
    plt.close()

# Run simulation
if __name__ == "__main__":
    simulate_swarm_with_navigation()
