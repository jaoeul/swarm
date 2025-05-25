import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import keyboard  # Requires `pip install keyboard`
import time
import math
import random

# Simulation parameters
NUM_DRONES = 10
SIM_STEPS = 500  # Total simulation steps
AREA_SIZE = 100.0  # 100x100 meter area
CELL_SIZE = 10.0  # 10x10 meter grid cells
COMM_RANGE = 20.0  # 2-hop communication range for STDMA
AVOIDANCE_RADIUS = 8.0  # Radius for collision avoidance
SLOTS_PER_FRAME = 200  # Number of time slots per frame
FRAME_DURATION = 0.1  # Frame duration in seconds
DT = FRAME_DURATION / SLOTS_PER_FRAME  # Time step size
REPULSION_STRENGTH = 10.0  # Strength of collision avoidance force
PREDICTION_TIME = 1.0  # Time horizon for predictive avoidance
MAX_DRONE_SIZE = 5 # Max area of drones
MIN_DRONE_SIZE = 5 # Min area of drones
MAX_VELOCITY = 200
NEIGHBOR_TIMEOUT = 100 # Number of steps before a neighbor is forgotten

np.random.seed(100)  # Deterministic initialization

class Neighbor:
    def __init__(self, drone, last_seen):
        self.drone = drone
        self.last_seen = last_seen

# Drone class with predictive collision avoidance
class Drone:
    def __init__(self, id, position, velocity, size, slot):
        self.id = id
        self.position = np.array(position, dtype=float)  # [x, y]
        self.velocity = np.array(velocity, dtype=float) * 100  # Increase velocity by 100
        self.size = size  # Size of drone (for visualization and collision)
        self.slot = slot  # Assigned time slot
        self.neighbors = []  # List of (position, velocity) from neighbors
        self.collided = False
        self.radius = math.sqrt(size / np.pi)
        self.is_broadcasting = False
        self.cell_id = 0

    def update_position(self, dt):
        if self.collided:
            return

        # Always increase velocity until it reaches the cap
        if np.linalg.norm(self.velocity) < MAX_VELOCITY:
            self.velocity *= 1.1

        self.position += self.velocity * dt
        # Bounce off boundaries
        if self.position[0] < 0 or self.position[0] > AREA_SIZE:
            self.velocity[0] = -self.velocity[0]
            self.position[0] = np.clip(self.position[0], 0, AREA_SIZE)
        if self.position[1] < 0 or self.position[1] > AREA_SIZE:
            self.velocity[1] = -self.velocity[1]
            self.position[1] = np.clip(self.position[1], 0, AREA_SIZE)

    def broadcast_position(self, step):
        current_step = step % SLOTS_PER_FRAME
        if self.slot == current_step:
            self.is_broadcasting = True
        else:
            self.is_broadcasting = False

    def avoid_collisions(self):
        pass
        #if self.collided:
        #    return
        #repulsion = np.zeros(2)
        #for neighbor_pos, neighbor_vel in self.neighbor_data:
        #    future_self = self.position + self.velocity * PREDICTION_TIME
        #    future_neighbor = neighbor_pos + neighbor_vel * PREDICTION_TIME
        #    delta = future_self - future_neighbor
        #    distance = np.linalg.norm(delta)
        #    if 0 < distance < AVOIDANCE_RADIUS:
        #        repulsion += (delta / distance) * (REPULSION_STRENGTH / distance)
        #self.velocity += repulsion * DT
        #speed = np.linalg.norm(self.velocity)
        #max_speed = 500.0  # Adjusted for scaled-up velocities
        #if speed > max_speed:
        #    self.velocity = (self.velocity / speed) * max_speed

    def calc_cell_id(self):
        self.cell_id = int(self.position[0] // CELL_SIZE), int(self.position[1] // CELL_SIZE)

    def assign_slot(self):
        slot = hash(self.cell_id) % SLOTS_PER_FRAME
        for neighbor in self.neighbors:
            while slot == neighbor.drone.slot:
                slot = (slot + 1) % SLOTS_PER_FRAME
        self.slot = slot

    def remove_old_neighbors(self, step):
        for i, n in enumerate(self.neighbors):
            if n.last_seen + NEIGHBOR_TIMEOUT < step:
                del self.neighbors[i]

# STDMA and communication

def distance_between_drones(a, b):
    return np.linalg.norm(a.position - b.position) - (a.radius + b.radius)

def get_neighbors(drones, step):
    for drone_a in drones:
        if not drone_a.is_broadcasting:
            continue
        for drone_b in drones:
            if drone_a.id == drone_b.id:
                continue
            # Check if any drones are close to the broadcasting drone
            if distance_between_drones(drone_a, drone_b) < COMM_RANGE:
                # Check if already a neighbor
                for neighbor in drone_b.neighbors:
                    if neighbor.drone.id == drone_a.id:
                        neighbor.last_seen = step
                        continue
                # Otherwise add new neighbors
                drone_b.neighbors.append(Neighbor(drone_a, step))

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
                  [np.random.uniform(-2, 2), np.random.uniform(-2, 2)], # Velocity
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

    def run_step(self, step):
        current_slot = step % SLOTS_PER_FRAME
        get_neighbors(self.drones, step)
        self.check_collisions()

        # Each drone runs executes a series of tasks every simulation step.
        for drone in self.drones:
            drone.calc_cell_id()
            drone.assign_slot()
            drone.avoid_collisions()
            drone.update_position(DT)
            drone.broadcast_position(step)
            drone.remove_old_neighbors(step)

        self.save_state(step)

# Visualization

def simulate_swarm_with_navigation():

    sim = SimulationState()
    sim.initialize_drones()

    for step in range(SIM_STEPS):
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
        current_slot = step % SLOTS_PER_FRAME
        for drone in sim.drones:
            if drone.collided:
                colors.append('red')
            elif drone.slot == current_slot:
                colors.append('yellow')  # Flash color for communicating drones
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
        for circle in avoidance_circles:
            circle.remove()
        avoidance_circles = []

        current_slot = step % SLOTS_PER_FRAME
        print(step)
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
        for drone in sim.drones:
            circle = plt.Circle(drone.position, AVOIDANCE_RADIUS, color='green', fill=False, linestyle='dotted', alpha=0.3)
            ax.add_patch(circle)
            avoidance_circles.append(circle)

        # Draw communication radius circles around drones
        for drone in sim.drones:
            circle = plt.Circle(drone.position, COMM_RANGE, color='magenta', fill=False, linestyle='dotted', alpha=0.3)
            ax.add_patch(circle)
            avoidance_circles.append(circle)

        ax.set_title(f"Step {step}/{SIM_STEPS-1}, Slot {current_slot}, {'Paused' if paused else 'Running'}")
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
