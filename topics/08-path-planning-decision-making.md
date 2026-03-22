# Path Planning and Decision Making for Autonomous Vehicles

## Introduction

Path planning and decision making are critical components of autonomous driving systems. They transform perception outputs into safe, comfortable, and efficient driving maneuvers. This document covers the algorithms, architectures, and best practices for path planning in autonomous vehicles.

## Planning Hierarchy

### Multi-Level Planning

```
┌─────────────────────────────────────────┐
│          Mission Planning               │
│    (Route: A → B, hours/days ahead)     │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│       Behavioral Planning               │
│   (Lane change, merge, turn, seconds)   │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│        Motion Planning                  │
│    (Trajectory generation, seconds)     │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│          Control                        │
│   (Steering, throttle, brake, ms)       │
└─────────────────────────────────────────┘
```

## Mission Planning

### Route Planning

```python
import networkx as nx

class RoutePlanner:
    """High-level route planning"""
    
    def __init__(self, map_data):
        self.graph = self.build_graph(map_data)
        
    def plan_route(self, start, goal):
        """
        Args:
            start: (lat, lon) start position
            goal: (lat, lon) goal position
        Returns:
            route: list of waypoints
        """
        # Find nearest nodes
        start_node = self.find_nearest_node(start)
        goal_node = self.find_nearest_node(goal)
        
        # Find shortest path
        route = nx.shortest_path(
            self.graph,
            start_node,
            goal_node,
            weight='travel_time'
        )
        
        # Convert to waypoints
        waypoints = [self.graph.nodes[n]['position'] for n in route]
        
        return waypoints
    
    def build_graph(self, map_data):
        """Build road network graph"""
        G = nx.DiGraph()
        
        for road in map_data.roads:
            # Add edges with attributes
            G.add_edge(
                road.start_node,
                road.end_node,
                length=road.length,
                speed_limit=road.speed_limit,
                travel_time=road.length / road.speed_limit
            )
        
        return G
```

## Behavioral Planning

### Finite State Machine

```python
from enum import Enum

class DrivingState(Enum):
    LANE_KEEPING = 0
    LANE_CHANGE_LEFT = 1
    LANE_CHANGE_RIGHT = 2
    LEFT_TURN = 3
    RIGHT_TURN = 4
    YIELDING = 5
    EMERGENCY_STOP = 6


class BehavioralPlanner:
    """Behavioral planning with FSM"""
    
    def __init__(self):
        self.state = DrivingState.LANE_KEEPING
        self.state_data = {}
        
    def update(self, perception_output, route):
        """
        Args:
            perception_output: detected objects, lanes, etc.
            route: planned route
        Returns:
            behavior: desired behavior (target lane, speed, etc.)
        """
        # Check transitions
        if self.state == DrivingState.LANE_KEEPING:
            # Check if need to change lanes
            if self.should_change_lane(perception_output, route):
                if self.can_change_left(perception_output):
                    self.transition(DrivingState.LANE_CHANGE_LEFT)
                elif self.can_change_right(perception_output):
                    self.transition(DrivingState.LANE_CHANGE_RIGHT)
            
            # Check if need to turn
            if self.approaching_turn(route):
                if route.next_turn == 'left':
                    self.transition(DrivingState.LEFT_TURN)
                else:
                    self.transition(DrivingState.RIGHT_TURN)
            
            # Check if need to yield
            if self.should_yield(perception_output):
                self.transition(DrivingState.YIELDING)
        
        elif self.state == DrivingState.LANE_CHANGE_LEFT:
            if self.lane_change_complete():
                self.transition(DrivingState.LANE_KEEPING)
        
        # Similar for other states...
        
        return self.get_behavior()
    
    def transition(self, new_state):
        """Transition to new state"""
        print(f"Transitioning from {self.state} to {new_state}")
        self.state = new_state
        self.state_data = {}
    
    def get_behavior(self):
        """Get current behavior specification"""
        behavior = {
            'state': self.state,
            'target_lane': self.get_target_lane(),
            'target_speed': self.get_target_speed(),
            'target_heading': self.get_target_heading()
        }
        return behavior
    
    def should_change_lane(self, perception, route):
        """Determine if lane change is needed"""
        # Check route
        if route.current_lane != route.target_lane:
            return True
        
        # Check for slower vehicle ahead
        if perception.vehicle_ahead and perception.vehicle_ahead.speed < self.target_speed * 0.8:
            return True
        
        return False
    
    def can_change_left(self, perception):
        """Check if left lane change is safe"""
        # Check for vehicles in left lane
        if perception.left_lane_vehicle:
            gap = perception.left_lane_vehicle.distance
            relative_speed = self.speed - perception.left_lane_vehicle.speed
            
            # Need sufficient gap
            min_gap = 20 + max(0, -relative_speed) * 2
            
            return gap > min_gap
        
        return True
```

### Behavior Trees

```python
class BehaviorTree:
    """Behavior tree for complex decision making"""
    
    def __init__(self):
        self.root = Selector([
            Sequence([
                Condition(self.emergency_condition),
                Action(self.emergency_stop)
            ]),
            Sequence([
                Condition(self.approaching_destination),
                Action(self.park)
            ]),
            Sequence([
                Condition(self.should_yield),
                Action(self.yield_to_traffic)
            ]),
            Sequence([
                Condition(self.should_change_lane),
                Action(self.execute_lane_change)
            ]),
            Action(self.lane_keeping)  # Default behavior
        ])
    
    def tick(self, blackboard):
        """Execute behavior tree"""
        return self.root.tick(blackboard)


class Selector:
    """Execute children until one succeeds"""
    
    def __init__(self, children):
        self.children = children
    
    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status == Status.SUCCESS:
                return Status.SUCCESS
        return Status.FAILURE


class Sequence:
    """Execute children in sequence, all must succeed"""
    
    def __init__(self, children):
        self.children = children
        self.current = 0
    
    def tick(self, blackboard):
        while self.current < len(self.children):
            status = self.children[self.current].tick(blackboard)
            if status == Status.RUNNING:
                return Status.RUNNING
            elif status == Status.FAILURE:
                self.current = 0
                return Status.FAILURE
            self.current += 1
        
        self.current = 0
        return Status.SUCCESS


class Condition:
    """Check a condition"""
    
    def __init__(self, condition_fn):
        self.condition_fn = condition_fn
    
    def tick(self, blackboard):
        if self.condition_fn(blackboard):
            return Status.SUCCESS
        return Status.FAILURE


class Action:
    """Execute an action"""
    
    def __init__(self, action_fn):
        self.action_fn = action_fn
    
    def tick(self, blackboard):
        return self.action_fn(blackboard)


class Status:
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3
```

## Motion Planning

### Frenet Frame Planning

```python
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import minimize

class FrenetPlanner:
    """Motion planning in Frenet frame"""
    
    def __init__(self, reference_path):
        self.reference_path = reference_path
        self.path_tree = KDTree(reference_path)
        
    def plan_trajectory(self, current_state, goal_state, obstacles, dt=0.1, T=5.0):
        """
        Args:
            current_state: (x, y, theta, v, a)
            goal_state: (x, y, theta, v)
            obstacles: list of obstacle polygons
            dt: time step
            T: planning horizon
        Returns:
            trajectory: list of (x, y, theta, v, a, t)
        """
        # Convert to Frenet frame
        s0, d0 = self.xy_to_frenet(current_state[0], current_state[1])
        s_dot0 = current_state[3]  # velocity
        d_dot0 = 0  # lateral velocity
        
        # Sample trajectories
        trajectories = []
        
        for s_target in np.linspace(s0 + 10, s0 + 100, 10):
            for d_target in np.linspace(-3, 3, 7):
                for T_sample in np.linspace(2, 5, 4):
                    traj = self.generate_trajectory(
                        s0, d0, s_dot0, d_dot0,
                        s_target, d_target, T_sample, dt
                    )
                    
                    # Check collision
                    if not self.check_collision(traj, obstacles):
                        # Compute cost
                        cost = self.compute_cost(traj, goal_state, obstacles)
                        trajectories.append((cost, traj))
        
        # Select best trajectory
        if trajectories:
            trajectories.sort(key=lambda x: x[0])
            return trajectories[0][1]
        
        return None
    
    def generate_trajectory(self, s0, d0, s_dot0, d_dot0, s_target, d_target, T, dt):
        """Generate quintic polynomial trajectory"""
        # Longitudinal trajectory (quintic)
        s_traj = self.quintic_polynomial(
            s0, s_dot0, 0,  # initial: position, velocity, acceleration
            s_target, 0, 0,  # final
            T, dt
        )
        
        # Lateral trajectory (quintic)
        d_traj = self.quintic_polynomial(
            d0, d_dot0, 0,
            d_target, 0, 0,
            T, dt
        )
        
        # Combine
        trajectory = []
        for i, (s, d) in enumerate(zip(s_traj, d_traj)):
            x, y = self.frenet_to_xy(s, d)
            theta = self.get_heading(s)
            v = (s_traj[i][1] if i < len(s_traj)-1 else 0)  # velocity
            
            trajectory.append({
                'x': x,
                'y': y,
                'theta': theta,
                'v': v,
                's': s[0],
                'd': d[0],
                't': i * dt
            })
        
        return trajectory
    
    def quintic_polynomial(self, x0, x0_dot, x0_ddot, x1, x1_dot, x1_ddot, T, dt):
        """Generate quintic polynomial trajectory"""
        # Solve for coefficients
        # x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ])
        
        b = np.array([x0, x0_dot, x0_ddot, x1, x1_dot, x1_ddot])
        
        coeffs = np.linalg.solve(A, b)
        
        # Sample trajectory
        t = np.arange(0, T + dt, dt)
        traj = []
        
        for ti in t:
            x = sum(coeffs[i] * ti**i for i in range(6))
            x_dot = sum(i * coeffs[i] * ti**(i-1) for i in range(1, 6))
            x_ddot = sum(i * (i-1) * coeffs[i] * ti**(i-2) for i in range(2, 6))
            
            traj.append([x, x_dot, x_ddot])
        
        return traj
    
    def compute_cost(self, trajectory, goal, obstacles):
        """Compute trajectory cost"""
        cost = 0
        
        # 1. Goal deviation
        final = trajectory[-1]
        goal_cost = (
            0.1 * (final['s'] - goal[0])**2 +
            1.0 * (final['d'] - goal[1])**2 +
            0.5 * (final['v'] - goal[3])**2
        )
        cost += goal_cost
        
        # 2. Smoothness (jerk)
        jerk_cost = 0
        for i in range(2, len(trajectory)):
            a1 = trajectory[i-1]['v'] - trajectory[i-2]['v']
            a2 = trajectory[i]['v'] - trajectory[i-1]['v']
            jerk = a2 - a1
            jerk_cost += jerk**2
        cost += 0.1 * jerk_cost
        
        # 3. Speed limit
        speed_cost = sum(max(0, t['v'] - 15)**2 for t in trajectory)  # 15 m/s limit
        cost += 10 * speed_cost
        
        # 4. Lateral offset
        offset_cost = sum(t['d']**2 for t in trajectory)
        cost += 0.01 * offset_cost
        
        # 5. Collision proximity
        collision_cost = 0
        for t in trajectory:
            for obs in obstacles:
                dist = self.distance_to_obstacle(t['x'], t['y'], obs)
                if dist < 5:
                    collision_cost += (5 - dist)**2
        cost += 100 * collision_cost
        
        return cost
    
    def xy_to_frenet(self, x, y):
        """Convert (x, y) to Frenet (s, d)"""
        # Find closest point on reference path
        _, idx = self.path_tree.query([x, y])
        
        # Compute s (arc length)
        s = 0
        for i in range(idx):
            s += np.linalg.norm(
                self.reference_path[i+1] - self.reference_path[i]
            )
        
        # Add partial segment
        s += np.linalg.norm(
            np.array([x, y]) - self.reference_path[idx]
        )
        
        # Compute d (lateral offset)
        dx = x - self.reference_path[idx][0]
        dy = y - self.reference_path[idx][1]
        
        # Get normal direction
        if idx < len(self.reference_path) - 1:
            tangent = self.reference_path[idx+1] - self.reference_path[idx]
        else:
            tangent = self.reference_path[idx] - self.reference_path[idx-1]
        
        normal = np.array([-tangent[1], tangent[0]])
        normal = normal / np.linalg.norm(normal)
        
        d = dx * normal[0] + dy * normal[1]
        
        return s, d
```

### Lattice Planning

```python
class LatticePlanner:
    """Lattice-based motion planning"""
    
    def __init__(self):
        self.motion_primitives = self.generate_motion_primitives()
        
    def generate_motion_primitives(self):
        """Generate set of motion primitives"""
        primitives = []
        
        # Different steering angles
        steering_angles = np.linspace(-0.5, 0.5, 11)  # radians
        
        for delta in steering_angles:
            # Different velocities
            for v in [5, 10, 15]:  # m/s
                primitive = self.compute_primitive(delta, v, dt=0.5, T=2.0)
                primitives.append(primitive)
        
        return primitives
    
    def compute_primitive(self, delta, v, dt, T):
        """Compute motion primitive using bicycle model"""
        L = 2.5  # wheelbase
        
        trajectory = []
        x, y, theta = 0, 0, 0
        
        for t in np.arange(0, T, dt):
            # Bicycle model
            x_dot = v * np.cos(theta)
            y_dot = v * np.sin(theta)
            theta_dot = v / L * np.tan(delta)
            
            x += x_dot * dt
            y += y_dot * dt
            theta += theta_dot * dt
            
            trajectory.append((x, y, theta, v))
        
        return {
            'steering': delta,
            'velocity': v,
            'trajectory': trajectory
        }
    
    def plan(self, start, goal, obstacles):
        """Plan using lattice graph"""
        # Build graph
        graph = self.build_lattice_graph(start, goal, obstacles)
        
        # Search
        path = self.a_star_search(graph, start, goal)
        
        return path
    
    def build_lattice_graph(self, start, goal, obstacles):
        """Build lattice graph from motion primitives"""
        import networkx as nx
        
        G = nx.DiGraph()
        G.add_node(start)
        
        # Expand from start
        frontier = [start]
        visited = set()
        
        while frontier:
            current = frontier.pop(0)
            
            if current in visited:
                continue
            visited.add(current)
            
            # Try each motion primitive
            for primitive in self.motion_primitives:
                next_state = self.apply_primitive(current, primitive)
                
                # Check collision
                if not self.collides(next_state, obstacles):
                    G.add_node(next_state)
                    G.add_edge(current, next_state, cost=primitive['cost'])
                    
                    if self.is_close_to_goal(next_state, goal):
                        G.add_node(goal)
                        G.add_edge(next_state, goal, cost=0)
                    else:
                        frontier.append(next_state)
        
        return G
```

### Model Predictive Control (MPC)

```python
class MPCPlanner:
    """Model Predictive Control for trajectory planning"""
    
    def __init__(self, horizon=20, dt=0.1):
        self.horizon = horizon
        self.dt = dt
        self.wheelbase = 2.5
        
        # Weights
        self.w_pos = 1.0
        self.w_vel = 0.1
        self.w_steer = 0.01
        self.w_accel = 0.01
        
    def plan(self, current_state, reference_trajectory, obstacles):
        """
        Args:
            current_state: (x, y, theta, v)
            reference_trajectory: desired path
            obstacles: list of obstacles
        Returns:
            control: (steering, acceleration)
            trajectory: planned trajectory
        """
        # Optimization variables
        n_states = 4  # x, y, theta, v
        n_controls = 2  # steering, acceleration
        
        # Initial guess
        x0 = np.zeros(self.horizon * n_controls)
        
        # Optimize
        result = minimize(
            self.cost_function,
            x0,
            args=(current_state, reference_trajectory, obstacles),
            method='SLSQP',
            bounds=self.get_bounds(),
            constraints=self.get_constraints(current_state)
        )
        
        # Extract controls
        controls = result.x.reshape(self.horizon, n_controls)
        
        # Generate trajectory
        trajectory = self.rollout(current_state, controls)
        
        return controls[0], trajectory
    
    def cost_function(self, controls, current_state, reference, obstacles):
        """MPC cost function"""
        # Reshape controls
        controls = controls.reshape(self.horizon, 2)
        
        # Rollout trajectory
        trajectory = self.rollout(current_state, controls)
        
        cost = 0
        
        # 1. Reference tracking
        for i, state in enumerate(trajectory):
            ref = reference[min(i, len(reference)-1)]
            cost += self.w_pos * ((state[0] - ref[0])**2 + (state[1] - ref[1])**2)
            cost += self.w_vel * (state[3] - ref[3])**2
        
        # 2. Control effort
        for i in range(len(controls) - 1):
            cost += self.w_steer * controls[i][0]**2
            cost += self.w_accel * controls[i][1]**2
        
        # 3. Control smoothness
        for i in range(len(controls) - 1):
            cost += self.w_steer * (controls[i+1][0] - controls[i][0])**2
            cost += self.w_accel * (controls[i+1][1] - controls[i][1])**2
        
        # 4. Obstacle avoidance
        for state in trajectory:
            for obs in obstacles:
                dist = self.distance_to_obstacle(state, obs)
                if dist < 2:
                    cost += 1000 * (2 - dist)**2
        
        return cost
    
    def rollout(self, initial_state, controls):
        """Rollout trajectory from controls"""
        trajectory = [initial_state]
        state = initial_state
        
        for control in controls:
            state = self.dynamics(state, control, self.dt)
            trajectory.append(state)
        
        return trajectory
    
    def dynamics(self, state, control, dt):
        """Bicycle model dynamics"""
        x, y, theta, v = state
        delta, a = control
        
        # Update
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + v / self.wheelbase * np.tan(delta) * dt
        v_new = v + a * dt
        
        return np.array([x_new, y_new, theta_new, v_new])
    
    def get_bounds(self):
        """Control bounds"""
        steering_bounds = [-0.5, 0.5]  # radians
        accel_bounds = [-3, 3]  # m/s^2
        
        bounds = []
        for _ in range(self.horizon):
            bounds.extend([steering_bounds, accel_bounds])
        
        return bounds
```

## Collision Checking

### Geometric Collision Checking

```python
class CollisionChecker:
    """Check collisions with obstacles"""
    
    def __init__(self, vehicle_params):
        self.length = vehicle_params['length']
        self.width = vehicle_params['width']
        
    def check_trajectory(self, trajectory, obstacles):
        """Check if trajectory collides with obstacles"""
        for state in trajectory:
            if self.check_state(state, obstacles):
                return True
        return False
    
    def check_state(self, state, obstacles):
        """Check single state for collision"""
        x, y, theta = state[0], state[1], state[2]
        
        # Get vehicle polygon
        vehicle_poly = self.get_vehicle_polygon(x, y, theta)
        
        for obstacle in obstacles:
            if self.polygons_intersect(vehicle_poly, obstacle):
                return True
        
        return False
    
    def get_vehicle_polygon(self, x, y, theta):
        """Get vehicle corners"""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Corners in body frame
        corners = [
            (-self.length/2, -self.width/2),
            (self.length/2, -self.width/2),
            (self.length/2, self.width/2),
            (-self.length/2, self.width/2)
        ]
        
        # Transform to world frame
        world_corners = []
        for cx, cy in corners:
            wx = x + cx * cos_t - cy * sin_t
            wy = y + cx * sin_t + cy * cos_t
            world_corners.append((wx, wy))
        
        return world_corners
    
    def polygons_intersect(self, poly1, poly2):
        """Check if two polygons intersect using SAT"""
        # Separating Axis Theorem
        for poly in [poly1, poly2]:
            for i in range(len(poly)):
                # Get edge
                edge = (
                    poly[(i+1) % len(poly)][0] - poly[i][0],
                    poly[(i+1) % len(poly)][1] - poly[i][1]
                )
                
                # Get perpendicular axis
                axis = (-edge[1], edge[0])
                
                # Project both polygons
                proj1 = self.project_polygon(poly1, axis)
                proj2 = self.project_polygon(poly2, axis)
                
                # Check for separation
                if proj1[1] < proj2[0] or proj2[1] < proj1[0]:
                    return False
        
        return True
    
    def project_polygon(self, polygon, axis):
        """Project polygon onto axis"""
        projections = []
        for point in polygon:
            proj = point[0] * axis[0] + point[1] * axis[1]
            projections.append(proj)
        
        return (min(projections), max(projections))
```

## Best Practices

### 1. Safety Constraints

```python
class SafetyChecker:
    """Ensure trajectory safety"""
    
    def __init__(self):
        self.min_distance = 1.0  # meters
        self.max_decel = 8.0  # m/s^2
        
    def check_safety(self, trajectory, obstacles, predictions):
        """Check if trajectory is safe"""
        for i, state in enumerate(trajectory):
            t = state['t']
            
            # Check against predicted obstacle positions
            for obs_id, pred_traj in predictions.items():
                obs_state = pred_traj[min(i, len(pred_traj)-1)]
                
                # Compute distance
                dist = np.linalg.norm(
                    state['position'] - obs_state['position']
                )
                
                # Check if too close
                if dist < self.min_distance:
                    return False
                
                # Check if can stop in time
                if not self.can_stop_safely(state, obs_state):
                    return False
        
        return True
    
    def can_stop_safely(self, ego_state, obstacle_state):
        """Check if ego can stop before hitting obstacle"""
        # Distance to obstacle
        dist = np.linalg.norm(
            ego_state['position'] - obstacle_state['position']
        )
        
        # Time to collision
        relative_velocity = ego_state['velocity'] - obstacle_state['velocity']
        ttc = dist / (relative_velocity + 1e-6)
        
        # Stopping distance
        stopping_dist = ego_state['velocity']**2 / (2 * self.max_decel)
        
        return dist > stopping_dist + self.min_distance
```

### 2. Comfort Optimization

```python
class ComfortOptimizer:
    """Optimize trajectory for passenger comfort"""
    
    def __init__(self):
        self.max_accel = 2.0  # m/s^2
        self.max_jerk = 5.0  # m/s^3
        self.max_lateral_accel = 1.5  # m/s^2
        
    def optimize_trajectory(self, trajectory):
        """Smooth trajectory for comfort"""
        # 1. Limit acceleration
        for i in range(1, len(trajectory)):
            dv = trajectory[i]['v'] - trajectory[i-1]['v']
            dt = trajectory[i]['t'] - trajectory[i-1]['t']
            accel = dv / dt
            
            if abs(accel) > self.max_accel:
                # Limit acceleration
                limited_dv = np.sign(accel) * self.max_accel * dt
                trajectory[i]['v'] = trajectory[i-1]['v'] + limited_dv
        
        # 2. Limit jerk
        for i in range(2, len(trajectory)):
            a1 = (trajectory[i-1]['v'] - trajectory[i-2]['v']) / \
                 (trajectory[i-1]['t'] - trajectory[i-2]['t'])
            a2 = (trajectory[i]['v'] - trajectory[i-1]['v']) / \
                 (trajectory[i]['t'] - trajectory[i-1]['t'])
            jerk = (a2 - a1) / (trajectory[i]['t'] - trajectory[i-1]['t'])
            
            if abs(jerk) > self.max_jerk:
                # Smooth jerk
                pass
        
        # 3. Limit lateral acceleration
        for i in range(1, len(trajectory)):
            curvature = self.compute_curvature(
                trajectory[i-1], trajectory[i]
            )
            lateral_accel = trajectory[i]['v']**2 * curvature
            
            if abs(lateral_accel) > self.max_lateral_accel:
                # Reduce speed
                max_v = np.sqrt(self.max_lateral_accel / abs(curvature))
                trajectory[i]['v'] = min(trajectory[i]['v'], max_v)
        
        return trajectory
```

### 3. Real-time Performance

```python
class RealtimePlanner:
    """Ensure real-time planning"""
    
    def __init__(self, max_time=0.1):
        self.max_time = max_time  # 100ms budget
        self.fallback_planner = SimplePlanner()
        
    def plan_with_timeout(self, state, goal, obstacles):
        """Plan with timeout"""
        import time
        start_time = time.time()
        
        # Try complex planner
        try:
            # Set timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError()
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, self.max_time)
            
            trajectory = self.complex_planner.plan(state, goal, obstacles)
            
            signal.setitimer(signal.ITIMER_REAL, 0)  # Cancel timer
            
            return trajectory
            
        except TimeoutError:
            # Fallback to simple planner
            print("Complex planner timeout, using fallback")
            return self.fallback_planner.plan(state, goal, obstacles)
```

## Conclusion

Path planning and decision making are complex challenges that require combining multiple techniques. Tesla's approach emphasizes end-to-end learning while maintaining safety constraints. Key aspects include behavioral planning for high-level decisions, motion planning for trajectory generation, and continuous safety verification.

## References

- "Planning Algorithms" (Steven LaValle)
- "Optimal State Estimation" (Dan Simon)
- Tesla AI Day presentations
- Papers on autonomous driving planning

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~4000 words  
**Size**: ~22KB
