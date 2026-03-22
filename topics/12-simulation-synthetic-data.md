# Simulation and Synthetic Data Generation

## Introduction

Simulation is crucial for training and testing autonomous systems. It allows generating unlimited training data, testing edge cases, and validating safety without real-world risks. This document covers simulation techniques, synthetic data generation, and best practices for sim-to-real transfer.

## Simulation Platforms

### Physics Simulation

```python
import pybullet as p
import numpy as np

class PhysicsSimulator:
    """Physics simulation using PyBullet"""
    
    def __init__(self):
        # Connect to physics engine
        self.physics_client = p.connect(p.DIRECT)
        
        # Set physics parameters
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        
        # Load environment
        self.load_environment()
        
    def load_environment(self):
        """Load simulation environment"""
        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load road
        self.road_id = self.create_road()
        
        # Load buildings
        self.buildings = self.load_buildings()
        
    def create_road(self):
        """Create road geometry"""
        # Create visual and collision shapes
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[100, 10, 0.1]
        )
        
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[100, 10, 0.1]
        )
        
        # Create body
        road_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 0]
        )
        
        return road_id
    
    def spawn_vehicle(self, vehicle_config):
        """Spawn vehicle in simulation"""
        vehicle_id = p.loadURDF(
            vehicle_config['urdf_path'],
            basePosition=vehicle_config['position'],
            baseOrientation=vehicle_config['orientation']
        )
        
        # Set vehicle properties
        for joint_idx in range(p.getNumJoints(vehicle_id)):
            joint_info = p.getJointInfo(vehicle_id, joint_idx)
            
            if joint_info[1].decode('utf-8') == 'steering':
                self.steer_joint = joint_idx
            elif joint_info[1].decode('utf-8') == 'wheel_fl':
                self.wheel_joints.append(joint_idx)
        
        return vehicle_id
    
    def step(self, action):
        """Step simulation with action"""
        # Apply action
        self.apply_control(action)
        
        # Step physics
        p.stepSimulation()
        
        # Get observation
        observation = self.get_observation()
        
        # Check termination
        done = self.check_termination()
        
        # Compute reward
        reward = self.compute_reward()
        
        return observation, reward, done
    
    def get_observation(self):
        """Get sensor observations"""
        observation = {}
        
        # Camera
        observation['camera'] = self.render_camera()
        
        # IMU
        observation['imu'] = self.get_imu_data()
        
        # Vehicle state
        observation['velocity'] = self.get_velocity()
        observation['position'] = self.get_position()
        
        return observation
```

### Sensor Simulation

```python
class SensorSimulator:
    """Simulate various sensors"""
    
    def __init__(self, physics_client):
        self.client = physics_client
        
    def simulate_camera(self, camera_params):
        """Simulate camera sensor"""
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_params['position'],
            cameraTargetPosition=camera_params['target'],
            cameraUpVector=camera_params['up']
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=camera_params['fov'],
            aspect=camera_params['aspect'],
            nearVal=camera_params['near'],
            farVal=camera_params['far']
        )
        
        # Render
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=camera_params['width'],
            height=camera_params['height'],
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix
        )
        
        return {
            'rgb': np.array(rgb_img).reshape(height, width, 4)[:, :, :3],
            'depth': np.array(depth_img).reshape(height, width),
            'segmentation': np.array(seg_img).reshape(height, width)
        }
    
    def simulate_lidar(self, lidar_params):
        """Simulate LiDAR sensor"""
        points = []
        
        # Ray casting
        for theta in np.linspace(0, 2*np.pi, lidar_params['horizontal_resolution']):
            for phi in np.linspace(
                -lidar_params['vertical_fov']/2,
                lidar_params['vertical_fov']/2,
                lidar_params['vertical_resolution']
            ):
                # Ray direction
                direction = np.array([
                    np.cos(theta) * np.cos(phi),
                    np.sin(theta) * np.cos(phi),
                    np.sin(phi)
                ])
                
                # Cast ray
                ray_start = lidar_params['position']
                ray_end = ray_start + direction * lidar_params['max_range']
                
                hit = p.rayTest(ray_start, ray_end)
                
                if hit[0][0] != -1:
                    # Hit point
                    hit_fraction = hit[0][2]
                    point = ray_start + direction * hit_fraction * lidar_params['max_range']
                    points.append(point)
        
        return np.array(points)
    
    def simulate_radar(self, radar_params):
        """Simulate radar sensor"""
        detections = []
        
        # Simple radar simulation
        for obj in self.get_objects_in_range(radar_params['range']):
            # Range
            distance = np.linalg.norm(obj.position - radar_params['position'])
            
            # Angle
            relative_pos = obj.position - radar_params['position']
            angle = np.arctan2(relative_pos[1], relative_pos[0])
            
            # Velocity (Doppler)
            relative_vel = obj.velocity - radar_params['velocity']
            radial_vel = np.dot(relative_vel, relative_pos / distance)
            
            # RCS (simplified)
            rcs = self.compute_rcs(obj, angle)
            
            if distance < radar_params['range'] and abs(angle) < radar_params['fov']/2:
                detections.append({
                    'range': distance,
                    'angle': angle,
                    'velocity': radial_vel,
                    'rcs': rcs
                })
        
        return detections
```

## Scene Generation

### Procedural Generation

```python
class SceneGenerator:
    """Procedurally generate driving scenes"""
    
    def __init__(self):
        self.road_generator = RoadGenerator()
        self.object_placer = ObjectPlacer()
        self.weather_system = WeatherSystem()
        
    def generate_scene(self, seed=None):
        """Generate random driving scene"""
        if seed is not None:
            np.random.seed(seed)
        
        scene = {
            'road': None,
            'objects': [],
            'weather': None,
            'lighting': None
        }
        
        # Generate road network
        scene['road'] = self.road_generator.generate()
        
        # Place objects
        scene['objects'] = self.object_placer.place(scene['road'])
        
        # Set weather
        scene['weather'] = self.weather_system.sample()
        
        # Set lighting
        scene['lighting'] = self.sample_lighting()
        
        return scene
    
    def sample_lighting(self):
        """Sample lighting conditions"""
        # Time of day
        hour = np.random.uniform(6, 20)  # 6am to 8pm
        
        # Sun angle
        sun_elevation = 90 * np.sin((hour - 6) / 14 * np.pi)
        sun_azimuth = np.random.uniform(0, 360)
        
        # Cloud cover
        cloud_cover = np.random.uniform(0, 1)
        
        return {
            'hour': hour,
            'sun_elevation': sun_elevation,
            'sun_azimuth': sun_azimuth,
            'cloud_cover': cloud_cover
        }


class RoadGenerator:
    """Generate road networks"""
    
    def __init__(self):
        self.road_types = ['highway', 'urban', 'residential', 'rural']
        
    def generate(self, road_type=None):
        """Generate road network"""
        if road_type is None:
            road_type = np.random.choice(self.road_types)
        
        if road_type == 'highway':
            return self.generate_highway()
        elif road_type == 'urban':
            return self.generate_urban()
        else:
            return self.generate_default()
    
    def generate_highway(self):
        """Generate highway road"""
        road = {
            'type': 'highway',
            'lanes': [],
            'exits': [],
            'signs': []
        }
        
        # Generate lanes
        num_lanes = np.random.randint(3, 6)
        lane_width = 3.5  # meters
        
        for i in range(num_lanes):
            lane = {
                'id': i,
                'centerline': self.generate_centerline(),
                'width': lane_width,
                'speed_limit': np.random.choice([100, 120, 130])  # km/h
            }
            road['lanes'].append(lane)
        
        return road
    
    def generate_centerline(self):
        """Generate lane centerline"""
        # Generate waypoints
        num_points = 100
        x = np.linspace(0, 1000, num_points)
        
        # Add curvature
        curvature = np.random.uniform(-0.01, 0.01)
        y = curvature * x**2 / 2
        
        points = np.column_stack([x, y])
        
        return points
```

### Traffic Generation

```python
class TrafficGenerator:
    """Generate traffic agents"""
    
    def __init__(self, simulator):
        self.simulator = simulator
        
    def generate_traffic(self, scene, density=0.5):
        """Generate traffic agents"""
        agents = []
        
        # Determine number of vehicles
        road_length = self.compute_road_length(scene['road'])
        num_vehicles = int(road_length * density * 0.01)
        
        for _ in range(num_vehicles):
            agent = self.create_agent(scene)
            agents.append(agent)
        
        return agents
    
    def create_agent(self, scene):
        """Create single traffic agent"""
        # Choose lane
        lane = np.random.choice(scene['road']['lanes'])
        
        # Sample position along lane
        t = np.random.uniform(0, 1)
        position = self.sample_position_on_lane(lane, t)
        
        # Choose vehicle type
        vehicle_type = np.random.choice(
            ['car', 'truck', 'bus', 'motorcycle'],
            p=[0.7, 0.15, 0.1, 0.05]
        )
        
        # Sample velocity
        velocity = np.random.normal(
            lane['speed_limit'] * 0.9,
            lane['speed_limit'] * 0.1
        )
        
        agent = {
            'type': vehicle_type,
            'position': position,
            'velocity': velocity,
            'lane_id': lane['id'],
            'behavior': self.sample_behavior()
        }
        
        return agent
    
    def sample_behavior(self):
        """Sample agent behavior profile"""
        behaviors = {
            'cautious': {
                'speed_factor': 0.8,
                'lane_change_freq': 0.1,
                'aggression': 0.2
            },
            'normal': {
                'speed_factor': 1.0,
                'lane_change_freq': 0.3,
                'aggression': 0.5
            },
            'aggressive': {
                'speed_factor': 1.2,
                'lane_change_freq': 0.6,
                'aggression': 0.8
            }
        }
        
        behavior_type = np.random.choice(
            ['cautious', 'normal', 'aggressive'],
            p=[0.2, 0.6, 0.2]
        )
        
        return behaviors[behavior_type]
```

## Domain Randomization

### Visual Randomization

```python
class VisualRandomizer:
    """Randomize visual appearance"""
    
    def __init__(self):
        self.params = {
            'brightness': (0.5, 1.5),
            'contrast': (0.5, 1.5),
            'saturation': (0.5, 1.5),
            'hue': (-0.1, 0.1),
            'noise': (0.0, 0.1),
            'blur': (0, 3)
        }
        
    def randomize(self, image):
        """Apply visual randomization"""
        # Sample parameters
        brightness = np.random.uniform(*self.params['brightness'])
        contrast = np.random.uniform(*self.params['contrast'])
        saturation = np.random.uniform(*self.params['saturation'])
        hue = np.random.uniform(*self.params['hue'])
        noise_std = np.random.uniform(*self.params['noise'])
        blur_sigma = np.random.uniform(*self.params['blur'])
        
        # Apply transformations
        image = self.adjust_brightness(image, brightness)
        image = self.adjust_contrast(image, contrast)
        image = self.adjust_saturation(image, saturation)
        image = self.adjust_hue(image, hue)
        
        if noise_std > 0:
            image = self.add_noise(image, noise_std)
        
        if blur_sigma > 0:
            image = self.add_blur(image, blur_sigma)
        
        return image
    
    def adjust_brightness(self, image, factor):
        """Adjust brightness"""
        return np.clip(image * factor, 0, 255)
    
    def adjust_contrast(self, image, factor):
        """Adjust contrast"""
        mean = image.mean()
        return np.clip((image - mean) * factor + mean, 0, 255)
```

### Physics Randomization

```python
class PhysicsRandomizer:
    """Randomize physics parameters"""
    
    def __init__(self):
        self.params = {
            'mass': (0.8, 1.2),
            'friction': (0.5, 1.5),
            'restitution': (0.0, 0.3),
            'gravity': (9.5, 10.0),
            'air_resistance': (0.0, 0.1)
        }
        
    def randomize(self, simulator):
        """Randomize physics in simulator"""
        # Randomize gravity
        gravity = np.random.uniform(*self.params['gravity'])
        p.setGravity(0, 0, -gravity)
        
        # Randomize vehicle properties
        for vehicle_id in simulator.vehicles:
            # Mass
            mass_factor = np.random.uniform(*self.params['mass'])
            self.adjust_mass(vehicle_id, mass_factor)
            
            # Friction
            friction = np.random.uniform(*self.params['friction'])
            self.adjust_friction(vehicle_id, friction)
    
    def adjust_mass(self, body_id, factor):
        """Adjust body mass"""
        dynamics = p.getDynamicsInfo(body_id, -1)
        original_mass = dynamics[0]
        new_mass = original_mass * factor
        
        p.changeDynamics(
            body_id,
            -1,
            mass=new_mass
        )
    
    def adjust_friction(self, body_id, friction):
        """Adjust friction coefficient"""
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=friction
        )
```

## Scenario Generation

### Edge Case Generation

```python
class EdgeCaseGenerator:
    """Generate rare and challenging scenarios"""
    
    def __init__(self):
        self.scenarios = [
            'jaywalking_pedestrian',
            'emergency_vehicle',
            'construction_zone',
            'weather_extreme',
            'sensor_failure',
            'adversarial_agent'
        ]
        
    def generate(self, scenario_type=None):
        """Generate specific edge case"""
        if scenario_type is None:
            scenario_type = np.random.choice(self.scenarios)
        
        generator = getattr(self, f'generate_{scenario_type}')
        return generator()
    
    def generate_jaywalking_pedestrian(self):
        """Generate jaywalking pedestrian scenario"""
        scenario = {
            'type': 'jaywalking_pedestrian',
            'description': 'Pedestrian crosses unexpectedly',
            'difficulty': 'hard',
            'agents': []
        }
        
        # Ego vehicle
        scenario['agents'].append({
            'type': 'ego',
            'position': [0, 0, 0],
            'velocity': 15  # m/s
        })
        
        # Jaywalking pedestrian
        # Time to appear: critical (2-3 seconds)
        time_to_cross = np.random.uniform(2, 3)
        cross_position = 15 * time_to_cross  # Distance ahead
        
        scenario['agents'].append({
            'type': 'pedestrian',
            'position': [cross_position, -5, 0],
            'velocity': [0, 2, 0],  # Crossing
            'crossing_time': time_to_cross
        })
        
        return scenario
    
    def generate_emergency_vehicle(self):
        """Generate emergency vehicle scenario"""
        scenario = {
            'type': 'emergency_vehicle',
            'description': 'Emergency vehicle approaches from behind',
            'difficulty': 'medium',
            'agents': []
        }
        
        # Ego vehicle
        scenario['agents'].append({
            'type': 'ego',
            'position': [0, 0, 0],
            'velocity': 15,
            'lane': 1
        })
        
        # Emergency vehicle
        scenario['agents'].append({
            'type': 'emergency_vehicle',
            'position': [-30, 0, 0],  # Behind
            'velocity': 25,
            'lane': 1,
            'sirens': True
        })
        
        return scenario
```

### Adversarial Scenarios

```python
class AdversarialScenarioGenerator:
    """Generate adversarial test scenarios"""
    
    def __init__(self):
        self.adversarial_agents = []
        
    def generate_cut_in(self, ego_speed):
        """Generate aggressive cut-in scenario"""
        scenario = {
            'type': 'adversarial_cut_in',
            'risk_level': 'high'
        }
        
        # Calculate timing for dangerous cut-in
        # Cut-in should happen just before ego arrives
        cut_in_distance = ego_speed * 2  # 2 seconds ahead
        
        scenario['agents'] = [
            {
                'type': 'ego',
                'position': [0, 0, 0],
                'velocity': ego_speed
            },
            {
                'type': 'adversarial',
                'position': [cut_in_distance, 3.5, 0],  # Adjacent lane
                'velocity': ego_speed * 0.7,
                'behavior': {
                    'action': 'cut_in',
                    'timing': 'critical',
                    'aggression': 1.0
                }
            }
        ]
        
        return scenario
    
    def generate_multi_agent_challenge(self):
        """Generate complex multi-agent scenario"""
        scenario = {
            'type': 'multi_agent_challenge',
            'description': 'Multiple agents creating complex situation'
        }
        
        agents = [
            # Ego
            {'type': 'ego', 'position': [0, 0, 0], 'velocity': 15},
            
            # Lead vehicle braking
            {'type': 'vehicle', 'position': [20, 0, 0], 'velocity': 12,
             'behavior': 'braking', 'deceleration': 5},
            
            # Vehicle merging from right
            {'type': 'vehicle', 'position': [15, 3.5, 0], 'velocity': 10,
             'behavior': 'merging'},
            
            # Pedestrian on shoulder
            {'type': 'pedestrian', 'position': [25, -4, 0], 'velocity': [1, 0, 0]}
        ]
        
        scenario['agents'] = agents
        
        return scenario
```

## Sim-to-Real Transfer

### Domain Adaptation

```python
class DomainAdaptation:
    """Adapt simulation-trained models to real world"""
    
    def __init__(self):
        self.feature_adaptor = FeatureAdaptor()
        self.style_transfer = StyleTransfer()
        
    def adapt(self, sim_model, real_data):
        """Adapt simulation model to real domain"""
        # 1. Feature-level adaptation
        adapted_features = self.feature_adaptor.adapt(real_data)
        
        # 2. Fine-tune on real data
        adapted_model = self.fine_tune(sim_model, adapted_features)
        
        return adapted_model
    
    def fine_tune(self, model, real_data):
        """Fine-tune on real data"""
        # Lower learning rate
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-5  # Lower than pre-training
        )
        
        for epoch in range(10):
            for batch in real_data:
                # Forward
                output = model(batch['image'])
                
                # Loss
                loss = F.cross_entropy(output, batch['label'])
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return model


class FeatureAdaptor:
    """Adapt features between domains"""
    
    def __init__(self):
        self.adaptor_net = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1)
        )
        
    def adapt(self, features):
        """Adapt features to target domain"""
        return self.adaptor_net(features)
```

### Progressive Transfer

```python
class ProgressiveTransfer:
    """Progressively transfer from sim to real"""
    
    def __init__(self, sim_env, real_env):
        self.sim_env = sim_env
        self.real_env = real_env
        self.mix_ratio = 0.0  # Start with all sim
        
    def train_progressive(self, agent, total_steps):
        """Train with progressive mixing"""
        for step in range(total_steps):
            # Update mix ratio
            self.mix_ratio = step / total_steps
            
            # Sample from sim or real
            if np.random.random() < 1 - self.mix_ratio:
                experience = self.sim_env.step(agent.select_action())
            else:
                experience = self.real_env.step(agent.select_action())
            
            # Update agent
            agent.update(experience)
        
        return agent
```

## Best Practices

### 1. Reality Gap Measurement

```python
class RealityGapMeasurer:
    """Measure gap between sim and real"""
    
    def __init__(self):
        self.metrics = {}
        
    def measure_gap(self, sim_performance, real_performance):
        """Measure performance gap"""
        gap = {}
        
        for metric in sim_performance:
            gap[metric] = real_performance[metric] - sim_performance[metric]
            
            print(f"{metric}:")
            print(f"  Sim: {sim_performance[metric]:.3f}")
            print(f"  Real: {real_performance[metric]:.3f}")
            print(f"  Gap: {gap[metric]:.3f}")
        
        return gap
```

### 2. Validation in Simulation

```python
class SimValidator:
    """Validate models in simulation before deployment"""
    
    def __init__(self, sim_env, test_scenarios):
        self.sim_env = sim_env
        self.test_scenarios = test_scenarios
        
    def validate(self, model):
        """Validate model in simulation"""
        results = {
            'passed': 0,
            'failed': 0,
            'scenarios': []
        }
        
        for scenario in self.test_scenarios:
            # Setup scenario
            self.sim_env.load_scenario(scenario)
            
            # Run episode
            success = self.run_episode(model)
            
            if success:
                results['passed'] += 1
            else:
                results['failed'] += 1
            
            results['scenarios'].append({
                'name': scenario['name'],
                'success': success
            })
        
        # Compute pass rate
        results['pass_rate'] = results['passed'] / len(self.test_scenarios)
        
        return results
```

## Conclusion

Simulation is a powerful tool for developing autonomous systems. By combining physics simulation, procedural generation, domain randomization, and careful sim-to-real transfer, we can efficiently train and test systems that work reliably in the real world.

## References

- "Domain Randomization for Sim-to-Real Transfer" (Tobin et al., 2017)
- "Sim-to-Real Transfer of Robotic Control" (Peng et al., 2018)
- NVIDIA Drive Sim documentation
- CARLA Simulator papers

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~3500 words  
**Size**: ~20KB
