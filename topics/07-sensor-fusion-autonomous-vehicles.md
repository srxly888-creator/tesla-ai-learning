# Sensor Fusion for Autonomous Vehicles

## Introduction

Sensor fusion combines data from multiple sensors to create a more accurate and robust understanding of the environment than any single sensor could provide. This document covers the techniques, architectures, and best practices for sensor fusion in autonomous vehicles.

## Sensor Types and Characteristics

### Sensor Comparison

```
┌──────────────────────────────────────────────────────────────┐
│                  Sensor Comparison Matrix                     │
├──────────────┬──────────┬──────────┬──────────┬─────────────┤
│ Sensor Type  │ Range    │ Weather  │ Detail   │ Cost        │
├──────────────┼──────────┼──────────┼──────────┼─────────────┤
│ Camera       │ Medium   │ Poor     │ High     │ Low         │
│ Radar        │ Long     │ Good     │ Low      │ Medium      │
│ LiDAR        │ Medium   │ Medium   │ High     │ High        │
│ Ultrasonic   │ Short    │ Good     │ Low      │ Very Low    │
│ IMU          │ N/A      │ Good     │ High     │ Low         │
│ GPS          │ Global   │ Good     │ Low      │ Low         │
└──────────────┴──────────┴──────────┴──────────┴─────────────┘
```

### Tesla's Vision-Centric Approach

Tesla primarily uses cameras with minimal radar:

```python
class TeslaSensorSuite:
    """Tesla's sensor configuration"""
    
    def __init__(self):
        # Primary sensors
        self.cameras = {
            'front_main': Camera(fov=50, range=150),
            'front_wide': Camera(fov=150, range=60),
            'front_narrow': Camera(fov=25, range=250),
            'left_fwd': Camera(fov=90, range=80),
            'right_fwd': Camera(fov=90, range=80),
            'left_rear': Camera(fov=90, range=80),
            'right_rear': Camera(fov=90, range=80),
            'rear': Camera(fov=90, range=80)
        }
        
        # Supplementary sensors
        self.radar = ForwardRadar(range=160)  # Optional, being phased out
        self.ultrasonics = [Ultrasonic() for _ in range(12)]
        
        # Localization
        self.imu = IMU()
        self.gps = GPS()
```

## Early vs Late Fusion

### Early Fusion

Combine raw sensor data before processing:

```python
class EarlyFusion(nn.Module):
    """Fuse raw sensor data early in the pipeline"""
    
    def __init__(self):
        super().__init__()
        
        # Process camera and radar together
        self.camera_encoder = nn.Conv2d(3, 64, 7, stride=2)
        self.radar_encoder = nn.Conv2d(5, 64, 7, stride=2)  # r, θ, v, σ, i
        
        # Fusion
        self.fusion_conv = nn.Conv2d(128, 256, 3, padding=1)
        
        # Shared backbone
        self.backbone = ResNet50()
        
    def forward(self, camera, radar):
        """
        Args:
            camera: [B, 3, H, W] RGB image
            radar: [B, 5, H, W] radar range, angle, velocity, etc.
        Returns:
            features: [B, 256, H', W'] fused features
        """
        # Encode each modality
        cam_feat = self.camera_encoder(camera)
        rad_feat = self.radar_encoder(radar)
        
        # Concatenate
        concat = torch.cat([cam_feat, rad_feat], dim=1)
        
        # Fuse
        fused = self.fusion_conv(concat)
        
        # Process together
        output = self.backbone(fused)
        
        return output
```

### Late Fusion

Combine processed outputs from each sensor:

```python
class LateFusion(nn.Module):
    """Fuse processed sensor outputs"""
    
    def __init__(self):
        super().__init__()
        
        # Independent processing for each sensor
        self.camera_detector = CameraDetector()
        self.radar_detector = RadarDetector()
        
        # Fusion module
        self.fusion = ObjectFusion()
        
    def forward(self, camera, radar):
        """
        Args:
            camera: camera image
            radar: radar data
        Returns:
            fused_objects: combined object list
        """
        # Detect independently
        cam_objects = self.camera_detector(camera)
        rad_objects = self.radar_detector(radar)
        
        # Fuse detections
        fused_objects = self.fusion(cam_objects, rad_objects)
        
        return fused_objects


class ObjectFusion:
    """Fuse object detections from multiple sensors"""
    
    def __init__(self):
        self.matcher = HungarianMatcher()
        
    def fuse(self, objects_a, objects_b):
        """Fuse two sets of object detections"""
        # Match objects between sensors
        matches = self.matcher.match(
            objects_a, objects_b,
            cost_fn=self.matching_cost
        )
        
        fused = []
        
        for (a_idx, b_idx) in matches:
            obj_a = objects_a[a_idx]
            obj_b = objects_b[b_idx]
            
            # Weighted average based on confidence
            w_a = obj_a.confidence
            w_b = obj_b.confidence
            
            fused_obj = Object()
            fused_obj.position = (w_a * obj_a.position + w_b * obj_b.position) / (w_a + w_b)
            fused_obj.velocity = (w_a * obj_a.velocity + w_b * obj_b.velocity) / (w_a + w_b)
            fused_obj.confidence = max(w_a, w_b)
            
            fused.append(fused_obj)
        
        # Add unmatched objects
        for obj in self.get_unmatched(objects_a, objects_b, matches):
            fused.append(obj)
        
        return fused
    
    def matching_cost(self, obj_a, obj_b):
        """Cost for matching two objects"""
        # Position distance
        pos_dist = np.linalg.norm(obj_a.position - obj_b.position)
        
        # Class agreement
        class_cost = 0 if obj_a.class_id == obj_b.class_id else 1
        
        # Size similarity
        size_cost = np.abs(obj_a.size - obj_b.size).mean()
        
        return pos_dist + 0.5 * class_cost + 0.3 * size_cost
```

### Mid-Level Fusion

Fuse at intermediate representation level:

```python
class MidLevelFusion(nn.Module):
    """Fuse at feature level"""
    
    def __init__(self):
        super().__init__()
        
        # Extract features from each sensor
        self.camera_backbone = ResNet50()
        self.radar_backbone = RadarEncoder()
        
        # Feature fusion
        self.feature_fusion = FeatureFusionModule(
            camera_dim=2048,
            radar_dim=512,
            fused_dim=256
        )
        
        # Task heads
        self.detection_head = DetectionHead(256)
        self.segmentation_head = SegmentationHead(256)
        
    def forward(self, camera, radar):
        # Extract features
        cam_feat = self.camera_backbone(camera)
        rad_feat = self.radar_backbone(radar)
        
        # Fuse features
        fused_feat = self.feature_fusion(cam_feat, rad_feat)
        
        # Apply task heads
        detections = self.detection_head(fused_feat)
        segmentation = self.segmentation_head(fused_feat)
        
        return detections, segmentation


class FeatureFusionModule(nn.Module):
    """Cross-attention based feature fusion"""
    
    def __init__(self, camera_dim, radar_dim, fused_dim):
        super().__init__()
        
        # Project to common dimension
        self.cam_proj = nn.Linear(camera_dim, fused_dim)
        self.rad_proj = nn.Linear(radar_dim, fused_dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=8
        )
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=8
        )
        
        # Output projection
        self.output_proj = nn.Linear(fused_dim * 2, fused_dim)
        
    def forward(self, cam_feat, rad_feat):
        # Project to common dimension
        cam_feat = self.cam_proj(cam_feat)
        rad_feat = self.rad_proj(rad_feat)
        
        # Cross-attention: camera attending to radar
        cross_out, _ = self.cross_attn(
            query=cam_feat,
            key=rad_feat,
            value=rad_feat
        )
        
        # Self-attention on fused features
        self_out, _ = self.self_attn(
            query=cross_out,
            key=cross_out,
            value=cross_out
        )
        
        # Combine
        fused = torch.cat([cam_feat, self_out], dim=-1)
        output = self.output_proj(fused)
        
        return output
```

## Radar-Camera Fusion

### Radar Point Cloud Processing

```python
class RadarProcessor:
    """Process radar data"""
    
    def __init__(self):
        self.radar_config = {
            'max_range': 160,  # meters
            'range_resolution': 0.5,
            'angle_range': 60,  # degrees
            'angle_resolution': 0.5,
            'velocity_range': 50,  # m/s
            'velocity_resolution': 0.5
        }
        
    def process(self, radar_data):
        """
        Args:
            radar_data: raw radar returns
        Returns:
            radar_points: [N, 5] (range, angle, velocity, rcs, snr)
        """
        # Parse radar returns
        points = self.parse_returns(radar_data)
        
        # Cluster points
        clusters = self.cluster_points(points)
        
        # Extract objects
        objects = []
        for cluster in clusters:
            obj = RadarObject()
            obj.position = self.compute_position(cluster)
            obj.velocity = self.compute_velocity(cluster)
            obj.rcs = cluster.mean_rcs
            obj.confidence = cluster.snr
            objects.append(obj)
        
        return objects
    
    def project_to_image(self, radar_objects, camera_calib):
        """Project radar points to camera image"""
        projected = []
        
        for obj in radar_objects:
            # Transform to camera frame
            pos_cam = camera_calib.extrinsic @ obj.position
            
            # Project to image
            if pos_cam[2] > 0:  # In front of camera
                u = pos_cam[0] * camera_calib.fx / pos_cam[2] + camera_calib.cx
                v = pos_cam[1] * camera_calib.fy / pos_cam[2] + camera_calib.cy
                
                projected.append({
                    'u': u,
                    'v': v,
                    'depth': pos_cam[2],
                    'velocity': obj.velocity,
                    'rcs': obj.rcs
                })
        
        return projected
```

### Radar-Camera Fusion Architecture

```python
class RadarCameraFusion(nn.Module):
    """Fuse radar and camera for object detection"""
    
    def __init__(self):
        super().__init__()
        
        # Camera branch
        self.camera_backbone = ResNet50()
        self.camera_neck = FPN()
        
        # Radar branch
        self.radar_encoder = RadarEncoder()
        
        # Fusion
        self.fusion = nn.ModuleDict({
            'attention': CrossAttention(256, 128),
            'fusion_conv': nn.Conv2d(256 + 128, 256, 1)
        })
        
        # Detection head
        self.detection_head = CenterPointHead(256)
        
    def forward(self, camera, radar):
        """
        Args:
            camera: [B, 3, H, W]
            radar: [B, N, 5] (x, y, z, velocity, rcs)
        Returns:
            detections: object detections
        """
        # Process camera
        cam_feat = self.camera_backbone(camera)
        cam_feat = self.camera_neck(cam_feat)
        
        # Process radar
        # Project radar to BEV
        radar_bev = self.project_radar_to_bev(radar, cam_feat.shape)
        rad_feat = self.radar_encoder(radar_bev)
        
        # Cross-attention fusion
        fused = self.fusion['attention'](cam_feat, rad_feat)
        fused = torch.cat([cam_feat, fused], dim=1)
        fused = self.fusion['fusion_conv'](fused)
        
        # Detect
        detections = self.detection_head(fused)
        
        return detections
```

## IMU and Odometry Fusion

### Extended Kalman Filter

```python
class ExtendedKalmanFilter:
    """EKF for sensor fusion"""
    
    def __init__(self):
        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.state = np.zeros(9)
        self.covariance = np.eye(9) * 100
        
        # Process noise
        self.Q = np.diag([0.1, 0.1, 0.1,  # position
                          0.5, 0.5, 0.5,  # velocity
                          0.01, 0.01, 0.01])  # orientation
        
        # Measurement noise
        self.R_imu = np.diag([0.1, 0.1, 0.1])  # acceleration
        self.R_gps = np.diag([5.0, 5.0, 5.0])  # position
        
    def predict(self, imu_data, dt):
        """Prediction step using IMU"""
        # Extract IMU data
        accel = imu_data['acceleration']
        gyro = imu_data['gyroscope']
        
        # Predict state
        # Position = Position + Velocity * dt
        self.state[0:3] += self.state[3:6] * dt
        
        # Velocity = Velocity + Acceleration * dt
        self.state[3:6] += accel * dt
        
        # Orientation (simplified)
        self.state[6:9] += gyro * dt
        
        # Predict covariance
        F = self.compute_jacobian(dt)
        self.covariance = F @ self.covariance @ F.T + self.Q * dt
        
    def update_gps(self, gps_position):
        """Update with GPS measurement"""
        # Measurement model
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)
        
        # Innovation
        y = gps_position - self.state[0:3]
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R_gps
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state += K @ y
        
        # Update covariance
        self.covariance = (np.eye(9) - K @ H) @ self.covariance
        
    def compute_jacobian(self, dt):
        """Compute Jacobian of state transition"""
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        return F
```

### Visual-Inertial Odometry

```python
class VisualInertialOdometry:
    """Fuse camera and IMU for odometry"""
    
    def __init__(self):
        self.imu_buffer = []
        self.feature_tracker = FeatureTracker()
        self.optimizer = BundleAdjustment()
        
    def process(self, camera_frame, imu_data):
        """
        Args:
            camera_frame: current camera image
            imu_data: IMU measurements since last frame
        Returns:
            pose: current vehicle pose
        """
        # Track features
        features = self.feature_tracker.track(camera_frame)
        
        # Preintegrate IMU
        imu_delta = self.preintegrate_imu(imu_data)
        
        # Predict pose from IMU
        predicted_pose = self.last_pose @ imu_delta
        
        # Initialize features in 3D
        if len(self.features_3d) < 100:
            self.triangulate_features(features, predicted_pose)
        
        # Optimize pose and features
        optimized_pose, self.features_3d = self.optimizer.optimize(
            features,
            self.features_3d,
            predicted_pose
        )
        
        self.last_pose = optimized_pose
        
        return optimized_pose
    
    def preintegrate_imu(self, imu_data):
        """Preintegrate IMU measurements"""
        delta_rotation = Rotation.identity()
        delta_velocity = np.zeros(3)
        delta_position = np.zeros(3)
        
        for meas in imu_data:
            dt = meas['dt']
            accel = meas['acceleration']
            gyro = meas['gyroscope']
            
            # Integrate rotation
            delta_rotation *= Rotation.from_rotvec(gyro * dt)
            
            # Integrate velocity
            delta_velocity += delta_rotation @ accel * dt
            
            # Integrate position
            delta_position += delta_velocity * dt
        
        return Transform(delta_rotation, delta_position)
```

## Multi-Object Tracking

### Tracking-by-Detection

```python
class MultiObjectTracker:
    """Track objects across frames"""
    
    def __init__(self):
        self.tracks = []
        self.next_id = 0
        self.max_age = 10
        self.min_hits = 3
        
    def update(self, detections):
        """
        Args:
            detections: list of current frame detections
        Returns:
            tracks: list of active tracks
        """
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        matches, unmatched_dets, unmatched_trks = self.match(
            detections, self.tracks
        )
        
        # Update matched tracks
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(detections[det_idx], self.next_id)
            self.next_id += 1
            self.tracks.append(new_track)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.age < self.max_age]
        
        # Return confirmed tracks
        return [t for t in self.tracks if t.hits >= self.min_hits]
    
    def match(self, detections, tracks):
        """Match detections to tracks using Hungarian algorithm"""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Compute cost matrix
        cost = np.zeros((len(detections), len(tracks)))
        for i, det in enumerate(detections):
            for j, trk in enumerate(tracks):
                cost[i, j] = self.compute_cost(det, trk)
        
        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        det_indices, trk_indices = linear_sum_assignment(cost)
        
        # Filter by threshold
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))
        
        for d, t in zip(det_indices, trk_indices):
            if cost[d, t] < 1.0:  # Threshold
                matches.append((d, t))
                unmatched_dets.remove(d)
                unmatched_trks.remove(t)
        
        return matches, unmatched_dets, unmatched_trks
    
    def compute_cost(self, detection, track):
        """Cost for matching detection to track"""
        # Position distance
        pos_dist = np.linalg.norm(
            detection.position - track.position
        )
        
        # Size similarity
        size_dist = np.abs(detection.size - track.size).mean()
        
        # Class agreement
        class_cost = 0 if detection.class_id == track.class_id else 1
        
        return pos_dist + 0.5 * size_dist + class_cost


class Track:
    """Single object track"""
    
    def __init__(self, detection, track_id):
        self.id = track_id
        self.position = detection.position
        self.velocity = np.zeros(3)
        self.size = detection.size
        self.class_id = detection.class_id
        
        self.hits = 1
        self.age = 0
        
        # Kalman filter
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.x = np.concatenate([self.position, self.velocity])
        
    def predict(self):
        """Predict next state"""
        self.kf.predict()
        self.position = self.kf.x[0:3]
        self.velocity = self.kf.x[3:6]
        self.age += 1
        
    def update(self, detection):
        """Update with new detection"""
        self.kf.update(detection.position)
        self.position = self.kf.x[0:3]
        self.velocity = self.kf.x[3:6]
        self.size = detection.size
        self.class_id = detection.class_id
        self.hits += 1
        self.age = 0
```

## Occupancy Grid

### Grid-Based Fusion

```python
class OccupancyGrid:
    """Grid-based representation of environment"""
    
    def __init__(self, resolution=0.1, size=100):
        self.resolution = resolution  # meters per cell
        self.size = size  # grid size in meters
        self.grid_size = int(size / resolution)
        
        # Occupancy grid (log odds)
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # Prior
        self.prior = 0.5
        
    def update(self, sensor_data, pose):
        """Update grid with new sensor data"""
        for measurement in sensor_data:
            # Get cell indices
            x_idx, y_idx = self.world_to_grid(measurement.position)
            
            if self.in_grid(x_idx, y_idx):
                # Update log odds
                log_odds = self.log_odds(measurement.occupied_prob)
                self.grid[x_idx, y_idx] += log_odds - self.log_odds(self.prior)
    
    def world_to_grid(self, position):
        """Convert world coordinates to grid indices"""
        x_idx = int((position[0] + self.size/2) / self.resolution)
        y_idx = int((position[1] + self.size/2) / self.resolution)
        return x_idx, y_idx
    
    def get_occupancy(self, position):
        """Get occupancy probability at position"""
        x_idx, y_idx = self.world_to_grid(position)
        
        if self.in_grid(x_idx, y_idx):
            return self.probability(self.grid[x_idx, y_idx])
        else:
            return 0.5  # Unknown
    
    def log_odds(self, p):
        """Convert probability to log odds"""
        return np.log(p / (1 - p + 1e-10))
    
    def probability(self, log_odds):
        """Convert log odds to probability"""
        return 1 / (1 + np.exp(-log_odds))
    
    def in_grid(self, x, y):
        """Check if indices are in grid"""
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
```

## Best Practices

### 1. Time Synchronization

```python
class TimeSynchronizer:
    """Synchronize sensor timestamps"""
    
    def __init__(self):
        self.buffers = defaultdict(list)
        self.max_delay = 0.1  # 100ms
        
    def add_measurement(self, sensor_id, timestamp, data):
        """Add measurement to buffer"""
        self.buffers[sensor_id].append({
            'timestamp': timestamp,
            'data': data
        })
        
    def get_synchronized(self):
        """Get synchronized measurements"""
        if not all(len(b) > 0 for b in self.buffers.values()):
            return None
        
        # Find common time
        latest_earliest = max(
            min(b['timestamp'] for b in buffer)
            for buffer in self.buffers.values()
        )
        
        # Get measurements closest to this time
        synchronized = {}
        for sensor_id, buffer in self.buffers.items():
            closest = min(
                buffer,
                key=lambda x: abs(x['timestamp'] - latest_earliest)
            )
            synchronized[sensor_id] = closest['data']
        
        return synchronized
```

### 2. Sensor Health Monitoring

```python
class SensorHealthMonitor:
    """Monitor sensor health and availability"""
    
    def __init__(self):
        self.sensor_status = {}
        self.last_update = {}
        
    def update(self, sensor_id, timestamp):
        """Update sensor timestamp"""
        self.last_update[sensor_id] = timestamp
        
    def check_health(self):
        """Check health of all sensors"""
        current_time = time.time()
        
        for sensor_id, last_time in self.last_update.items():
            time_since_update = current_time - last_time
            
            if time_since_update > 1.0:
                self.sensor_status[sensor_id] = 'FAILED'
            elif time_since_update > 0.5:
                self.sensor_status[sensor_id] = 'DEGRADED'
            else:
                self.sensor_status[sensor_id] = 'OK'
        
        return self.sensor_status
```

### 3. Fallback Strategies

```python
class SensorFallback:
    """Graceful degradation when sensors fail"""
    
    def __init__(self):
        self.sensors = {
            'camera': True,
            'radar': True,
            'lidar': True
        }
        
    def get_perception_mode(self):
        """Determine perception mode based on available sensors"""
        if self.sensors['camera'] and self.sensors['lidar']:
            return 'full_fusion'
        elif self.sensors['camera'] and self.sensors['radar']:
            return 'camera_radar_fusion'
        elif self.sensors['camera']:
            return 'camera_only'
        elif self.sensors['lidar']:
            return 'lidar_only'
        else:
            return 'safe_stop'
    
    def execute_fallback(self):
        """Execute fallback strategy"""
        mode = self.get_perception_mode()
        
        if mode == 'safe_stop':
            # Safely stop vehicle
            return self.emergency_stop()
        elif mode == 'camera_only':
            # Increase caution
            return self.reduce_speed(0.5)
        else:
            # Normal operation
            return None
```

## Conclusion

Sensor fusion is critical for robust autonomous driving. By combining information from multiple sensors, the system can overcome individual sensor limitations and build a comprehensive understanding of the environment. Key aspects include proper time synchronization, health monitoring, and graceful degradation strategies.

## References

- "Multi-Sensor Fusion for Autonomous Driving" (papers)
- Kalman Filter tutorials
- "Probabilistic Robotics" (Thrun et al.)
- Tesla AI Day presentations

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~3500 words  
**Size**: ~20KB
