# Computer Vision for Self-Driving Cars

## Introduction

Computer vision is the foundation of Tesla's autonomous driving system. This document covers the fundamental techniques, algorithms, and best practices for building robust computer vision systems for self-driving applications.

## Camera Systems

### Camera Configuration

Tesla's camera suite provides 360° coverage:

```
Camera Layout:

        Front-Narrow
             |
             v
    ┌─────────────────────┐
    │      Front-Wide     │
    │                     │
Left│    [Vehicle]        │Right
Fwd │                     │Fwd
    │      Front-Main     │
    └─────────────────────┘
             |
        Rear-View
    
    Plus: Left-Rear, Right-Rear
```

### Camera Specifications

```python
class CameraConfig:
    """Tesla camera specifications"""
    
    CAMERAS = {
        'front_main': {
            'resolution': (1280, 960),
            'fov': 50,  # degrees
            'fps': 36,
            'purpose': 'Long-range detection'
        },
        'front_wide': {
            'resolution': (1280, 960),
            'fov': 150,  # degrees
            'fps': 36,
            'purpose': 'Wide-angle context'
        },
        'front_narrow': {
            'resolution': (1280, 960),
            'fov': 25,  # degrees
            'fps': 36,
            'purpose': 'Far-distance detail'
        },
        'left_fwd': {
            'resolution': (1280, 960),
            'fov': 90,
            'fps': 36,
            'purpose': 'Blind spot, intersections'
        },
        'right_fwd': {
            'resolution': (1280, 960),
            'fov': 90,
            'fps': 36,
            'purpose': 'Blind spot, intersections'
        },
        'left_rear': {
            'resolution': (1280, 960),
            'fov': 90,
            'fps': 36,
            'purpose': 'Lane changes, parking'
        },
        'right_rear': {
            'resolution': (1280, 960),
            'fov': 90,
            'fps': 36,
            'purpose': 'Lane changes, parking'
        },
        'rear': {
            'resolution': (1280, 960),
            'fov': 90,
            'fps': 36,
            'purpose': 'Rear monitoring'
        }
    }
```

## Image Preprocessing

### Image Signal Processing

```python
class ImageProcessor:
    """Process raw camera data"""
    
    def __init__(self):
        self.params = {
            'exposure': 0.0,      # Exposure compensation
            'white_balance': True,
            'denoise': True,
            'tone_mapping': 'HDR',
            'sharpness': 1.0
        }
        
    def process(self, raw_image):
        """Process raw Bayer image"""
        # 1. Demosaic
        rgb = self.demosaic(raw_image)
        
        # 2. Black level adjustment
        rgb = self.adjust_black_level(rgb)
        
        # 3. White balance
        if self.params['white_balance']:
            rgb = self.auto_white_balance(rgb)
        
        # 4. Denoise
        if self.params['denoise']:
            rgb = self.denoise(rgb)
        
        # 5. HDR tone mapping
        if self.params['tone_mapping'] == 'HDR':
            rgb = self.hdr_tone_map(rgb)
        
        # 6. Color correction
        rgb = self.color_correct(rgb)
        
        # 7. Gamma correction
        rgb = self.gamma_correct(rgb)
        
        return rgb
    
    def demosaic(self, raw):
        """Convert Bayer pattern to RGB"""
        # Raw: [H, W] with RGGB pattern
        # Output: [H, W, 3] RGB image
        
        # Bilinear interpolation
        r = raw[0::2, 0::2]
        g = (raw[0::2, 1::2] + raw[1::2, 0::2]) / 2
        b = raw[1::2, 1::2]
        
        return np.stack([r, g, b], axis=-1)
```

### Augmentation Pipeline

```python
import albumentations as A

class DrivingAugmentation:
    """Domain-specific augmentations for driving"""
    
    def __init__(self, mode='train'):
        if mode == 'train':
            self.transform = A.Compose([
                # Weather simulation
                A.OneOf([
                    A.RandomRain(p=1.0),
                    A.RandomFog(p=1.0),
                    A.RandomSunFlare(p=1.0),
                    A.RandomShadow(p=1.0),
                ], p=0.3),
                
                # Lighting changes
                A.OneOf([
                    A.RandomBrightnessContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    A.RGBShift(p=1.0),
                ], p=0.5),
                
                # Noise and blur
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(p=1.0),
                    A.MotionBlur(p=1.0),
                ], p=0.2),
                
                # Geometric (minimal)
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=3,
                    p=0.3
                ),
            ])
        else:
            self.transform = A.Compose([])
    
    def __call__(self, image, annotations):
        transformed = self.transform(image=image, **annotations)
        return transformed['image'], transformed
```

## Object Detection

### 2D Object Detection

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectDetector(nn.Module):
    """Detect objects in camera images"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Backbone
        self.backbone = ResNet50(pretrained=True)
        
        # Feature Pyramid Network
        self.fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        # Detection heads
        self.cls_head = nn.Conv2d(256, num_classes, 3, padding=1)
        self.reg_head = nn.Conv2d(256, 4, 3, padding=1)  # x, y, w, h
        
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Build feature pyramid
        pyramid_features = self.fpn(features)
        
        # Detect at each scale
        detections = []
        for feat in pyramid_features:
            cls_scores = self.cls_head(feat)
            bbox_deltas = self.reg_head(feat)
            detections.append((cls_scores, bbox_deltas))
        
        # Decode detections
        boxes, scores, classes = self.decode_detections(detections)
        
        return boxes, scores, classes
    
    def decode_detections(self, detections):
        """Convert network outputs to bounding boxes"""
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for cls_scores, bbox_deltas in detections:
            # Get class predictions
            scores, classes = cls_scores.max(dim=1)
            
            # Decode bounding boxes
            boxes = self.delta_to_box(bbox_deltas)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)
        
        # Concatenate all scales
        boxes = torch.cat(all_boxes, dim=1)
        scores = torch.cat(all_scores, dim=1)
        classes = torch.cat(all_classes, dim=1)
        
        # Apply NMS
        keep = self.nms(boxes, scores, iou_threshold=0.5)
        
        return boxes[keep], scores[keep], classes[keep]
```

### 3D Object Detection

```python
class Object3DDetector(nn.Module):
    """Detect objects in 3D from monocular images"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Shared backbone
        self.backbone = ResNet50()
        
        # 3D detection heads
        self.center_3d = nn.Conv2d(256, 3, 1)  # x, y, z
        self.size_3d = nn.Conv2d(256, 3, 1)    # l, w, h
        self.heading_3d = nn.Conv2d(256, 2, 1) # sin, cos
        self.cls_head = nn.Conv2d(256, num_classes, 1)
        
        # Depth estimation (auxiliary)
        self.depth_head = nn.Conv2d(256, 1, 1)
        
    def forward(self, image, calib):
        """
        Args:
            image: [B, 3, H, W]
            calib: camera calibration parameters
        Returns:
            boxes_3d: [N, 7] (x, y, z, l, w, h, heading)
            scores: [N]
            classes: [N]
        """
        # Extract features
        features = self.backbone(image)
        
        # Predict 3D properties
        center_3d = self.center_3d(features)
        size_3d = self.size_3d(features)
        heading_3d = self.heading_3d(features)
        cls_scores = self.cls_head(features)
        
        # Decode 3D boxes
        boxes_3d = self.decode_3d_boxes(
            center_3d, size_3d, heading_3d, calib
        )
        
        # Get scores and classes
        scores, classes = cls_scores.max(dim=1)
        
        return boxes_3d, scores, classes
    
    def decode_3d_boxes(self, center, size, heading, calib):
        """Convert predictions to 3D boxes in world coordinates"""
        # Unproject image coordinates to 3D
        depth = center[:, 2:3]  # Predicted depth
        
        # Image coordinates
        u = center[:, 0:1]
        v = center[:, 1:2]
        
        # Unproject to camera frame
        x = (u - calib.cx) * depth / calib.fx
        y = (v - calib.cy) * depth / calib.fy
        z = depth
        
        # Heading angle
        heading_angle = torch.atan2(heading[:, 0], heading[:, 1])
        
        # Combine
        boxes_3d = torch.cat([
            x, y, z,           # Center
            size,              # Dimensions
            heading_angle      # Orientation
        ], dim=-1)
        
        return boxes_3d
```

## Lane Detection

### Lane Segmentation

```python
class LaneDetector(nn.Module):
    """Detect lane markings and boundaries"""
    
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = ResNet34(pretrained=True)
        
        # Decoder
        self.decoder = UNetDecoder(
            encoder_channels=[64, 64, 128, 256, 512],
            decoder_channels=[256, 128, 64, 32]
        )
        
        # Lane head
        self.lane_head = nn.Conv2d(32, 5, 1)  # 4 lane classes + background
        
    def forward(self, image):
        # Encode
        features = self.encoder(image)
        
        # Decode
        decoded = self.decoder(features)
        
        # Lane segmentation
        lane_seg = self.lane_head(decoded)
        
        return lane_seg
    
    def get_lane_lines(self, lane_seg):
        """Extract lane line coordinates from segmentation"""
        # Convert to probabilities
        probs = F.softmax(lane_seg, dim=1)
        
        # Extract each lane
        lanes = []
        for lane_idx in range(1, 5):  # Skip background
            lane_prob = probs[:, lane_idx]
            
            # Threshold
            lane_mask = (lane_prob > 0.5).float()
            
            # Get lane points
            points = self.mask_to_points(lane_mask)
            
            # Fit polynomial
            lane_poly = self.fit_lane_polynomial(points)
            
            lanes.append(lane_poly)
        
        return lanes
    
    def fit_lane_polynomial(self, points):
        """Fit polynomial to lane points"""
        # Use RANSAC for robustness
        from sklearn.linear_model import RANSACRegressor
        
        X = points[:, 0].reshape(-1, 1)  # x coordinates
        y = points[:, 1]  # y coordinates
        
        # Fit 2nd degree polynomial
        ransac = RANSACRegressor()
        ransac.fit(X, y)
        
        return ransac
```

### Lane Topology

```python
class LaneTopology:
    """Understand lane relationships and structure"""
    
    def __init__(self):
        self.graph = LaneGraph()
        
    def build_topology(self, lane_segments):
        """Build lane connectivity graph"""
        for segment in lane_segments:
            # Add segment to graph
            self.graph.add_node(segment.id, segment)
            
            # Connect to adjacent lanes
            if segment.left_neighbor:
                self.graph.add_edge(
                    segment.id, 
                    segment.left_neighbor,
                    relation='left'
                )
            
            if segment.right_neighbor:
                self.graph.add_edge(
                    segment.id,
                    segment.right_neighbor,
                    relation='right'
                )
            
            # Connect to predecessor/successor
            if segment.predecessor:
                self.graph.add_edge(
                    segment.predecessor,
                    segment.id,
                    relation='precedes'
                )
            
            if segment.successor:
                self.graph.add_edge(
                    segment.id,
                    segment.successor,
                    relation='precedes'
                )
    
    def get_drivable_path(self, start_lane, goal_lane):
        """Find path through lane network"""
        # Use graph search
        path = self.graph.shortest_path(
            start_lane,
            goal_lane,
            weight='length'
        )
        
        return path
```

## Depth Estimation

### Monocular Depth Estimation

```python
class DepthEstimator(nn.Module):
    """Estimate depth from single camera"""
    
    def __init__(self):
        super().__init__()
        
        # Encoder-decoder architecture
        self.encoder = ResNet50(pretrained=True)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        
    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W]
        Returns:
            depth: [B, 1, H, W] in meters
        """
        # Encode
        features = self.encoder(image)
        
        # Decode
        depth = self.decoder(features)
        
        # Ensure positive depth
        depth = F.relu(depth)
        
        return depth
    
    def supervised_loss(self, pred_depth, gt_depth):
        """Supervised depth loss"""
        # Only compute loss where we have valid ground truth
        mask = (gt_depth > 0).float()
        
        # Scale-invariant loss
        log_diff = torch.log(pred_depth) - torch.log(gt_depth)
        loss = (log_diff ** 2 * mask).sum() / (mask.sum() + 1e-7)
        
        return loss
```

### Stereo Depth

```python
class StereoDepth(nn.Module):
    """Depth from stereo camera pair"""
    
    def __init__(self, max_disp=192):
        super().__init__()
        self.max_disp = max_disp
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        
        # Cost volume construction
        self.cost_volume = CostVolume(max_disp)
        
        # Disparity estimation
        self.disp_net = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, 3, padding=1)
        )
        
    def forward(self, left_image, right_image):
        """
        Args:
            left_image: [B, 3, H, W]
            right_image: [B, 3, H, W]
        Returns:
            disparity: [B, 1, H, W]
        """
        # Extract features
        left_feat = self.feature_extractor(left_image)
        right_feat = self.feature_extractor(right_image)
        
        # Build cost volume
        cost_vol = self.cost_volume(left_feat, right_feat)
        
        # Estimate disparity
        disparity = self.disp_net(cost_vol)
        
        # Soft argmin for differentiable
        disparity = self.soft_argmin(disparity)
        
        # Convert to depth
        depth = self.focal_length * self.baseline / (disparity + 1e-7)
        
        return depth
    
    def soft_argmin(self, cost):
        """Differentiable argmin operation"""
        # Get disparity probabilities
        prob = F.softmax(-cost, dim=1)
        
        # Expected disparity
        disp_candidates = torch.arange(
            self.max_disp, 
            device=cost.device,
            dtype=cost.dtype
        ).view(1, -1, 1, 1)
        
        disp = (prob * disp_candidates).sum(dim=1, keepdim=True)
        
        return disp
```

## Optical Flow and Motion

### Optical Flow Estimation

```python
class OpticalFlowNet(nn.Module):
    """Estimate pixel motion between frames"""
    
    def __init__(self):
        super().__init__()
        
        # Feature pyramid
        self.encoder = ResNet18()
        
        # Flow estimation
        self.flow_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1)  # u, v flow
        )
        
        # Flow refinement
        self.refine = FlowRefinement()
        
    def forward(self, frame1, frame2):
        """
        Args:
            frame1: [B, 3, H, W] time t
            frame2: [B, 3, H, W] time t+1
        Returns:
            flow: [B, 2, H, W] optical flow (u, v)
        """
        # Extract features
        feat1 = self.encoder(frame1)
        feat2 = self.encoder(frame2)
        
        # Correlation
        corr = self.compute_correlation(feat1, feat2)
        
        # Initial flow
        flow = self.flow_head(corr)
        
        # Upsample to full resolution
        flow = F.interpolate(flow, scale_factor=4, mode='bilinear')
        
        # Refine
        flow = self.refine(frame1, frame2, flow)
        
        return flow
    
    def compute_correlation(self, feat1, feat2):
        """Compute correlation volume"""
        B, C, H, W = feat1.shape
        
        # Reshape for correlation
        feat1 = feat1.view(B, C, H * W)
        feat2 = feat2.view(B, C, H * W)
        
        # Correlation
        corr = torch.bmm(feat1.transpose(1, 2), feat2)
        corr = corr.view(B, H, W, H, W)
        
        return corr
```

## Segmentation

### Semantic Segmentation

```python
class SemanticSegmentor(nn.Module):
    """Segment image into semantic classes"""
    
    def __init__(self, num_classes=20):
        super().__init__()
        
        # DeepLabV3+ architecture
        self.backbone = ResNet50(pretrained=True)
        
        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp = ASPP(
            in_channels=2048,
            out_channels=256,
            dilations=[1, 6, 12, 18]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        low_level = self.backbone.layer1(x)
        
        # ASPP
        aspp_features = self.aspp(features)
        
        # Upsample
        aspp_features = F.interpolate(
            aspp_features,
            size=low_level.shape[2:],
            mode='bilinear'
        )
        
        # Concatenate
        concat = torch.cat([aspp_features, low_level], dim=1)
        
        # Decode
        output = self.decoder(concat)
        
        # Upsample to input size
        output = F.interpolate(
            output,
            size=x.shape[2:],
            mode='bilinear'
        )
        
        return output


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels, out_channels, dilations):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        # 1x1 convolution
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        )
        
        # Dilated convolutions
        for dilation in dilations[1:]:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=dilation,
                        dilation=dilation
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            )
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        res = []
        
        for conv in self.convs:
            res.append(conv(x))
        
        # Global pooling
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(
            global_feat,
            size=x.shape[2:],
            mode='bilinear'
        )
        res.append(global_feat)
        
        # Concatenate and fuse
        res = torch.cat(res, dim=1)
        res = self.fusion(res)
        
        return res
```

## Best Practices

### 1. Calibration

```python
class CameraCalibration:
    """Maintain accurate camera calibration"""
    
    def __init__(self):
        self.intrinsics = {}  # Camera intrinsics
        self.extrinsics = {}  # Camera extrinsics
        self.distortion = {}  # Distortion coefficients
        
    def calibrate(self, camera_id, images):
        """Calibrate camera from checkerboard images"""
        import cv2
        
        # Find corners
        objpoints = []
        imgpoints = []
        
        for img in images:
            ret, corners = cv2.findChessboardCorners(img, (9, 6))
            if ret:
                objpoints.append(self.object_points)
                imgpoints.append(corners)
        
        # Calibrate
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img.shape[:2], None, None
        )
        
        self.intrinsics[camera_id] = mtx
        self.distortion[camera_id] = dist
        
    def undistort(self, image, camera_id):
        """Remove lens distortion"""
        import cv2
        
        mtx = self.intrinsics[camera_id]
        dist = self.distortion[camera_id]
        
        undistorted = cv2.undistort(image, mtx, dist)
        
        return undistorted
```

### 2. Real-time Performance

```python
class OptimizedInference:
    """Optimize for real-time inference"""
    
    def __init__(self, model):
        self.model = model
        
        # Quantize model
        self.quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        # Use half precision
        self.half_model = model.half()
        
    @torch.no_grad()
    def inference(self, image):
        """Fast inference"""
        # Use half precision
        image = image.half()
        
        # Run inference
        output = self.half_model(image)
        
        return output.float()
```

### 3. Robustness Testing

```python
class VisionTestSuite:
    """Test vision system robustness"""
    
    def __init__(self, model):
        self.model = model
        self.test_cases = []
        
    def add_test_case(self, name, image, expected):
        """Add test case"""
        self.test_cases.append({
            'name': name,
            'image': image,
            'expected': expected
        })
        
    def run_tests(self):
        """Run all test cases"""
        results = []
        
        for test in self.test_cases:
            # Run inference
            output = self.model(test['image'])
            
            # Check against expected
            passed = self.check_output(output, test['expected'])
            
            results.append({
                'name': test['name'],
                'passed': passed
            })
        
        return results
```

## Conclusion

Computer vision is fundamental to autonomous driving. Tesla's approach combines classical techniques with deep learning to build a robust perception system. Key aspects include multi-camera fusion, 3D understanding, temporal reasoning, and real-time performance optimization.

## References

- Tesla AI Day presentations
- "Deep Learning for Computer Vision" (Stanford CS231n)
- "Multi-View 3D Object Detection" papers
- "Optical Flow Estimation" literature

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~3000 words  
**Size**: ~18KB
