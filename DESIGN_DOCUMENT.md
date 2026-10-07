# AI Traffic System Scanning - Design Document

## Project Overview

**AI Traffic System Scanning** is a computer vision application that detects, classifies, and tracks vehicles in real-time from images and video streams. The system uses YOLOv8 (deep learning) to identify cars, motorcycles, buses, and trucks, providing traffic analysis and monitoring capabilities.

**Target Users:** Traffic engineers, city planners, security personnel, traffic researchers

**Platform:** Google Colab (cloud-based, no local GPU required)

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
Input Layer (Images/Videos)
        ↓
Preprocessing Module
        ↓
YOLOv8 Detection Model
        ↓
Post-Processing & Filtering
        ↓
Analytics & Reporting
        ↓
Output Layer (Visualizations/Reports/CSV)
```

### 1.2 Component Breakdown

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **Input Handler** | Load images/videos from Colab storage or URL | OpenCV, PIL |
| **Model Loader** | Load pretrained YOLOv8 model | Ultralytics |
| **Frame Processor** | Process each frame for detection | OpenCV, NumPy |
| **Detector Engine** | Run YOLO inference on frames | YOLOv8 |
| **Tracker** | Optional: track vehicle movement across frames | DeepSORT (optional) |
| **Analytics Engine** | Count, classify, generate statistics | Pandas, NumPy |
| **Visualizer** | Draw bounding boxes and annotate results | OpenCV, Matplotlib |
| **Reporter** | Generate CSV, JSON, and summary reports | Pandas, JSON |
| **Dashboard** | Optional: interactive UI for results | Streamlit |

---

## 2. Data Flow

### 2.1 Image Processing Flow

```
Upload Image to Colab
        ↓
Load with OpenCV/PIL
        ↓
Resize to 640×640 (YOLO standard)
        ↓
Run YOLO Inference
        ↓
Extract Detections (class, confidence, bbox)
        ↓
Filter by Traffic Classes (car, bus, truck, motorcycle)
        ↓
Draw Bounding Boxes
        ↓
Display/Save Annotated Image
        ↓
Generate Statistics (count, confidence)
```

### 2.2 Video Processing Flow

```
Upload Video to Colab
        ↓
Open with OpenCV VideoCapture
        ↓
Initialize VideoWriter (for output)
        ↓
Loop Through Frames:
  ├─ Read Frame
  ├─ Run YOLO Inference
  ├─ Filter Traffic Classes
  ├─ Count Vehicles in Frame
  ├─ Draw Annotations
  ├─ Write Frame to Output Video
  └─ Log Frame Statistics
        ↓
Close Video Files
        ↓
Generate Summary Report (total vehicles, avg per frame)
        ↓
Save Output Video + CSV Report
```

---

## 3. Functional Requirements

### 3.1 Core Features

| Feature | Description | Priority |
|---------|-------------|----------|
| Vehicle Detection | Detect cars, buses, trucks, motorcycles | HIGH |
| Vehicle Classification | Classify detected objects by type | HIGH |
| Vehicle Counting | Count total vehicles per image/video | HIGH |
| Confidence Scoring | Show detection confidence (0-1) | HIGH |
| Bounding Box Visualization | Draw boxes around detected vehicles | HIGH |
| Batch Processing | Process multiple images/videos | MEDIUM |
| CSV Export | Export statistics to CSV | MEDIUM |
| Traffic Density Report | Generate low/medium/high traffic status | MEDIUM |
| Frame-by-Frame Analysis | Log data for each frame | MEDIUM |
| Video Annotation | Save output video with detections | HIGH |

### 3.2 Nice-to-Have Features

| Feature | Description |
|---------|-------------|
| Vehicle Tracking | Track same vehicle across multiple frames |
| Speed Estimation | Estimate vehicle speed (requires calibration) |
| Lane Detection | Identify vehicles by lane |
| Alerts | Trigger alerts for anomalies (e.g., sudden traffic spike) |
| Heatmap Generation | Show traffic concentration areas |
| Real-time Dashboard | Interactive web UI with Streamlit |
| Database Integration | Store results in cloud database |

---

## 4. Technical Specifications

### 4.1 Model Details

```
Model Name:        YOLOv8 Nano (yolov8n.pt)
Framework:         Ultralytics
Input Size:        640 × 640 pixels
Output:            Detections (class, confidence, bbox coordinates)
Inference Speed:   ~30ms per frame (GPU) / ~100ms (CPU)
Pre-training:      COCO dataset (80 classes)
```

### 4.2 YOLO Class Mapping (Traffic-Relevant)

```
Class ID | Class Name    | Include in Traffic Count?
---------|---------------|------------------------
2        | car           | YES
3        | motorcycle    | YES
5        | bus           | YES
7        | truck         | YES
6        | train         | NO (rail, not road)
8        | boat          | NO (water)
```

### 4.3 Confidence Threshold

- **Default:** 0.25 (25%)
- **Conservative:** 0.50 (50%) — fewer false positives
- **Aggressive:** 0.10 (10%) — catches more detections, more false positives

### 4.4 Image/Video Specifications

| Parameter | Specification |
|-----------|---------------|
| Image Format | JPEG, PNG, BMP, WebP |
| Video Format | MP4, AVI, MOV, MKV |
| Resolution | 480p minimum (1080p+ recommended) |
| FPS (Video) | 24, 30, 60 FPS supported |
| File Size Limit | 500 MB per Colab session (storage) |

---

## 5. System Environment (Google Colab)

### 5.1 Dependencies

```
Python Version:     3.8 - 3.11
ultralytics         ≥ 8.0.0
opencv-python       ≥ 4.7.0
numpy               ≥ 1.21.0
pandas              ≥ 1.3.0
matplotlib          ≥ 3.5.0
torch               ≥ 1.13.0 (auto-installed with ultralytics)
```

### 5.2 Installation (Colab Cell)

```bash
!pip install ultralytics opencv-python pandas matplotlib
```

### 5.3 Runtime Environment

```
GPU:     Tesla T4 (or V100, A100 depending on Colab quota)
RAM:     12.7 GB
Storage: 108 GB (shared with all Colab files)
```

---

## 6. Data Models

### 6.1 Detection Object (Per Vehicle)

```json
{
  "frame_id": 0,
  "vehicle_id": "v_001",
  "class_id": 2,
  "class_name": "car",
  "confidence": 0.92,
  "bbox": {
    "x1": 100,
    "y1": 150,
    "x2": 250,
    "y2": 300
  },
  "center_x": 175,
  "center_y": 225
}
```

### 6.2 Frame Statistics

```json
{
  "frame_number": 0,
  "timestamp": "00:00:00.000",
  "total_vehicles": 5,
  "by_class": {
    "car": 3,
    "bus": 1,
    "truck": 1,
    "motorcycle": 0
  },
  "avg_confidence": 0.89,
  "traffic_density": "medium"
}
```

### 6.3 Video Summary Report

```json
{
  "video_name": "traffic_video.mp4",
  "duration_seconds": 60,
  "total_frames": 1800,
  "fps": 30,
  "total_vehicles_detected": 145,
  "avg_vehicles_per_frame": 2.41,
  "by_class": {
    "car": 98,
    "bus": 20,
    "truck": 15,
    "motorcycle": 12
  },
  "avg_confidence": 0.87,
  "traffic_density_label": "high"
}
```

---

## 7. Output Formats

### 7.1 CSV Report

```
frame_id, timestamp, total_vehicles, cars, buses, trucks, motorcycles, avg_confidence, traffic_density
0, 00:00:00, 5, 3, 1, 1, 0, 0.89, medium
1, 00:00:00.033, 6, 4, 1, 1, 0, 0.87, medium
2, 00:00:00.066, 4, 3, 0, 1, 0, 0.91, low
...
```

### 7.2 JSON Report

```json
{
  "metadata": {
    "analysis_type": "video",
    "file_name": "traffic_video.mp4",
    "analysis_date": "2026-10-07T10:30:00Z"
  },
  "summary": {
    "total_frames": 1800,
    "total_vehicles": 145,
    "avg_vehicles_per_frame": 2.41
  },
  "by_class": {...},
  "frames": [{...}]
}
```

### 7.3 Annotated Output

- **For Images:** PNG/JPEG with bounding boxes + labels
- **For Videos:** MP4 with overlaid bounding boxes, vehicle counts, and traffic status

---

## 8. Traffic Density Classification

### 8.1 Classification Logic

```
Traffic Density = (Total Vehicles Detected) / (Bounding Area Coverage)

LOW:     0-2 vehicles per frame
MEDIUM:  3-5 vehicles per frame
HIGH:    6+ vehicles per frame
```

### 8.2 Color Coding (Visualizations)

```
LOW:     Green   (#00FF00)
MEDIUM:  Yellow  (#FFFF00)
HIGH:    Red     (#FF0000)
```

---

## 9. Workflow in Google Colab

### 9.1 Notebook Structure

```
Cell 1:  Install Dependencies
Cell 2:  Import Libraries
Cell 3:  Load YOLO Model
Cell 4:  Set Configuration (confidence, class filters)
Cell 5:  [Option A] Process Single Image
Cell 6:  [Option B] Process Video
Cell 7:  Generate Reports
Cell 8:  Display Results
Cell 9:  Download Outputs
```

### 9.2 User Workflow

1. Open Google Colab notebook
2. Run Cell 1-3 to install and load model
3. Upload image/video to Colab Files
4. Run detection cell (A or B)
5. View annotated output
6. Download CSV report and output video
7. (Optional) Modify confidence threshold and re-run

---

## 10. Error Handling & Validation

### 10.1 Input Validation

```
✓ File exists and is readable
✓ File format is supported (jpg, png, mp4, avi, etc.)
✓ File is not corrupted
✓ Resolution is within acceptable range (min 480p)
✓ Confidence threshold is between 0.0 - 1.0
```

### 10.2 Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | File path incorrect | Verify path in Colab Files panel |
| `cv2.error: (-5, 'OpenCV(4.x.x)...')` | Video format unsupported | Convert to MP4 with ffmpeg |
| `CUDA out of memory` | Model too large for GPU | Use lighter model (yolov8n vs yolov8x) |
| `Zero detections` | Low confidence threshold | Lower confidence to 0.15 or 0.10 |
| `Slow processing` | CPU fallback (no GPU) | Request GPU in Colab runtime |

---

## 11. Performance Metrics

### 11.1 Expected Performance

| Metric | Benchmark |
|--------|-----------|
| Image Processing | 1-2 seconds (1080p image) |
| Video Processing (1 min) | 2-5 minutes (GPU), 15-30 min (CPU) |
| Model Load Time | 5-10 seconds |
| Detection Accuracy (COCO) | ~80-90% mAP |
| False Positive Rate | ~5-15% (varies by scene) |
| Inference Time/Frame | ~30ms (GPU) |

### 11.2 Optimization Tips

1. **Use YOLOv8n (Nano)** instead of YOLOv8x for speed
2. **Lower image size to 416×416** for faster processing (trade-off: accuracy)
3. **Batch processing:** Process multiple frames at once
4. **GPU Acceleration:** Enable GPU runtime in Colab (Runtime → Change Runtime Type → GPU)

---

## 12. Scalability & Future Enhancements

### 12.1 Short-term (Next Release)

- [ ] Add vehicle tracking across frames (DeepSORT)
- [ ] Implement speed estimation (with camera calibration)
- [ ] Add lane detection
- [ ] Generate traffic flow heatmaps

### 12.2 Medium-term (Future)

- [ ] Multi-camera support
- [ ] Cloud database storage (Firebase, Google Cloud Storage)
- [ ] Web dashboard (Streamlit Cloud deployment)
- [ ] Real-time alerts API

### 12.3 Long-term

- [ ] Custom YOLOv8 model training on domain-specific data
- [ ] Integration with traffic management systems
- [ ] Mobile app for real-time monitoring
- [ ] Edge deployment (NVIDIA Jetson, Raspberry Pi)

---

## 13. Security & Privacy

### 13.1 Data Handling

- **Video/Image Storage:** Kept in Colab session only (deleted when session ends)
- **No Cloud Sync:** User controls all file uploads/downloads
- **Privacy:** No facial recognition or license plate OCR (by design)

### 13.2 Best Practices

- Don't share Colab notebooks with sensitive video files
- Use anonymized/public traffic footage for testing
- Clear Colab runtime after processing sensitive data

---

## 14. Testing & Quality Assurance

### 14.1 Test Cases

```
TC-001: Detect single vehicle in image → PASS (confidence > 0.8)
TC-002: Count multiple vehicles in video → PASS (±10% accuracy)
TC-003: Filter non-traffic classes → PASS (ignore person, dog, etc.)
TC-004: Handle corrupted video file → FAIL gracefully
TC-005: Generate CSV with correct format → PASS
TC-006: Confidence threshold filtering → PASS
```

### 14.2 Validation Data

- Use public traffic datasets: KITTI, BDD100K, Cityscapes
- Test with various lighting conditions (day, night, rain)
- Test with different vehicle types (cars, buses, trucks, motorcycles)

---

## 15. Documentation & Support

### 15.1 Code Comments

- Each function has docstring with input/output spec
- Inline comments for complex logic
- Configuration parameters labeled clearly

### 15.2 User Guide

- Step-by-step Colab notebook with markdown cells
- Example videos and images provided
- Troubleshooting section in notebook

### 15.3 API Reference

```python
detect_image(image_path, conf=0.25, classes=[2,3,5,7])
detect_video(video_path, output_path, conf=0.25, classes=[2,3,5,7])
generate_report(detections, output_format='csv')
visualize_detections(image, detections)
```

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **YOLO** | You Only Look Once — real-time object detection model |
| **mAP** | Mean Average Precision — accuracy metric for detection |
| **Confidence** | Model's certainty in a detection (0-1 scale) |
| **bbox** | Bounding Box — rectangle around detected object |
| **IOU** | Intersection over Union — overlap metric for bounding boxes |
| **FPS** | Frames Per Second — video playback rate |
| **GPU** | Graphics Processing Unit — accelerates inference |
| **Inference** | Running model on input data to generate predictions |

---

## 17. Appendix: Reference Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   GOOGLE COLAB ENVIRONMENT                  │
├─────────────────────���───────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐                                             │
│  │  File Upload│  ← Images/Videos from user                 │
│  └──────┬──────┘                                             │
│         │                                                     │
│  ┌──────▼──────────────────────────────────────────────┐    │
│  │         INPUT PREPROCESSING MODULE                  │    │
│  │  (OpenCV: resize, normalize, format conversion)     │    │
│  └──────┬───────────────────────────────────────────────┘   │
│         │                                                     │
│  ┌──────▼──────────────────────────────────────────────┐    │
│  │     YOLOv8 INFERENCE ENGINE (GPU/CPU)               │    │
│  │  (ultralytics library + PyTorch backend)            │    │
│  └──────┬───────────────────────────────────────────────┘   │
│         │                                                     │
│  ┌──────▼──────────────────────────────────────────────┐    │
│  │      POST-PROCESSING & FILTERING LAYER              │    │
│  │  (Filter by class ID, confidence threshold)         │    │
│  └──────┬───────────────────────────────────────────────┘   │
│         │                                                     │
│  ┌──────┴─────────────────────────────────────────────┐    │
│  │                                                      │    │
│  ▼                                                      ▼    │
│┌─────────────────┐                          ┌──────────────┐│
││ VISUALIZATION   │                          │  ANALYTICS   ││
││ MODULE          │                          │  ENGINE      ││
││(Draw boxes,     │                          │(Count,       ││
││ labels, arrows) │                          │ statistics)  ││
│└────────┬────────┘                          └────────┬─────┘│
│         │                                            │       │
│  ┌──────┴────────────────────────────────────────────┴────┐ │
│  │            REPORTING & EXPORT MODULE                   │ │
│  │  (CSV, JSON, PNG, MP4 with annotations)               │ │
│  └──────┬─────────────────────────────────────────────────┘ │
│         │                                                     │
│  ┌──────▼──────────────────────────────────────────────┐    │
│  │         OUTPUT DOWNLOAD (User Download)             │    │
│  │  (Annotated image/video, CSV report, summary)       │    │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 1.0  
**Last Updated:** 2026-10-07  
**Author:** AI Traffic System Team  
**Status:** APPROVED
