# EMG Hand Gesture Recognition

A hand gesture recognition system based on surface electromyography (sEMG) signals. The pipeline acquires EMG signals, preprocesses them, detects muscle activations, extracts relevant features, and classifies hand gestures using machine learning.

**Recognized gestures:** Open Palm · Closed Fist · Thumb Up · Neutral

**Pipeline:** Acquisition → Preprocessing → Segmentation → Feature Extraction → Classification

## Hardware

EMG signals were acquired using a **DFRobot SEN0240 EMG sensor** connected to an **Arduino UNO**, with a sampling frequency of **1000 Hz**.

## Project Files

| File | Purpose |
|---|---|
| `emg_recording.py` | EMG signal acquisition and data logging |
| `emg_processing.py` | Signal preprocessing, filtering, and quality analysis |
| `gesture_analysis.py` | Gesture segmentation, activation detection, and feature extraction |
| `model_training.py` | Machine learning training and evaluation |

## Results

Best-performing model: Random Forest classifier, achieving 91%+ macro accuracy across the four recognized gestures.

## Requirements

```bash
pip install pandas numpy scipy scikit-learn matplotlib pyserial
```

## Documentation

See `EMG_Project_Paper.pdf` for the full methodology and results.
