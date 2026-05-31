EMG Hand Gesture Recognition

This project implements a hand gesture recognition system based on surface electromyography (sEMG) signals.

The objective is to acquire EMG signals, preprocess them, detect muscle activations, extract relevant features and classify different hand gestures using machine learning techniques.

Recognized gestures:

Open Palm

Closed Fist

Thumb Up

Neutral

Processing pipeline:

Acquisition → Preprocessing → Segmentation → Feature Extraction → Classification

Project files:

pruebaemg.py : EMG signal acquisition and data logging

procesaremg.py : signal preprocessing, filtering and signal quality analysis

mediagestos.py : gesture segmentation, activation detection and feature extraction

modeltraining.py : machine learning training and evaluation
