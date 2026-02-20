# SUPERIOT_UoB
Resources for Node and Network-level Energy Measurement, Analysis, Modeling and Prediction in the SUPERIOT Framework.

This repository collects software, datasets, and supporting material produced in SUPERIOT (Hybrid OWC/RF Reconfigurable IoT System), with a focus on energy measurement, modelling, prediction, and optimisation at node, access point, and network level.

# Project website: https://superiot.eu/

# Recommended reading (preprints):
https://arxiv.org/abs/2510.09842 

https://arxiv.org/abs/2501.10093

https://arxiv.org/abs/2511.09303

# What’s included
## 1) Node-level Energy Prediction App (RIoT node)

A user-friendly energy consumption + prediction application for the SUPERIOT Silicon-based Reconfigurable IoT node, built with MATLAB App Designer.

### Download (Windows installer / standalone app):
https://uob-my.sharepoint.com/:u:/g/personal/mb13530_bristol_ac_uk/IQBgAbYJYKh7TY4Qc2z0W44TAesNomKxTbdM7Kh05YFCr40?e=WklI4A

### App overview (4 tabs):

Tab 1 — Normal (unoptimised) node operation

Tab 2 — Low-power (software-optimised) node operation

Tab 3 — Very low-power (hardware-optimised) node operation

Tab 4 — Demo (includes energy harvesting dynamics)

### See: Useful resources/node_app_snapshot.pdf

### Key features (Tabs 1–3):

- Specify different input parameters such as BLE TX power level, advertising interval, connection interval, operating duration, idle (standby) periods, activate and deactivate different functionalities such as Sensing, E-ink display (optimized or unoptimized),  NBVLC, BLE, deep sleep, etc.

Visualise:

- Total energy consumption

- State-wise energy consumption over a periodic duty cycle

### Additional features (Tab 4 — Demo):

- Custom operation definition: model arbitrary scenarios using states such as BLE, VLC, sensing, E-ink, wake-up, idle, sleep, etc. Supports periodic duty cycles and randomly triggered events.

- Energy harvesting simulation: RF + solar harvesting with user-defined parameters, plus supercapacitor charge/discharge dynamics, driven by the defined node operation.

- Save/Load results: save intermediate progress and resume later.

- Realistic current-profile dataset generation: generate labelled datasets from defined operations, optionally with or without energy-harvesting dynamics, with:

   - Configurable sampling frequency

   - Add controlled variations in currents/durations (to mimic real-world profiles)

### Demo video:
DEMO_Node_Energy_Prediction_App.mp4
https://uob-my.sharepoint.com/:f:/g/personal/mb13530_bristol_ac_uk/IgBe3OrBNvEoQZRwfM5wk6MIATmK5e5hM3cqq7x1kI9mENY?e=dzGDcp

## 2) Sample Datasets (generated from the Node-level App)

This repo also provides a link to download time-series datasets created using the node-level app, spanning a wide range of states, modes, and configurations (randomised).

Dataset download: https://uob-my.sharepoint.com/:f:/g/personal/mb13530_bristol_ac_uk/IgAM_ExlEvG5QJjhXY-ItFj-AT4SXYDtPxeA_GsLgJhXKnw?e=6mBZrW

### Dataset contents:

- 7 CSV files: dataset_1.csv … dataset_7.csv

- Resolution: 1 ms (1000 samples/s)

- Node operating voltage: 3.3 V

- Duration: ~24 hours per dataset (continuous operation)

- Operating modes covered:

- Normal mode: BLE TX/RX, NBVLC TX/RX, startup, sensing, E-ink, BLE idle

- Low-power mode: as normal, but NBVLC RX disabled; BLE parameters remain configurable

- Very low-power mode: BLE disabled, NBVLC RX disabled, NBVLC TX optional; startup/wake-up, sensing, idle, E-ink updates, deep sleep

- Hardware optimisation options (very-low power mode deep sleep):

  - U6 and U9 shorted

  - U6 cut only

  - U9 cut only

  - both U6 and U9 cut (lowest deep sleep current)

  Where:

    U6: point linking the E-ink power-control circuitry to the always-on supply

    U9: measuring point for NBVLC RX-only current

### Schema (columns):

- Time_s — timestamp (seconds)

- Current_mA — instantaneous current (mA)

- State — functional state (e.g., ble_fast_advertising, sleep, sensing, eink, vlc_tx, ble_rx, …)

- StateDuration_s — duration of the current state (seconds)

- BLE_TX_Power_dBm

- BLE_Connection_Interval_ms

- BLE_Advertising_Interval_ms

- BLE_TX_Functionality, BLE_RX_Functionality, NBVLC_TX_Functionality, NBVLC_RX_Functionality — enabled/disabled flags

- Hardware_Optimisation — active U6/U9 configuration

- Energy_nJ — energy consumed during the sample period (nJ)

Note: These sample datasets were generated without energy-harvesting dynamics enabled.

To generate datasets for your own scenarios (with or without harvesting), use Tab 4 in the node app.

## 3) Access Point Energy Prediction App (BBB-based Hybrid OWC/RF AP)

A companion tool (also built with MATLAB App Designer) for energy profiling and dataset generation of the BeagleBone Black (BBB)-based SUPERIOT Access Point.

### Download:
https://uob-my.sharepoint.com/:u:/g/personal/mb13530_bristol_ac_uk/IQDXH7K0pFbHRryJPAjk1ZbXAVCn8P0r2krf2dKgi3brW5A?e=h94od0

### What it models:

- boot and idle states

- VLC and BLE activity modes

- Ethernet data transmission

- combined VLC–BLE operations

### What you can configure:

- VLC PWM duty cycle (lamp brightness)

- number of VLC frame chunks (32-bit)

- delays / idle periods / duration for ethernet data transmission

- BLE scan interval/window

- enable/disable VLC and BLE modes

- periods of VLC command transmission and data reception

- BLE scanning and connected modes (with/without data exchange)

### Visualisations + outputs:

- current profile

- energy consumption breakdown

- optional dataset generation

### Supporting snapshot: Useful resources/bbb_app_snapshot.pdf

### Demo video:
DEMO_BBB_AP_Energy_Prediciton_App.mp4

https://uob-my.sharepoint.com/:f:/g/personal/mb13530_bristol_ac_uk/IgBe3OrBNvEoQZRwfM5wk6MIATmK5e5hM3cqq7x1kI9mENY?e=dzGDcp

### Installation (Windows)

The node-level and BBB access point energy prediction apps were developed using MATLAB 2024b.

To run the apps on Windows:

Download the provided .exe installer(s)

Install and launch the standalone application

MATLAB does not need to be installed to run the standalone apps.


## 4) Network-level Energy Measurement, Prediction, and Optimisation

For network-level tooling and measurement setup:

### Read: Network Level Energy Measurement, Optimization and Prediction/Apps-Predictor-Optimizer/Instructions.docx

### Demo videos:

Network Level Energy Measurement Demo.mp4

Network Level Energy Predictor Demo.mp4

https://uob-my.sharepoint.com/:f:/g/personal/mb13530_bristol_ac_uk/IgBe3OrBNvEoQZRwfM5wk6MIATmK5e5hM3cqq7x1kI9mENY?e=DK9kvm

# Notes on access to downloads

Several downloads are hosted on University of Bristol SharePoint. Access may require appropriate permissions or an authenticated account.

# Citation

If you use these tools or datasets in academic work, please cite the relevant SUPERIOT papers (see the “Recommended reading” links above). If you’d like, paste your preferred citation format (BibTeX / IEEE / ACM) and I’ll format the references accordingly.
