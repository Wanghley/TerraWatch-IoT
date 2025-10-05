# TerraWatch-IoT Project Structure

## 📁 Folder Organization

This project is organized into four main components:

### 🔧 firmware-sensing-l1-l2/
**Level 1 & 2 Sensing - ESP32**
- Motion detection (Level 1)
- Multi-sensor fusion & ML classification (Level 2)
- Sensors: PIR, AMG8833 thermal, microphone, mmWave radar

### 📷 firmware-sensing-l3/
**Level 3 Sensing - Orange Pi Vision**
- CNN-based image analysis
- Animal species classification
- Decision engine for deterrence activation

### ⚡ firmware-deterrence/
**Deterrence System - ESP32**
- Floodlight & strobe control
- Predator call audio playback
- Wifi-triggered activation

### 🎨 hardware-design/
**CAD, PCB & Mechanical Design**
- 3D CAD models for enclosures and mounts
- PCB schematics and Gerber files
- Fabrication files (STL, DXF)
- Bill of materials

---

## 🚀 Getting Started

### Open the Workspace
1. Open `terrawatch.code-workspace` in VS Code
2. Install recommended extensions when prompted
3. Each firmware folder appears as a separate workspace root

### Install VS Code Extensions
For the best experience, install:
- **Material Icon Theme** - Beautiful folder icons
- **PlatformIO IDE** - ESP32 development
- **Python** - Orange Pi development

### Enable Material Icon Theme
1. Press `Cmd+Shift+P`
2. Type "Material Icons: Activate Icon Theme"
3. Select "Material Icon Theme"

---

## 📊 System Architecture

```
[PIR Sensor] ──┐
[Thermal Array]─┤
[Microphone] ───┼──> [ESP32 L1/L2] ──LoRa/Wire──> [Orange Pi L3] ──WiFi──> [ESP32 Deterrence]
[mmWave Radar]──┘      (ML Model)              (CNN Model)              (Actuators)
```

---

## 🔄 Workflow

1. **Level 1**: PIR detects motion → triggers Level 2
2. **Level 2**: Multi-sensor data → ML classification → if animal detected → trigger Level 3
3. **Level 3**: Camera capture → CNN analysis → species ID → if threat → trigger Deterrence
4. **Deterrence**: Activate appropriate response (light, sound, etc.)

---

## 📝 Documentation

- Technical docs in `docs/`
- Each subsystem has its own README
- Hardware specs and assembly guides in `hardware-design/`

---

## 🧪 Testing

- Unit tests in each firmware's `test/` directory
- Integration tests in `testing/integration/`
- Field test data logs in `testing/field-data/`
