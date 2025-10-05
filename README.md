<!-- PROJECT SHIELDS -->
<a name="readme-top"></a>
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/wanghley)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/wanghley/TerraWatch-IoT">
    <img src="image.gif" alt="Logo" width="280">
  </a>

  <h3 align="center">TerraWatch-IoT (sic Agronauts)</h3>

  <p align="center">
    IoT-based intelligent pest detection and deterrence system for agricultural protection
    <br />
    <a href="#"><strong>Explore the code »</strong></a>
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

TerraWatch-IoT is a comprehensive, energy-efficient Internet-of-Things (IoT) pest detection and deterrence system specifically designed to protect 0.6m x 3m crop rows through intelligent, multi-level sensing and automatic deterrence mechanisms. 

This fullstack IoT agricultural protection system employs a three-level detection hierarchy that balances power consumption with accuracy:

**Level 1 - Motion Detection:** Three passive infrared (PIR) sensors arranged in overlapping 90-degree coverage zones provide low-power motion detection as an initial trigger.

**Level 2 - Intelligent Classification:** Thermal array sensors, microphones, and mmWave radar sensors feed data to a lightweight machine learning model running on an ESP32 microcontroller for real-time classification of detected movement into categories such as raccoons, squirrels, other animals, and humans. When classification confidence exceeds 90%, the system bypasses the power-intensive Level 3 stage.

**Level 3 - Image Verification:** CNN-based analysis on an Orange Pi 4A provides final verification when needed for lower-confidence detections.

**Deterrent Subsystem:** Upon animal detection, the system coordinates multiple response mechanisms including:
- Floodlight illuminating a reflective-material-enhanced scarecrow
- Perimeter strobing lights creating predator-eye patterns
- Randomized predator call playback through a speaker system

With an average of 17.6W consumption and a 320Wh battery, the system provides 18 hours of full battery autonomy, making it a practical solution for remote agricultural deployments.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

<img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="cpp" style="vertical-align:top; margin:4px"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="python" style="vertical-align:top; margin:4px"> <img src="https://img.shields.io/badge/ESP32-000000?style=for-the-badge&logo=espressif&logoColor=white" alt="esp32" style="vertical-align:top; margin:4px"> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="tensorflow" style="vertical-align:top; margin:4px"> <img src="https://img.shields.io/badge/PlatformIO-FF7F00?style=for-the-badge&logo=platformio&logoColor=white" alt="platformio" style="vertical-align:top; margin:4px">

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get started with TerraWatch-IoT, follow the instructions below:

### Prerequisites

* [PlatformIO](https://platformio.org/)
* [Python 3.8+](https://www.python.org/)
* ESP32 Development Board
* Orange Pi 4A (for Level 3 processing)

### Hardware Requirements

* 3x PIR Motion Sensors
* AMG8833 Thermal Array Sensor
* mmWave Radar Sensor
* Microphone Module
* Camera Module
* LED Floodlight and Strobe Lights
* Speaker System
* 320Wh Battery Pack
* Power Management Board

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/wanghley/TerraWatch-IoT.git
   ```
2. Install PlatformIO dependencies
   ```sh
   cd TerraWatch-IoT
   platformio lib install
   ```
3. Upload firmware to ESP32
   ```sh
   platformio run --target upload
   ```
4. Set up Orange Pi 4A
   ```sh
   cd orange-pi-setup
   pip install -r requirements.txt
   python setup.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->
## Usage

TerraWatch-IoT provides an intelligent, multi-tiered approach to pest detection and deterrence in agricultural settings. The system operates autonomously to protect crop rows.

### System Operation

The system operates in three detection levels:

**Level 1: Motion Detection**
- PIR sensors continuously monitor the protected area
- Low power consumption (~50mW per sensor)
- Triggers Level 2 upon motion detection

**Level 2: ML Classification**
- ESP32 processes data from thermal array, microphone, and mmWave radar
- Lightweight ML model classifies detected entities
- Categories: Raccoon, Squirrel, Other Animals, Humans
- When confidence > 90%, activates deterrent without Level 3

**Level 3: Image Verification**
- Orange Pi 4A performs CNN-based image analysis
- Activated only for low-confidence detections (<90%)
- Provides final classification verification

### Deterrent Activation

Upon detecting an animal pest (not human):
1. Floodlight activates, illuminating reflective scarecrow
2. Perimeter strobing lights create predator-eye patterns
3. Random predator calls play through speakers
4. System logs event with timestamp and classification

### Configuration

Edit `config.h` to customize:
- Detection sensitivity thresholds
- Classification confidence levels
- Deterrent activation patterns
- Power management settings
- Network connectivity options

### Monitoring

Access real-time system status:
```bash
# Via serial monitor
platformio device monitor

# Via web interface (if WiFi enabled)
http://[device-ip]:8080
```

### Power Management

- Average consumption: 17.6W
- Battery capacity: 320Wh
- Autonomy: ~18 hours on battery
- Solar panel integration supported

### Use Cases

- **Small-Scale Farms:** Protect individual crop rows from nocturnal pests
- **Community Gardens:** Automated pest control without harmful chemicals
- **Research Facilities:** Monitor and study animal behavior around crops
- **Organic Farming:** Non-lethal pest deterrence for organic certification

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Implement basic cache simulation
- [x] Add support for different cache policies
- [ ] Optimize code for performance
-

 [ ] Explore additional features based on user feedback

See the [open issues](https://github.com/your_username/CacheSimulator/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Wanghley Soares Martins - [@wanghley](https://instagram.com/wanghley) - wanghley@wanghley.com

Project Link: [https://github.com/wanghley/TerraWatch-IoT](https://github.com/wanghley/TerraWatch-IoT)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/wanghley/TerraWatch-IoT?style=for-the-badge
[contributors-url]: https://github.com/wanghley/TerraWatch-IoT/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wanghley/TerraWatch-IoT.svg?style=for-the-badge
[forks-url]: https://github.com/wanghley/TerraWatch-IoT/network/members
[stars-shield]: https://img.shields.io/github/stars/wanghley/TerraWatch-IoT.svg?style=for-the-badge
[stars-url]: https://github.com/wanghley/TerraWatch-IoT/stargazers
[issues-shield]: https://img.shields.io/github/issues/wanghley/TerraWatch-IoT.svg?style=for-the-badge
[issues-url]: https://github.com/wanghley/TerraWatch-IoT/issues
[license-shield]: https://img.shields.io/github/license/wanghley/TerraWatch-IoT.svg?style=for-the-badge
[license-url]: https://github.com/wanghley/TerraWatch-IoT/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/wanghley
[product-screenshot]: images/screenshot.png
```