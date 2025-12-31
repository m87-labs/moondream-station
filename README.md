<div>
   <p align="center">
   <img src="https://raw.githubusercontent.com/m87-labs/moondream-station/55878b16dd3a675a9ccf9a292975aee97d055beb/assets/md_logo_clean.png" alt="Moondream Station Logo" width="200"/>
   </p>

   <h3 align="center"><strong>Moondream Station: Local Visual Reasoning</strong></h3>

   <p align="center">
      <a href="https://moondream.ai/station" target="_blank"><img src="https://img.shields.io/badge/Home-%F0%9F%8F%A0-blue?style=flat-square" alt="Home Page"></a>
      <a href="https://discord.gg/QTaWPdDZ" target="_blank"><img src="https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white&style=flat-square" alt="Discord"></a>
      <a href="https://x.com/moondreamai" target="_blank"><img src="https://img.shields.io/badge/follow-%40moondreamai-000000?style=flat-square&logo=x&logoColor=white" alt="Follow @moondreamai"></a>
      <a href="LICENSE" target="_blank"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License"></a>
   </p>
</div>

<hr style="height:3px;border:none;background:#e0e0e0;margin:24px 0;">

<table align="center">
<tr>
<td width="420" align="center" valign="middle">

<!-- Demo video -->
![Video showing Moondream Station running in a terminal](https://raw.githubusercontent.com/m87-labs/moondream-station/55878b16dd3a675a9ccf9a292975aee97d055beb/assets/md_station_demo.gif)

</td>
<td width="400" align="left" valign="middle">

### How It Works

🚀 **Launches Local Server**  
   All inference runs on your device

🔧 **Control via CLI**  
   Caption images, answer questions, and manage settings

🌐 **Access via HTTP**  
   Connect to `http://localhost:2020/v1` through REST or our [Python](https://pypi.org/project/moondream/), [Node](https://www.npmjs.com/package/moondream), or [OpenAI](https://github.com/openai/openai-python) client

</td>
</tr>
</table>

## Installation

Install from PyPI:
```bash
$ pip install moondream-station
```

Install from source:
```bash
$ git clone https://github.com/m87-labs/moondream-station.git
$ cd moondream-station
$ pip install -e .
```
That's it! Moondream Station will automatically set itself up.

## Usage

### Launch Moondream Station

To fire up Moondream Station, execute this command in your terminal:
```
$ moondream-station
```

### Launch Moondream Station using Docker

To fire up Moondream Station via Docker, execute this command in your terminal:
```
$ docker build -t moondream-cpu:latest -f Dockerfile.cpu .
```

Run the above docker image as container in interactive way (For CPU)
```
$ docker run -it -p 2020:2020 moondream-cpu:latest
```

### Model Management
By default, Moondream Station uses the latest model your machine supports. If you want to view or activate other Moondream models, use the following commands:
- `models` - List available models
- `models switch <model>` - Switch to a model

### Service Control
We like to think Moondream has 20/20 vision; that’s why, by default, we launch Moondream Station on port 2020. If that port is taken, Moondream Station will try a nearby free port. Additionally, you can control the port and status of the inference service with the following commands:
- `start [port]` - Start REST server (default: port 2020)
- `stop` - Stop server
- `restart` - Restart server

### Inference
**Access via HTTP**: 
Point any of our inference clients at your Moondream Station; for example, with our python client you can do:

```python
import moondream as md
from PIL import Image

# connect to Moondream Station
model = md.vl(endpoint="http://localhost:2020/v1")

# Load an image
image = Image.open("path/to/image.jpg")

# Ask a question
answer = model.query(image, "What's in this image?")["answer"]
print("Answer:", answer)
```
For more information on our clients visit: [Python](https://pypi.org/project/moondream/), [Node](https://www.npmjs.com/package/moondream), [Quick Start](https://moondream.ai/c/docs/quickstart)

**Connect via CLI**: 
Use all the capabilities of Moondream directly through your terminal. No need to touch any code!

- `infer <function> [args]` - Run single inference
- `inference` - Enter interactive inference mode

### Settings
Control the number of workers, queue size, privacy settings, and more through Settings:

- `settings` - Show configuration
- `settings set <key> <value>` - Set setting value

Moondream Station collects anonymous usage metrics to help us improve the app. The following data is collected:

- **Event data**: when you use features like caption, query, detect, or point.
- **Version information**: active bootstrap, hypervisor, inference client, and model version.
- **System information**: OS version, IP address, and Python version/runtime.

No personal information, images, or prompts/responses are ever collected. To opt out of logging, run: `settings set logging false`.

### Utility
The utility functions provide insight into what Moondream Station is currently doing. To view statistics for your current session, use the `session` mode. To view a log of requests processed by Moondream Station, use the `history` command.

- `session` - Show session stats
- `help` - Show available commands
- `history` - Show command history
- `reset` - Reset app data & settings
- `clear` - Clear screen
- `exit` - Quit application