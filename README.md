<div>
   <p align="center">
   <img src="assets/md_logo_clean.png" alt="Moondream Station Logo" width="200"/>
   </p>

   <h3 align="center"><strong>Moondream Station: 100% free local visual reasoning</strong></h3>

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
![Video showing Magnitude tests running in a terminal and agent taking actions in the browser](assets/md_station_demo.gif)

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

## Getting Started

### Installation

1. **Download** Moondream Station using this terminal command:
   ```
   $ curl https://depot.moondream.ai/station/install.sh | bash
   ```

2. **Launch** by double-clicking the newly created "Moondream Station" app. On Mac, if the app does not automatically appear in finder,
look under `~/Applications ` or `~/Downloads`

That's it! Moondream Station will automatically set itself up.

> **Note**: Currently only supports **macOS 15+** and **Ubuntu 22+**. Windows support coming soon.

### Basic Usage

To interact with a running Moondream Station, use the CLI in either of the two modes:

1. **REPL (Interactive Shell) Mode**: Start an interactive session where you can issue multiple commands
   ```
   $ moondream
   ```

2. **Command Mode**: Execute a single command directly
   ```
   $ moondream caption path/to/image.jpg
   ```

## Inference

**Connect via CLI**: 
Use all the capabilities of Moondream directly through your terminal. No need to touch any code!

```
moondream> query "What's in this image?" path/to/image.jpg
```

**Access via HTTP**: 
Point any of our inference clients at your Moondream Station. For example, with our python client you can do:

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

### Docker Inference only Server
To run an inference only server in a docker container. The following command builds and runs the inference only docker container:
```
cd app/
make inference-server
```

Make sure you have the build dependencies installed before running the command (make).
*(Note: on ubuntu, you can install make with `sudo apt install build-essential`)*

## Administrative Commands

Access administrative functions using the `admin` command:

### Model Management

List all available models
```
moondream> admin model-list
```

Switch to a specific model version
```
moondream> admin model-use v2025.4.14 --confirm
```

### Updates

Check if updates are available
```
moondream> admin check-updates
```

Update Moondream Station to the latest version
```
moondream> admin update --confirm
```

## Reported Metrics

Moondream Station collects anonymous usage metrics to help us improve the app. The following data is collected:

- **Event data**: When you use features like caption, query, detect, or point
- **Version information**: Active bootstrap, hypervisor, inference client, and model version
- **System information**: OS version, IP address, and Python version/runtime

No personal information, image, or prompt/response is ever collected.

### Check Metrics Status

To check if metrics reporting is enabled:

```
moondream> admin get-config
```

Look for the `"metrics_reporting": true` or `"metrics_reporting": false` entry in the output.

### Opt Out of Metrics

You can toggle metrics reporting on/off using the CLI:

```
moondream> admin toggle-metrics --confirm
```

The command will return the new state (true/false) after toggling. This action is stateful and will persist even after Moondream Station is closed.

## FAQ

**Q: How do I know if Moondream Station is running?**  
A: Use the `health` command: `moondream> health`.

**Q: I get "Could not connect to server" errors. What should I do?**  
A: Ensure Moondream Station is running. The CLI expects it to be available at http://localhost:2020 by default.

**Q: What's the hardware requirement?**  
A: Moondream Station currently runs on any Apple Silicon Mac with macOS 15+. Linux and Windows are coming soon.
