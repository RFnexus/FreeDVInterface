FreeDVInterface 
-
This is a CustomInterface for Reticulum that provides a plug and play, cross-platform sound modem interface for HF radios using the FreeDV API.
It supports VOX, serial, and Hamilb PTT. The interface also provides highly configurable options for collision avoidance. 

⚠ Work in progress

Installation:
-
### 1. Install codec2
https://github.com/drowe67/codec2?tab=readme-ov-file#quickstart


Debian / Ubuntu 

TODO

Raspberry Pi

TODO

Windows

TODO

Fedora

TODO

### 2. Install Python dependencies 

pip install pyaudio numpy

### 3. (Optional) Install hamlib for rigctl PTT support



### 4. Add FreeDVInterface.py to your Reticulum "interfaces" folder 


---

Setup and getting on the air
-
Example config with an ICOM-7300 and using rigctl over network with rigctld
```
[[FreeDVInterface IC-7300]]
    type = FreeDVInterface
    interface_enabled = True
    input_device = 2
    output_device = 2
    freedv_mode = datac1
    tx_volume = 100
    ptt_type = hamlib
    hamlib_network = True
    hamlib_host = localhost
    hamlib_port = 4532
    csma_enabled = True
```

## Troubleshooting
> I keep seeing "Invalid sample rate"

Ensure that you have the right audio devices configured

>I'm using a single board computer like the Raspberry Pi Zero and keep seeing "ALSA lib pcm.c:8526:(send_pcm_recover) underrun occured"

Ensure that your Pi / other SBC is getting enough power and that there are no issues with the USB soundcard or other USB device connected to your radio
>



