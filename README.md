FreeDVInterface 
-
This is a CustomInterface for Reticulum that provides a plug and play, cross-platform sound modem interface for HF radios using the FreeDV API.
It supports VOX, serial, and Hamilb PTT. The interface also provides highly configurable options for collision avoidance. 

⚠ Work in progress

Setup:
-

### Raspberry Pi (Zero, 4, 5) / Debian / Ubuntu


1. Build and install codec2:
```
sudo apt install git build-essential cmake
git clone https://github.com/drowe67/codec2
cd codec2
mkdir build_linux
cd build_linux
cmake ..
make
```

2. Install pre-requisites 
```
sudo apt install portaudio19-dev
pip install pyaudio OR sudo apt install python-pyaudio 
pip install numpy OR sudo apt install python-numpy

```
3. (Optional) install Hamlib for rigctl PTT support 
```
sudo apt install libhamlib-utils
```

4. Move FreeDVInterface.py to the "interfaces" folder in your Reticulum install location  
```
git clone https://github.com/RFnexus/FreeDVInterface.git
cd FreeDVInterface
mv FreeDVInterface.py ~/.reticulum/interfaces
```
5. Add a FreeDVInterface to your Reticulum config 

An example config looks like this. Here we are using `rigctld` to key on / off the PTT on a ICOM-7300. 

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

### Windows

`TODO`

### Fedora

`TODO`

---

## Troubleshooting
> I keep seeing "Invalid sample rate"

Ensure that you have the right audio devices configured

>I'm using a single board computer like the Raspberry Pi Zero and keep seeing "ALSA lib pcm.c:8526:(send_pcm_recover) underrun occured"

Ensure that your Pi / other SBC is getting enough power and that there are no issues with the USB soundcard or other USB device connected to your radio
>



