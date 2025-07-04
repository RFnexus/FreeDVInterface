import time
import threading
import queue
import numpy as np
import pyaudio
from ctypes import *
import platform
import math
from threading import Lock
import subprocess
import RNS
from RNS.Interfaces.Interface import Interface

MODE_DATAC1 = 10
MODE_DATAC3 = 12
MODE_DATAC4 = 18

class FreeDVData:
    def __init__(self, mode):
        system = platform.system()
        libname = None
        self.debug = False

        if system == 'Windows':
            libname = 'libcodec2.dll'
        elif system == 'Linux':
            libname = 'libcodec2.so'
        elif system == 'Darwin':
            libname = 'libcodec2.dylib'

        lib_paths = [
            libname,
            f'./lib/{libname}',
            f'~/.reticulum/interfaces/lib/{libname}',
            f'/usr/local/lib/{libname}',
            f'/usr/lib/{libname}',
        ]

        loaded = False
        for path in lib_paths:
            try:
                self.c_lib = CDLL(path)
                loaded = True
                break
            except:
                continue

        if not loaded:
            raise Exception(f"Could not load FreeDV library. Tried paths: {lib_paths}")

        self.mode = mode

        self.c_lib.freedv_open.restype = POINTER(c_ubyte)
        self.freedv = self.c_lib.freedv_open(mode)

        self.c_lib.freedv_get_n_max_modem_samples.argtypes = [c_void_p]
        self.c_lib.freedv_get_n_max_modem_samples.restype = c_int

        self.c_lib.freedv_get_n_tx_modem_samples.argtypes = [c_void_p]
        self.c_lib.freedv_get_n_tx_modem_samples.restype = c_size_t

        self.c_lib.freedv_get_n_tx_preamble_modem_samples.argtypes = [c_void_p]
        self.c_lib.freedv_get_n_tx_preamble_modem_samples.restype = c_size_t

        self.c_lib.freedv_get_n_tx_postamble_modem_samples.argtypes = [c_void_p]
        self.c_lib.freedv_get_n_tx_postamble_modem_samples.restype = c_size_t

        self.c_lib.freedv_get_bits_per_modem_frame.argtypes = [c_void_p]
        self.c_lib.freedv_get_bits_per_modem_frame.restype = c_size_t

        self.c_lib.freedv_nin.argtypes = [c_void_p]
        self.c_lib.freedv_nin.restype = c_int

        self.c_lib.freedv_rawdatarx.argtypes = [c_void_p, c_void_p, c_void_p]
        self.c_lib.freedv_rawdatarx.restype = c_int

        self.c_lib.freedv_rawdatapreambletx.argtypes = [c_void_p, c_void_p]
        self.c_lib.freedv_rawdatapreambletx.restype = c_int

        self.c_lib.freedv_rawdatatx.argtypes = [c_void_p, c_void_p, c_void_p]
        self.c_lib.freedv_rawdatatx.restype = c_void_p

        self.c_lib.freedv_rawdatapostambletx.argtypes = [c_void_p, c_void_p]
        self.c_lib.freedv_rawdatapostambletx.restype = c_int

        self.c_lib.freedv_gen_crc16.argtypes = [c_void_p, c_size_t]
        self.c_lib.freedv_gen_crc16.restype = c_uint16

        self.c_lib.freedv_set_frames_per_burst.argtypes = [c_void_p, c_int]
        self.c_lib.freedv_set_frames_per_burst.restype = c_void_p

        self.c_lib.freedv_set_verbose.argtypes = [c_void_p, c_int]
        self.c_lib.freedv_set_verbose.restype = c_void_p

        self.c_lib.freedv_close.argtypes = [c_void_p]
        self.c_lib.freedv_close.restype = c_void_p

        self.c_lib.freedv_get_sync.argtypes = [c_void_p]
        self.c_lib.freedv_get_sync.restype = c_int

        self.c_lib.freedv_get_rx_status.argtypes = [c_void_p]
        self.c_lib.freedv_get_rx_status.restype = c_int

        self.c_lib.freedv_get_modem_sample_rate.argtypes = [c_void_p]
        self.c_lib.freedv_get_modem_sample_rate.restype = c_int

        self.c_lib.freedv_get_n_nom_modem_samples.argtypes = [c_void_p]
        self.c_lib.freedv_get_n_nom_modem_samples.restype = c_int

        self.c_lib.freedv_get_modem_stats.argtypes = [c_void_p, POINTER(c_int), POINTER(c_float)]
        self.c_lib.freedv_get_modem_stats.restype = None

        self.bytes_per_modem_frame = self.c_lib.freedv_get_bits_per_modem_frame(self.freedv) // 8
        self.payload_bytes_per_modem_frame = self.bytes_per_modem_frame - 2
        self.n_mod_out = self.c_lib.freedv_get_n_tx_modem_samples(self.freedv)
        self.n_tx_modem_samples = self.c_lib.freedv_get_n_tx_modem_samples(self.freedv)
        self.n_tx_preamble_modem_samples = self.c_lib.freedv_get_n_tx_preamble_modem_samples(self.freedv)
        self.n_tx_postamble_modem_samples = self.c_lib.freedv_get_n_tx_postamble_modem_samples(self.freedv)

        self.c_lib.freedv_set_frames_per_burst(self.freedv, 1)
        self.c_lib.freedv_set_verbose(self.freedv, 0)

        self.nin = self.get_freedv_rx_nin()

    def tx_burst(self, data_in):
        num_frames = math.ceil(len(data_in) / self.payload_bytes_per_modem_frame)

        if self.debug:
            RNS.log(f"FreeDV modem TX: {len(data_in)} bytes in {num_frames} frames", RNS.LOG_EXTREME)

        mod_out = create_string_buffer(self.n_tx_modem_samples * 2)
        mod_out_preamble = create_string_buffer(self.n_tx_preamble_modem_samples * 2)
        mod_out_postamble = create_string_buffer(self.n_tx_postamble_modem_samples * 2)

        # Preamble
        self.c_lib.freedv_rawdatapreambletx(self.freedv, mod_out_preamble)
        txbuffer = bytes(mod_out_preamble)

        # Create data frames
        for i in range(num_frames):
            # Main data buffer
            buffer = bytearray(self.payload_bytes_per_modem_frame)
            data_chunk = data_in[i * self.payload_bytes_per_modem_frame:(i + 1) * self.payload_bytes_per_modem_frame]
            buffer[:len(data_chunk)] = data_chunk

            # Add CRC16
            crc16 = c_ushort(self.c_lib.freedv_gen_crc16(bytes(buffer), self.payload_bytes_per_modem_frame))
            crc16 = crc16.value.to_bytes(2, byteorder='big')
            buffer += crc16

            data = (c_ubyte * self.bytes_per_modem_frame).from_buffer_copy(buffer)
            self.c_lib.freedv_rawdatatx(self.freedv, mod_out, data)
            txbuffer += bytes(mod_out)

        self.c_lib.freedv_rawdatapostambletx(self.freedv, mod_out_postamble)
        txbuffer += mod_out_postamble

        s_multiplier = 2

        silence_samples = int((100 / 1000) * 8000) * s_multiplier
        txbuffer += bytes(silence_samples)

        # extra_silence = self.c_lib.freedv_get_n_nom_modem_samples(self.freedv) * 2 * 2
        # txbuffer += bytes(extra_silence)

        return txbuffer

    def get_freedv_rx_nin(self):
        return self.c_lib.freedv_nin(self.freedv)

    def get_modem_sample_rate(self):
        return self.c_lib.freedv_get_modem_sample_rate(self.freedv)

    def get_n_nom_modem_samples(self):
        return self.c_lib.freedv_get_n_nom_modem_samples(self.freedv)

    def get_sync(self):
        sync = c_int()
        snr = c_float()
        self.c_lib.freedv_get_modem_stats(self.freedv, byref(sync), byref(snr))
        return sync.value

    def get_rx_status(self):
        return self.c_lib.freedv_get_rx_status(self.freedv)

    def rx(self, demod_in):
        bytes_out = create_string_buffer(self.bytes_per_modem_frame)
        nbytes_out = self.c_lib.freedv_rawdatarx(self.freedv, bytes_out, demod_in)
        self.nin = self.get_freedv_rx_nin()

        # get FreeDV modem stats
        sync = c_int()
        snr = c_float()
        self.c_lib.freedv_get_modem_stats(self.freedv, byref(sync), byref(snr))

        return nbytes_out, bytes_out[:nbytes_out], sync.value, snr.value

    def close(self):
        self.c_lib.freedv_close(self.freedv)


class AudioBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.int16)
        self.nbuffer = 0
        self.mutex = Lock()

    def push(self, samples):
        self.mutex.acquire()
        try:
            space_available = self.size - self.nbuffer
            if len(samples) > space_available:
                samples_to_drop = len(samples) - space_available
                self.buffer[:-samples_to_drop] = self.buffer[samples_to_drop:]
                self.nbuffer -= samples_to_drop

            self.buffer[self.nbuffer: self.nbuffer + len(samples)] = samples
            self.nbuffer += len(samples)
        finally:
            self.mutex.release()

    def pop(self, size):
        self.mutex.acquire()
        try:
            if size > self.nbuffer:
                size = self.nbuffer
            self.nbuffer -= size
            self.buffer[: self.nbuffer] = self.buffer[size: size + self.nbuffer]
        finally:
            self.mutex.release()

    def get_samples(self, size):
        self.mutex.acquire()
        try:
            if self.nbuffer >= size:
                return self.buffer[:size].copy()
            return None
        finally:
            self.mutex.release()


class FreeDVInterface(Interface):
    DEFAULT_IFAC_SIZE = 8

    def __init__(self, owner, configuration):
        super().__init__()

        # Parse configuration
        ifconf = Interface.get_config_obj(configuration)
        self.name = ifconf["name"]

        # Get configuration parameters
        self.input_device = int(ifconf["input_device"]) if "input_device" in ifconf else 0
        self.output_device = int(ifconf["output_device"]) if "output_device" in ifconf else 0
        self.freedv_mode_str = ifconf["freedv_mode"].lower() if "freedv_mode" in ifconf else "datac1"
        self.tx_volume = float(ifconf.get("tx_volume", 100)) / 100.0
        self.debug = ifconf.get("debug", "false").lower() == "true"

        # PTT configuration
        self.ptt_type = ifconf.get("ptt_type", "none").lower()  # none, serial, hamlib, vox
        self.ptt_enabled = self.ptt_type != "none"

        # serial PTT settings
        self.ptt_port = ifconf.get("ptt_port", None)
        self.ptt_serial_type = ifconf.get("ptt_serial_type", "DTR").upper()

        # Hamlib settings
        self.hamlib_model = ifconf.get("hamlib_model", "1")  # 1 = dummy/test
        self.hamlib_device = ifconf.get("hamlib_device", "/dev/ttyUSB0")
        self.hamlib_rigctl = ifconf.get("hamlib_rigctl", "rigctl")  # path to rigctl
        self.hamlib_speed = ifconf.get("hamlib_speed", "19200")
        self.hamlib_data_bits = ifconf.get("hamlib_data_bits", "8")
        self.hamlib_stop_bits = ifconf.get("hamlib_stop_bits", "1")
        self.hamlib_parity = ifconf.get("hamlib_parity", "N").upper()
        self.hamlib_extra_args = ifconf.get("hamlib_extra_args", "")
        self.hamlib_network = ifconf.get("hamlib_network", "false").lower() == "true"
        self.hamlib_host = ifconf.get("hamlib_host", "localhost")
        self.hamlib_port = ifconf.get("hamlib_port", "4532")

        # PTT timing
        self.ptt_on_delay = float(ifconf.get("ptt_on_delay", "0.1"))  # Delay after PTT on
        self.ptt_off_delay = float(ifconf.get("ptt_off_delay", "0.1"))  # Delay before PTT off
        self.ptt_tail_delay = float(ifconf.get("ptt_tail_delay", "0.1"))  # Delay after PTT off
        self.vox_delay = float(ifconf.get("vox_delay", "0.5"))  # Extra delay for VOX activation
        self.vox_tone = ifconf.get("vox_tone", "false").lower() == "true"  # Send tone for VOX activation

        # Set FreeDV mode
        if self.freedv_mode_str == "datac3":
            self.freedv_mode = MODE_DATAC3
            self.HW_MTU = 508  # temp
            self.bitrate = 250
            raise NotImplementedError

        elif self.freedv_mode_str == "datac4":
            self.freedv_mode = MODE_DATAC4
            self.HW_MTU = 508  # temp
            self.bitrate = 125
            raise NotImplementedError

        else:
            self.freedv_mode = MODE_DATAC1
            self.HW_MTU = 508  # temp
            self.bitrate = 980

        # Initialize properties
        self.online = False
        self.owner = owner
        self.audio_frames_per_buffer = 256
        self.is_transmitting = False

        # Channel state for CSMA
        self.channel_busy = False
        self.last_sync_time = 0
        self.sync_lock = Lock()
        self.signal_threshold = float(ifconf.get("signal_threshold", "0.1"))  # 0-1 normalized
        self.recent_signal_level = 0.0

        # CSMA settings
        self.csma_enabled = ifconf.get("csma_enabled", "true").lower() == "true"
        self.csma_wait_time = float(ifconf.get("csma_wait_time", "2.0"))  # wait time after channel busy
        self.channel_busy_timeout = float(
            ifconf.get("channel_busy_timeout", "0.5"))  # time after last activity before channel is clear
        self.signal_threshold = float(ifconf.get("signal_threshold", "0.1"))  # audio level threshold

        self.tx_queue = queue.Queue(maxsize=100)
        self.rx_queue = queue.Queue()

        # audio buffers
        self.rx_audio_buffer = AudioBuffer(self.audio_frames_per_buffer * 10000)
        self.tx_audio_buffer = AudioBuffer(self.audio_frames_per_buffer * 10000)

        self.running = True
        self.tx_thread = None
        self.rx_thread = None
        self.audio_thread = None
        self.monitor_thread = None

        # PTT handling
        self.ptt_serial = None
        if self.ptt_type == "serial" and self.ptt_port:
            try:
                import serial
                self.ptt_serial = serial.Serial(self.ptt_port)
                if self.debug:
                    RNS.log(f"Serial PTT enabled on {self.ptt_port}", RNS.LOG_DEBUG)
            except Exception as e:
                RNS.log(f"Could not open PTT port {self.ptt_port}: {e}", RNS.LOG_ERROR)
                self.ptt_enabled = False
        elif self.ptt_type == "hamlib":
            # Check if rigctl exists
            try:
                subprocess.run([self.hamlib_rigctl, "-V"], capture_output=True, timeout=2)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                RNS.log(f"rigctl not found or not working: {e}", RNS.LOG_ERROR)
                self.ptt_enabled = False
                return

            # Test hamlib connection
            if self.test_hamlib():
                RNS.log(f"Hamlib PTT enabled - Model: {self.hamlib_model}, Device: {self.hamlib_device}", RNS.LOG_INFO)
            else:
                RNS.log("Hamlib PTT test failed - check configuration", RNS.LOG_ERROR)
                self.ptt_enabled = False
        elif self.ptt_type == "vox":
            RNS.log(f"VOX PTT enabled with {self.vox_delay}s activation delay", RNS.LOG_INFO)

        # start our threads
        try:
            self.freedv = FreeDVData(self.freedv_mode)
            self.init_audio()
            self.start_threads()
            self.online = True

            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()

            RNS.log(f"FreeDV Interface [{self.name}] is online using {self.freedv_mode_str.upper()}", RNS.LOG_INFO)
            RNS.log(f"  MTU: {self.HW_MTU} bytes, Bitrate: ~{self.bitrate} bps", RNS.LOG_INFO)
            if self.debug:
                RNS.log(f"  Mode: {self.freedv}, Payload/frame: {self.freedv.payload_bytes_per_modem_frame} bytes",
                        RNS.LOG_DEBUG)

            if self.csma_enabled:
                RNS.log(f"  CSMA: Enabled (wait={self.csma_wait_time}s, timeout={self.channel_busy_timeout}s)",
                        RNS.LOG_INFO)
            else:
                RNS.log(f"  CSMA: Disabled", RNS.LOG_INFO)

            if self.ptt_enabled:
                if self.ptt_type == "hamlib":
                    if self.hamlib_network:
                        RNS.log(f"  PTT: Hamlib network mode ({self.hamlib_host}:{self.hamlib_port})", RNS.LOG_INFO)
                    else:
                        RNS.log(f"  PTT: Hamlib model {self.hamlib_model} on {self.hamlib_device}", RNS.LOG_INFO)
                elif self.ptt_type == "serial":
                    RNS.log(f"  PTT: Serial {self.ptt_serial_type} on {self.ptt_port}", RNS.LOG_INFO)
                elif self.ptt_type == "vox":
                    RNS.log(f"  PTT: VOX with {self.vox_delay}s delay", RNS.LOG_INFO)
        except Exception as e:
            RNS.log(f"Could not initialize FreeDV interface [{self.name}]: {e}", RNS.LOG_ERROR)
            raise e

    def init_audio(self):
        self.p = pyaudio.PyAudio()

        if self.debug:
            RNS.log(f"Initializing audio for FreeDV [{self.name}]", RNS.LOG_DEBUG)
            RNS.log(f"Input device: {self.input_device}, Output device: {self.output_device}", RNS.LOG_DEBUG)

        try:
            # Open audio stream
            self.stream = self.p.open(
                rate=8000,
                channels=1,
                format=pyaudio.paInt16,
                frames_per_buffer=self.audio_frames_per_buffer,
                input=True,
                output=True,
                input_device_index=self.input_device,
                output_device_index=self.output_device,
                stream_callback=self.audio_callback
            )

            self.stream.start_stream()

            if not self.stream.is_active():
                raise Exception("Audio stream failed to start - check configuration")

            if self.debug:
                RNS.log(f"Audio stream started  [{self.name}]", RNS.LOG_DEBUG)

        except Exception as e:
            RNS.log(f"Failed audio | [{self.name}]: {e}", RNS.LOG_ERROR)
            raise

    def audio_callback(self, in_data, frame_count, time_info, status):
        try:
            # always capture input audio
            samples_int16 = np.frombuffer(in_data, dtype=np.int16)
            self.rx_audio_buffer.push(samples_int16)

            # check if we have TX audio to send
            if self.tx_audio_buffer.nbuffer >= frame_count:
                tx_samples = self.tx_audio_buffer.buffer[:frame_count].copy()
                self.tx_audio_buffer.pop(frame_count)

                # TODO
                tx_samples = (tx_samples * self.tx_volume).astype(np.int16)
                return tx_samples.tobytes(), pyaudio.paContinue

            return b'\x00' * (frame_count * 2), pyaudio.paContinue

        except Exception as e:
            RNS.log(f"Audio callback error [{self.name}]: {e}", RNS.LOG_ERROR)
            return b'\x00' * (frame_count * 2), pyaudio.paContinue

    def start_threads(self):
        """Start processing threads"""
        self.tx_thread = threading.Thread(target=self.tx_loop, daemon=True)
        self.rx_thread = threading.Thread(target=self.rx_loop, daemon=True)

        self.tx_thread.start()
        self.rx_thread.start()

    def build_hamlib_cmd(self):
        cmd = [self.hamlib_rigctl]

        if self.hamlib_network:
            # network mode
            cmd.extend(["-m", "2"])  # 2 = NET rigctl
            cmd.extend(["-r", f"{self.hamlib_host}:{self.hamlib_port}"])
        else:
            # direct serial mode
            cmd.extend(["-m", self.hamlib_model])
            cmd.extend(["-r", self.hamlib_device])
            cmd.extend(["-s", self.hamlib_speed])
            cmd.extend(["-C", f"data_bits={self.hamlib_data_bits}"])
            cmd.extend(["-C", f"stop_bits={self.hamlib_stop_bits}"])
            cmd.extend(["-C", f"serial_parity={self.hamlib_parity}"])

        if self.hamlib_extra_args:
            cmd.extend(self.hamlib_extra_args.split())

        return cmd

    def test_hamlib(self):
        try:
            cmd = self.build_hamlib_cmd()
            cmd.append("f")

            if self.debug:
                RNS.log(f"Testing hamlib with command: {' '.join(cmd)}", RNS.LOG_DEBUG)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if self.debug and result.returncode != 0:
                RNS.log(f"Hamlib test output: {result.stderr}", RNS.LOG_DEBUG)

            return result.returncode == 0
        except Exception as e:
            if self.debug:
                RNS.log(f"Hamlib test failed: {e}", RNS.LOG_ERROR)
            return False

    def ptt_on(self):
        if not self.ptt_enabled:
            return

        try:
            if self.ptt_type == "vox":
                # For VOX, just add extra delay for VOX circuit to activate
                time.sleep(self.vox_delay)
            elif self.ptt_type == "serial" and self.ptt_serial:
                if self.ptt_serial_type == "DTR":
                    self.ptt_serial.dtr = True
                elif self.ptt_serial_type == "RTS":
                    self.ptt_serial.rts = True
                time.sleep(self.ptt_on_delay)
            elif self.ptt_type == "hamlib":
                cmd = self.build_hamlib_cmd()
                cmd.extend(["T", "1"])

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)

                if result.returncode != 0 and self.debug:
                    RNS.log(f"Hamlib PTT ON failed: {result.stderr}", RNS.LOG_ERROR)

                time.sleep(self.ptt_on_delay)

                if self.debug:
                    RNS.log(f"Hamlib PTT ON", RNS.LOG_DEBUG)
        except Exception as e:
            RNS.log(f"PTT control error: {e}", RNS.LOG_ERROR)

    def ptt_off(self):
        if not self.ptt_enabled:
            return

        if self.ptt_type == "vox":
            #
            return

        try:
            time.sleep(self.ptt_off_delay)  # PTT delay before off

            if self.ptt_type == "serial" and self.ptt_serial:
                if self.ptt_serial_type == "DTR":
                    self.ptt_serial.dtr = False
                elif self.ptt_serial_type == "RTS":
                    self.ptt_serial.rts = False
            elif self.ptt_type == "hamlib":
                cmd = self.build_hamlib_cmd()
                cmd.extend(["T", "0"])  # PTT off command

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)

                if result.returncode != 0 and self.debug:
                    RNS.log(f"Hamlib PTT OFF failed: {result.stderr}", RNS.LOG_ERROR)

                if self.debug:
                    RNS.log(f"Hamlib PTT OFF", RNS.LOG_DEBUG)
        except Exception as e:
            RNS.log(f"PTT control error: {e}", RNS.LOG_ERROR)

    def monitor_loop(self):
        last_log = 0
        while self.running:
            try:
                now = time.time()
                if now - last_log > 30 and self.debug:
                    RNS.log(f"FreeDV [{self.name}] status: TX queue={self.tx_queue.qsize()}, " +
                            f"RX buffer={self.rx_audio_buffer.nbuffer}, TX buffer={self.tx_audio_buffer.nbuffer}, " +
                            f"Transmitting={self.is_transmitting}, Channel busy={self.channel_busy}", RNS.LOG_DEBUG)
                    last_log = now

                # check if our audio stream is still active
                if hasattr(self, 'stream') and self.stream and not self.stream.is_active():
                    RNS.log(f"Audio stream died on FreeDV [{self.name}], attempting restart", RNS.LOG_ERROR)
                    self.online = False
                    break

                time.sleep(1)
            except Exception as e:
                RNS.log(f"Monitor error in FreeDV [{self.name}]: {e}", RNS.LOG_ERROR)

    def update_channel_state(self, has_sync, signal_level=None):
        with self.sync_lock:
            if signal_level is not None:
                self.recent_signal_level = signal_level

            # channel is busy if we have sync OR signal level is above threshold
            if has_sync or (self.recent_signal_level > self.signal_threshold):
                self.channel_busy = True
                self.last_sync_time = time.time()
                if self.debug and (has_sync or self.recent_signal_level > self.signal_threshold):
                    reason = "sync detected" if has_sync else f"signal level {self.recent_signal_level:.3f}"
                    RNS.log(f"FreeDV [{self.name}] channel busy - {reason}", RNS.LOG_DEBUG)
            else:
                # channel is considered free if no sync/signal above threshold for timeout period
                if time.time() - self.last_sync_time > self.channel_busy_timeout:
                    if self.channel_busy and self.debug:
                        RNS.log(f"FreeDV [{self.name}] channel now clear", RNS.LOG_DEBUG)
                    self.channel_busy = False

    def is_channel_clear(self):
        with self.sync_lock:
            # also check if RX buffer has significant data (someone might be transmitting)
            rx_buffer_busy = self.rx_audio_buffer.nbuffer > (self.audio_frames_per_buffer * 10)

            if rx_buffer_busy and self.debug:
                RNS.log(f"FreeDV [{self.name}] channel busy - RX buffer has {self.rx_audio_buffer.nbuffer} samples",
                        RNS.LOG_DEBUG)

            return not self.channel_busy and not rx_buffer_busy

    def wait_for_clear_channel(self):
        if not self.csma_enabled:
            return True

        # check if channel is already clear
        if self.is_channel_clear():
            if self.debug:
                RNS.log(f"FreeDV [{self.name}] channel clear, transmitting", RNS.LOG_DEBUG)
            return True

        # Channel is busy, wait for it to clear
        RNS.log(f"FreeDV [{self.name}] channel busy, waiting...", RNS.LOG_DEBUG)
        wait_start = time.time()
        last_log = wait_start

        while not self.is_channel_clear():
            time.sleep(0.1)  # Check every 100ms

            # log every 5 seconds while waiting
            if time.time() - last_log > 5.0:
                wait_time = time.time() - wait_start
                RNS.log(f"FreeDV [{self.name}] still waiting for clear channel ({wait_time:.1f}s)", RNS.LOG_DEBUG)
                last_log = time.time()

        time.sleep(self.csma_wait_time)

        # check one more time that channel is still clear
        if not self.is_channel_clear():
            return self.wait_for_clear_channel()

        total_wait = time.time() - wait_start
        if total_wait > 0.5:
            RNS.log(f"channel clear after {total_wait:.1f}s, transmitting", RNS.LOG_DEBUG)

        return True

    def tx_loop(self):
        while self.running:
            try:
                packet = self.tx_queue.get(timeout=0.1)

                self.wait_for_clear_channel()
                self.is_transmitting = True

                max_payload = self.freedv.payload_bytes_per_modem_frame
                if len(packet) > max_payload:
                    RNS.log(f"FreeDV [{self.name}] packet too large: {len(packet)} > {max_payload}",
                            RNS.LOG_ERROR)
                    self.is_transmitting = False
                    continue

                if self.debug:
                    RNS.log(f"FreeDV [{self.name}] TX: {len(packet)} bytes",
                            RNS.LOG_DEBUG)

                self.ptt_on()

                # For VOX with tone enabled send a short tone to trigger VOX
                if self.ptt_type == "vox" and self.vox_tone:
                    # 100ms of 1kHz tone at 8kHz sample rate
                    tone_duration = 0.1
                    sample_rate = 8000
                    frequency = 1000
                    t = np.linspace(0, tone_duration, int(sample_rate * tone_duration))
                    tone = (np.sin(2 * np.pi * frequency * t) * 32767 * 0.3).astype(np.int16)
                    self.tx_audio_buffer.push(tone)
                    time.sleep(tone_duration + 0.1)

                tx_audio = self.freedv.tx_burst(packet)

                self.tx_audio_buffer.nbuffer = 0
                self.tx_audio_buffer.push(np.frombuffer(tx_audio, dtype=np.int16))

                # calculate expected time
                samples_to_tx = len(tx_audio) // 2  # 16-bit samples
                expected_time = samples_to_tx / 8000.0  # 8kHz sample rate

                time.sleep(expected_time + 0.5)

                # clear any remaining samples
                self.tx_audio_buffer.nbuffer = 0

                # disable PTT
                self.ptt_off()

                # add PTT tail delay
                time.sleep(self.ptt_tail_delay)

                # clear transmitting flag
                self.is_transmitting = False

                if self.debug:
                    RNS.log(f"FreeDV [{self.name}] transmission complete", RNS.LOG_DEBUG)

            except queue.Empty:
                continue
            except Exception as e:
                self.is_transmitting = False
                RNS.log(f"TX error in FreeDV interface [{self.name}]: {e}", RNS.LOG_ERROR)
                import traceback
                RNS.log(traceback.format_exc(), RNS.LOG_ERROR)

    def rx_loop(self):
        while self.running:
            try:
                # get our samples for demodulation
                nin = self.freedv.nin
                samples = self.rx_audio_buffer.get_samples(nin)

                if samples is not None:
                    self.rx_audio_buffer.pop(nin)

                    signal_level = np.sqrt(np.mean(samples.astype(float) ** 2)) / 32768.0

                    nbytes_out, rx_bytes, sync_state, snr_value = self.freedv.rx(samples.tobytes())

                    self.update_channel_state(sync_state > 0, signal_level)

                    if nbytes_out > 0:
                        if self.debug:
                            RNS.log(f"FreeDV [{self.name}] raw RX: {nbytes_out} bytes", RNS.LOG_DEBUG)

                        # For FreeDV, we get the full frame including CRC
                        # The CRC is the last 2 bytes. but FreeDV already validates it
                        # If were here, the CRC was good, so we can use the payload

                        if len(rx_bytes) >= 2:
                            # remove the CRC (last 2 bytes)
                            payload = rx_bytes[:-2]

                            actual_length = len(payload)
                            while actual_length > 0 and payload[actual_length - 1] == 0:
                                actual_length -= 1

                            if actual_length > 0:
                                packet = payload[:actual_length]
                                self.process_incoming(bytes(packet))
                                if self.debug:
                                    RNS.log(f"FreeDV [{self.name}] processed {len(packet)} byte packet",
                                            RNS.LOG_DEBUG)
                            elif self.debug:
                                RNS.log(f"FreeDV [{self.name}] received empty/padding-only frame",
                                        RNS.LOG_DEBUG)
                        else:
                            if self.debug:
                                RNS.log(f"FreeDV",
                                        RNS.LOG_DEBUG)

                else:
                    # no samples available, sleep a bit and update channel state
                    time.sleep(0.01)
                    self.update_channel_state(False, self.recent_signal_level * 0.95)

            except Exception as e:
                RNS.log(f"RX error in FreeDV interface [{self.name}]: {e}", RNS.LOG_ERROR)

    def process_incoming(self, data):
        self.rxb += len(data)
        if self.debug:
            RNS.log(f"FreeDV [{self.name}] received {len(data)} byte packet (total: {self.rxb} bytes)", RNS.LOG_DEBUG)
        self.owner.inbound(data, self)

    def process_outgoing(self, data):
        if self.online:
            max_payload = self.freedv.payload_bytes_per_modem_frame

            if len(data) > max_payload:
                return

            try:
                # Add packet to queue - it will be sent when channel is clear
                self.tx_queue.put(data, timeout=1.0)
                self.txb += len(data)

                if self.csma_enabled and self.channel_busy and self.debug:
                    RNS.log(
                        f"FreeDV [{self.name}] queued packet #{self.tx_queue.qsize()}, will send when channel is clear",
                        RNS.LOG_DEBUG)
            except queue.Full:
                RNS.log(f"TX queue full on FreeDV interface [{self.name}]", RNS.LOG_WARNING)

    def should_ingress_limit(self):
        return False

    def get_hash(self):
        import hashlib
        return hashlib.sha256(f"FreeDVInterface.{self.name}".encode()).digest()

    def close(self):
        self.online = False
        self.running = False

        # Ckear any pending transmissions
        self.is_transmitting = False

        if self.tx_thread:
            self.tx_thread.join(timeout=2)
        if self.rx_thread:
            self.rx_thread.join(timeout=2)
        if hasattr(self, 'monitor_thread') and self.monitor_thread:
            self.monitor_thread.join(timeout=2)

        if hasattr(self, 'stream') and self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except:
                pass

        if hasattr(self, 'p'):
            self.p.terminate()

        if hasattr(self, 'freedv'):
            self.freedv.close()

        if self.ptt_serial:
            self.ptt_serial.close()

        if self.debug:
            RNS.log(f"FreeDV Interface [{self.name}] closed", RNS.LOG_DEBUG)


# Register
interface_class = FreeDVInterface

# Helper script to list audio devices and hamlib radios when run directly
if __name__ == "__main__":
    import sys

    print("FreeDV Interface for Reticulum")
    print("=" * 60)

    print("\nAvailable audio devices:")
    print("-" * 50)
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']}")
        print(f"  Channels: {info['maxInputChannels']} in, {info['maxOutputChannels']} out")
        print(f"  Sample Rate: {info['defaultSampleRate']}")
    p.terminate()

    print("\nHamlib radio models")
    print("-" * 50)
    try:
        result = subprocess.run(["rigctl", "--list"], capture_output=True, text=True)
    except:
        print("rigctl not found - install hamlib for radio control (view README for install instructions)")

    print("\nConfiguration example:")
    print("-" * 50)
    print("""

  [[FreeDV with IC-7300]]
    type = FreeDVInterface
    enabled = yes
    input_device = 2
    output_device = 2
    freedv_mode = datac1
    ptt_type = hamlib
    hamlib_model = 3073 # run rigctl -l to list devices
    hamlib_device = /dev/ttyUSB0
    hamlib_speed = 19200
""")