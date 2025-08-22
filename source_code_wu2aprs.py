
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WU2APRS 2.1 — Hotkeys (W,S,R,I)
- W : send WX now (IS if ON, RF/AFSK if ON)
- S : send Status now
- R : toggle RF (AFSK) ON/OFF at runtime
- I : toggle IS (APRS-IS) ON/OFF at runtime

Other features kept:
- Payload preview logs, device/meta controls, No-Op preamble, channel routing, WAV mirror
- AFSK-only or IS-only operation supported
"""

import os, sys, time, socket, wave, math, threading
import configparser
from datetime import datetime, timezone

import requests

try:
    import numpy as np
except Exception:
    np = None

# =========================
#   Helpers / Formatting
# =========================

def to_aprs_latlon(lat, lon):
    """decimal degrees → APRS ddmm.mmN/dddmm.mmE + symbol '_' (weather station)"""
    def dm(v, lat=False):
        sign = 1 if v >= 0 else -1
        v = abs(v)
        deg = int(v)
        minutes = (v - deg) * 60
        if lat:
            return f"{deg:02d}{minutes:05.2f}", ('N' if sign >= 0 else 'S')
        else:
            return f"{deg:03d}{minutes:05.2f}", ('E' if sign >= 0 else 'W')

    lat_dm, ns = dm(lat, lat=True)
    lon_dm, ew = dm(lon, lat=False)
    return f"{lat_dm}{ns}/{lon_dm}{ew}_"  # '/' + '_' (weather station)

def mph_to_knots(mph):
    return int(round(float(mph) * 0.868976)) if mph is not None else 0

def inhg_to_tenths_mbar(inhg):
    if inhg is None:
        return None
    mbar = float(inhg) * 33.8638866667
    return f"{int(round(mbar * 10)):05d}"  # 5 digits

def tempf_to_t_field(tf):
    if tf is None:
        return "000"
    t = int(round(float(tf)))
    return f"{t:03d}" if t >= 0 else f"-{abs(t):02d}"

def hum_to_h_field(h):
    if h is None:
        return None
    h = int(round(float(h)))
    if h >= 100:
        return "00"  # 00 == 100%
    return f"{max(0, min(99, h)):02d}"

def inches_to_hundredths(inches, width=3):
    v = 0.0 if inches is None else max(0.0, float(inches))
    return f"{int(round(v * 100)):{'0'}{width}d}"

def _preview(s: str, n: int) -> str:
    s = (s or "").replace("\r", "\\r").replace("\n", "\\n")
    return s if len(s) <= n else (s[: max(0, n-3)] + "...")

def _nowz():
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

# =========================
#   Meta / Device builders
# =========================

def build_device_string(cfg):
    enabled = cfg.getboolean("device", "enabled", fallback=False)
    if not enabled:
        return ""

    name    = cfg.get("device", "name", fallback="").strip()
    model   = cfg.get("device", "model", fallback="").strip()
    fw      = cfg.get("device", "firmware", fallback="").strip()
    sensors = cfg.get("device", "sensors", fallback="").strip()
    custom  = cfg.get("device", "custom_text", fallback="").strip()

    parts = []
    nm = " ".join([x for x in [name, model] if x])
    if nm:      parts.append(nm)
    if fw:      parts.append(f"FW:{fw}")
    if sensors: parts.append(f"Sensors:{sensors}")
    if custom:  parts.append(custom)

    if not parts:
        return ""
    return "Dev: " + " ".join(parts)

def build_meta_comment(cfg, device_str, include_comment: bool, include_device: bool):
    parts = []
    if include_comment:
        website_url   = cfg.get("meta", "website_url",  fallback="").strip()
        comment_text  = cfg.get("meta", "comment_text", fallback="").strip()
        if website_url:
            u = website_url
            if not u.startswith(("http://", "https://")):
                u = "https://" + u
            parts.append(u)
        if comment_text:
            parts.append(comment_text)
    if include_device and device_str:
        parts.append(device_str)

    if not parts:
        return ""
    return (" " + " | ".join(parts))[:100]

def status_with_addons(base_text, meta_cfg, device_str, include_meta: bool, include_device: bool):
    base = (base_text or "").strip()
    parts = [base] if base else []
    if include_meta:
        website_url   = meta_cfg.get("website_url", fallback="").strip()
        comment_text  = meta_cfg.get("comment_text", fallback="").strip()
        if website_url:
            u = website_url
            if not u.startswith(("http://", "https://")):
                u = "https://" + u
            parts.append(u)
        if comment_text:
            parts.append(comment_text)
    if include_device and device_str:
        parts.append(device_str)
    if not parts:
        return ""
    return (" | ".join(parts))[:120]

# =========================
#   WX / APRS-IS common
# =========================

def build_wx_string(obs):
    winddir = obs.get("winddir")
    windSpeed_mph = obs.get("windSpeed")
    windGust_mph  = obs.get("windGust")
    tempF         = obs.get("temp")
    humidity      = obs.get("humidity")
    pressure_in   = obs.get("pressure")
    precip_total_in = obs.get("precipTotal")  # since midnight
    solar_wm2     = obs.get("solarRadiation")

    ddd = f"{int(round(float(winddir)))%360:03d}" if winddir is not None else "000"
    sss = mph_to_knots(windSpeed_mph)
    ggg = mph_to_knots(windGust_mph)
    t_field = tempf_to_t_field(tempF)

    # r (1h) / p (24h) not provided directly → 000
    r_field = "000"
    p_field = "000"
    P_field = inches_to_hundredths(precip_total_in, width=3)

    h_field = hum_to_h_field(humidity)
    h_part = f"h{h_field}" if h_field is not None else ""

    b_field = inhg_to_tenths_mbar(pressure_in)
    b_part = f"b{b_field}" if b_field is not None else ""

    L_part = ""
    if solar_wm2 is not None:
        try:
            val = int(round(max(0.0, float(solar_wm2))))
            val = min(val, 1999)
            L_part = f"l{val:04d}" if val >= 1000 else f"L{val:03d}"
        except Exception:
            pass

    return f"{ddd}/{sss:03d}g{ggg:03d}t{t_field}r{r_field}p{p_field}P{P_field}{h_part}{b_part}{L_part}"

def fetch_wu_observation(url):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    obs_list = data.get("observations", [])
    if not obs_list:
        raise RuntimeError("WU: no observations in response")
    o = obs_list[0]
    imperial = o.get("imperial", {})
    return {
        "winddir": o.get("winddir"),
        "windSpeed": imperial.get("windSpeed"),
        "windGust": imperial.get("windGust"),
        "temp": imperial.get("temp"),
        "humidity": o.get("humidity"),
        "pressure": imperial.get("pressure"),
        "precipTotal": imperial.get("precipTotal"),
        "solarRadiation": o.get("solarRadiation"),
    }

def build_aprs_info_wx(position_block, wx_block, timestamp_z=True, comment_text=""):
    ts = datetime.now(timezone.utc).strftime("%d%H%Mz") if timestamp_z else ""
    info = f"@{ts}{position_block}{wx_block}"
    if comment_text:
        info += f"{comment_text}"
    return info

def build_status_info(text, include_timestamp=False):
    prefix = datetime.now(timezone.utc).strftime("%d%H%Mz") if include_timestamp else ""
    return f">{prefix}{(text or '').strip()}"

def send_to_aprs_is(server, port, callsign, passcode, software_tag, info_payload, tocall="APRS"):
    packet = f"{callsign}>{tocall},TCPIP*:{info_payload}"
    with socket.create_connection((server, port), timeout=15) as s:
        login = f"user {callsign} pass {passcode} vers {software_tag}\r\n"
        s.sendall(login.encode("ascii", errors="ignore"))
        try:
            s.settimeout(5)
            _ = s.recv(1024)
        except socket.timeout:
            pass
        s.sendall((packet + "\r\n").encode("ascii", errors="ignore"))
    return packet

# ===================================
#   AX.25 / HDLC / AFSK (Bell 202)
# ===================================

def _callsign_ssid_parts(callsign_ssid):
    cs = callsign_ssid.upper().strip()
    if '-' in cs:
        base, ssid = cs.split('-', 1)
        try:
            ssid = int(ssid)
        except Exception:
            ssid = 0
    else:
        base, ssid = cs, 0
    return base, max(0, min(15, ssid))

def _encode_ax25_address(callsign_ssid, last=False, has_been_repeated=False):
    base, ssid = _callsign_ssid_parts(callsign_ssid)
    base = (base + "      ")[:6]
    addr = [(ord(c) << 1) & 0xFE for c in base]
    ssid_byte = 0x60 | ((ssid & 0x0F) << 1)
    if last:
        ssid_byte |= 0x01
    if has_been_repeated:
        ssid_byte |= 0x80
    addr.append(ssid_byte & 0xFF)
    return bytes(addr)

def build_ax25_ui_frame(dest, src, digis, info_bytes):
    addrs = []
    all_addr = [dest, src] + list(digis or [])
    for i, a in enumerate(all_addr):
        last = (i == len(all_addr) - 1)
        addrs.append(_encode_ax25_address(a, last=last, has_been_repeated=False))
    address_field = b''.join(addrs)
    control = b'\x03'  # UI
    pid     = b'\xF0'  # no layer 3
    return address_field + control + pid + info_bytes

def crc16_x25(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    crc = (~crc) & 0xFFFF
    return crc

def _bytes_lsb_first(b: bytes):
    for by in b:
        v = by
        for _ in range(8):
            yield v & 1
            v >>= 1

def _bit_stuff(bits):
    out = []
    ones = 0
    for bit in bits:
        out.append(bit)
        if bit == 1:
            ones += 1
            if ones == 5:
                out.append(0)
                ones = 0
        else:
            ones = 0
    return out

def ax25_to_hdlc_bits(frame_no_flags: bytes, pre_flags: int, post_flags: int):
    fcs = crc16_x25(frame_no_flags)
    with_fcs = frame_no_flags + bytes([fcs & 0xFF, (fcs >> 8) & 0xFF])
    payload_bits = list(_bytes_lsb_first(with_fcs))
    stuffed = _bit_stuff(payload_bits)
    flag_bits = list(_bytes_lsb_first(b'\x7E'))
    bits = []
    for _ in range(pre_flags):
        bits += flag_bits
    bits += flag_bits
    bits += stuffed
    bits += flag_bits
    for _ in range(post_flags):
        bits += flag_bits
    return bits

def nrzi_encode(bits, initial=1):
    cur = initial  # 1=mark, 0=space
    out = []
    for b in bits:
        if b == 0:
            cur ^= 1
        out.append(cur)
    return out

def _tone(fs, hz, dur_ms, amp=0.8):
    if np is None:
        raise RuntimeError("numpy not available; cannot synthesize tone")
    n = max(0, int(round(fs * dur_ms / 1000.0)))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    y = np.sin(2.0 * math.pi * hz * (np.arange(n, dtype=np.float32) / fs)) * amp
    ramp = min(200, int(fs * 0.02))  # up to 20 ms
    if ramp > 0 and n >= 2*ramp:
        w = np.linspace(0,1,ramp,dtype=np.float32)
        y[:ramp] *= w
        y[-ramp:] *= w[::-1]
    return y.astype(np.float32)

def afsk_wave_from_symbols(symbols, fs=48000, baud=1200, mark=1200, space=2200, amp=0.8,
                           noop_ms=0, noop_type='mark',
                           start_silence_ms=0, end_silence_ms=0):
    if np is None:
        raise RuntimeError("numpy not available; cannot synthesize AFSK audio")
    samples_per_bit = int(round(fs / baud))
    dphi_mark = 2.0 * math.pi * (mark / fs)
    dphi_space = 2.0 * math.pi * (space / fs)

    total_samples = samples_per_bit * len(symbols)
    y = np.zeros(total_samples, dtype=np.float32)

    phi = 0.0
    idx = 0
    for sym in symbols:
        dphi = dphi_mark if sym == 1 else dphi_space
        for _ in range(samples_per_bit):
            y[idx] = 0.8 * math.sin(phi) * amp
            phi += dphi
            if phi >= 2.0 * math.pi:
                phi -= 2.0 * math.pi
            idx += 1

    # No-Op preamble (tone/silence)
    pre = np.zeros(0, dtype=np.float32)
    if noop_ms and noop_ms > 0:
        t = (noop_type or 'mark').lower()
        if t == 'silence':
            pre = np.zeros(int(round(fs * noop_ms / 1000.0)), dtype=np.float32)
        elif t == 'space':
            pre = _tone(fs, space, noop_ms, amp=amp)
        else:
            pre = _tone(fs, mark, noop_ms, amp=amp)

    # Surrounding silence blocks
    s0 = np.zeros(int(round(fs * max(0, start_silence_ms) / 1000.0)), dtype=np.float32)
    s1 = np.zeros(int(round(fs * max(0, end_silence_ms) / 1000.0)), dtype=np.float32)

    out = np.concatenate([s0, pre, y, s1]) if (pre.size or s0.size or s1.size) else y

    # Gentle ramp
    ramp = min(200, samples_per_bit * 2)
    if ramp > 0 and out.size >= 2*ramp:
        window = np.linspace(0, 1, ramp, dtype=np.float32)
        out[:ramp] *= window
        out[-ramp:] *= window[::-1]
    return out

def write_wav(path, pcm, fs=48000):
    pcm16 = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    import wave
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm16.tobytes())

def write_wav_stereo(path, pcm_lr, fs=48000):
    pcm16 = np.clip(pcm_lr, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    import wave
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm_lr.tobytes())

def play_soundcard(pcm, fs=48000, device=None, channel_mode='stereo_dup', output_gain=1.0,
                   mirror_to_wav=False, mirror_wav_path='aprs_mirror.wav'):
    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError("sounddevice not installed") from e

    x = np.asarray(pcm, dtype=np.float32) * float(output_gain)

    chm = (channel_mode or 'stereo_dup').lower()
    if chm == 'mono':
        out = x  # shape: (N,)
        if mirror_to_wav:
            write_wav(mirror_wav_path, out, fs=fs)
    elif chm in ('left_only', 'right_only', 'stereo_dup'):
        L = x if chm in ('left_only', 'stereo_dup') else np.zeros_like(x)
        R = x if chm in ('right_only', 'stereo_dup') else np.zeros_like(x)
        out = np.column_stack([L, R])
        if mirror_to_wav:
            write_wav_stereo(mirror_wav_path, out, fs=fs)
    else:
        out = np.column_stack([x, x])
        if mirror_to_wav:
            write_wav_stereo(mirror_wav_path, out, fs=fs)

    try:
        import sounddevice as sd
        sd.play(out, fs, device=device, blocking=True)
    finally:
        try:
            sd.stop()
        except Exception:
            pass

class SerialPTT:
    def __init__(self, port, rts=True, dtr=False):
        self.port = port
        self.rts = rts
        self.dtr = dtr
        self.ser = None

    def __enter__(self):
        try:
            import serial
        except Exception as e:
            raise RuntimeError("pyserial not installed") from e
        self.ser = serial.Serial(self.port)
        self.ser.setDTR(self.dtr)
        self.ser.setRTS(self.rts)
        return self

    def key(self):
        if self.ser:
            self.ser.setDTR(self.dtr)
            self.ser.setRTS(self.rts)

    def unkey(self):
        if self.ser:
            self.ser.setDTR(False)
            self.ser.setRTS(False)

    def __exit__(self, exc_type, exc, tb):
        try:
            self.unkey()
        finally:
            if self.ser:
                self.ser.close()

# =========================
#       TX helpers
# =========================

def build_ax25_ui_frame(dest, src, digis, info_bytes):
    addrs = []
    all_addr = [dest, src] + list(digis or [])
    for i, a in enumerate(all_addr):
        last = (i == len(all_addr) - 1)
        addrs.append(_encode_ax25_address(a, last=last, has_been_repeated=False))
    address_field = b''.join(addrs)
    control = b'\x03'  # UI
    pid     = b'\xF0'  # no layer 3
    return address_field + control + pid + info_bytes

def do_tx_afsk(info, cfg):
    # Build AFSK and play/write
    src_call = cfg['aprs'].get('callsign', 'N0CALL-13').strip()
    dest     = cfg['rf'].get('dest', 'APRS').strip() or 'APRS'
    path_str = cfg['rf'].get('path', 'WIDE1-1,WIDE2-1').strip()
    digis = [p.strip() for p in path_str.split(',') if p.strip()]

    pre_flags  = cfg['afsk'].getint('preamble_flags', fallback=25)
    post_flags = cfg['afsk'].getint('tail_flags', fallback=5)

    fs    = cfg['afsk'].getint('sample_rate', fallback=48000)
    baud  = cfg['afsk'].getint('baud', fallback=1200)
    mark  = cfg['afsk'].getint('mark_hz', fallback=1200)
    space = cfg['afsk'].getint('space_hz', fallback=2200)
    amp   = float(cfg['afsk'].get('amplitude', fallback='0.8'))
    gain  = float(cfg['afsk'].get('output_gain', fallback='1.0'))

    noop_ms   = cfg['afsk'].getint('noop_preamble_ms', fallback=0)
    noop_type = cfg['afsk'].get('noop_preamble_type', fallback='mark')

    start_silence = cfg['afsk'].getint('start_silence_ms', fallback=0)
    end_silence   = cfg['afsk'].getint('end_silence_ms',   fallback=0)

    out_mode = cfg['afsk'].get('output', fallback='wav').lower()
    out_wav  = cfg['afsk'].get('wav_path', fallback='aprs_tx.wav')
    device   = cfg['afsk'].get('soundcard_device', fallback=None)
    ch_mode  = cfg['afsk'].get('channel_mode', fallback='stereo_dup')
    mirror   = cfg['afsk'].getboolean('mirror_to_wav', fallback=False)
    mirror_p = cfg['afsk'].get('mirror_wav_path', fallback='aprs_mirror.wav')

    info_b = info.encode('ascii', errors='ignore')
    frame  = build_ax25_ui_frame(dest, src_call, digis, info_b)
    bits   = ax25_to_hdlc_bits(frame, pre_flags, post_flags)
    symbols = nrzi_encode(bits, initial=1)
    pcm = afsk_wave_from_symbols(symbols, fs=fs, baud=baud, mark=mark, space=space, amp=amp,
                                 noop_ms=noop_ms, noop_type=noop_type,
                                 start_silence_ms=start_silence, end_silence_ms=end_silence)

    ptt_mode = cfg['afsk'].get('ptt', fallback='none').lower()
    ptt_port = cfg['afsk'].get('ptt_serial_port', fallback='COM3')
    ptt_rts  = cfg['afsk'].getboolean('ptt_rts', fallback=True)
    ptt_dtr  = cfg['afsk'].getboolean('ptt_dtr', fallback=False)

    def _do_play():
        if out_mode == 'soundcard':
            play_soundcard(pcm, fs=fs, device=device, channel_mode=ch_mode, output_gain=gain,
                           mirror_to_wav=mirror, mirror_wav_path=mirror_p)
        else:
            write_wav(out_wav, pcm, fs=fs)

    if ptt_mode == 'serial':
        with SerialPTT(ptt_port, rts=ptt_rts, dtr=ptt_dtr):
            _do_play()
    else:
        _do_play()

def do_send_wx(cfg, wu_url, position_block, device_str, dev_in_comment, meta_in_comment,
               is_on, rf_on, server, port, callsign, passcode, sw_tag, tocall, log_preview, log_maxlen):
    obs = fetch_wu_observation(wu_url)
    wx  = build_wx_string(obs)
    comment_text = build_meta_comment(cfg, device_str,
                                      include_comment=meta_in_comment,
                                      include_device=dev_in_comment)
    info = build_aprs_info_wx(position_block, wx, timestamp_z=True, comment_text=comment_text)
    # APRS-IS
    if is_on:
        try:
            send_to_aprs_is(server, port, callsign, passcode, sw_tag, info, tocall=tocall)
            if log_preview:
                print(f"[{_nowz()}] IS TX (wx): { _preview(info, log_maxlen) }")
            else:
                print(f"[{_nowz()}] IS TX (wx)")
        except Exception as e:
            print("ERROR APRS-IS WX:", e)
    # AFSK
    if rf_on and cfg.getboolean("afsk", "send_wx", fallback=True):
        try:
            do_tx_afsk(info, cfg)
            if log_preview:
                print(f"[{_nowz()}] AFSK TX (wx): { _preview(info, log_maxlen) }")
            else:
                print(f"[{_nowz()}] AFSK TX (wx)")
        except Exception as e:
            print("ERROR AFSK WX:", e)

def do_send_status(cfg, device_str, meta_in_status, dev_in_status,
                   is_on, rf_on, server, port, callsign, passcode, sw_tag, tocall, log_preview, log_maxlen):
    st_text      = cfg.get("status", "text", fallback="WX bridge online")
    st_timestamp = cfg.getboolean("status", "include_timestamp", fallback=False)
    st_payload   = status_with_addons(st_text, cfg["meta"] if "meta" in cfg else configparser.SectionProxy(cfg, "meta"),
                                      device_str, include_meta=meta_in_status, include_device=dev_in_status)
    info = build_status_info(st_payload, include_timestamp=st_timestamp)
    # APRS-IS
    if is_on:
        try:
            send_to_aprs_is(server, port, callsign, passcode, sw_tag, info, tocall=tocall)
            if log_preview:
                print(f"[{_nowz()}] IS TX (status): { _preview(info, log_maxlen) }")
            else:
                print(f"[{_nowz()}] IS TX (status)")
        except Exception as e:
            print("ERROR APRS-IS Status:", e)
    # AFSK
    if rf_on and cfg.getboolean("afsk", "send_status", fallback=False):
        try:
            do_tx_afsk(info, cfg)
            if log_preview:
                print(f"[{_nowz()}] AFSK TX (status): { _preview(info, log_maxlen) }")
            else:
                print(f"[{_nowz()}] AFSK TX (status)")
        except Exception as e:
            print("ERROR AFSK Status:", e)

# =========================
#           MAIN
# =========================

def main():
    cfg = configparser.ConfigParser()
    cfg.read("config_2.1.ini", encoding="utf-8")

    # Logging controls
    log_preview = cfg.getboolean("logging", "payload_preview", fallback=True)
    log_maxlen  = cfg.getint("logging", "payload_maxlen", fallback=220)

    # Hotkeys
    hotkeys_enabled = cfg.getboolean("hotkeys", "enabled", fallback=True)
    hotkeys_hint    = cfg.getboolean("hotkeys", "hint", fallback=True)
    hotkeys_mode    = cfg.get("hotkeys", "mode", fallback="auto").strip().lower()

    # APRS-IS settings
    callsign = cfg.get("aprs", "callsign", fallback=os.environ.get("APRS_CALLSIGN", "N0CALL-13"))
    passcode = cfg.get("aprs", "passcode", fallback=os.environ.get("APRS_PASSCODE", "00000"))
    server   = cfg.get("aprs", "server", fallback=os.environ.get("APRS_SERVER", "asia.aprs2.net"))
    port     = cfg.getint("aprs", "port", fallback=int(os.environ.get("APRS_PORT", "14580")))
    sw_tag   = cfg.get("aprs", "software_tag", fallback="WU2APRS 2.1")
    tocall   = cfg.get("aprs", "tocall", fallback="APRS").strip() or "APRS"
    is_on    = cfg.getboolean("aprs", "enable_is", fallback=True)  # runtime toggle-able

    # Intervals
    wx_interval     = cfg.getint("aprs", "interval_sec", fallback=int(os.environ.get("APRS_INTERVAL", "600")))
    st_enabled      = cfg.getboolean("status", "enabled", fallback=False)
    st_on_start     = cfg.getboolean("status", "send_on_startup", fallback=True)
    st_interval     = cfg.getint("status", "interval_sec", fallback=0)

    # Station
    lat = cfg.getfloat("station", "lat", fallback=float(os.environ.get("STATION_LAT", "0")))
    lon = cfg.getfloat("station", "lon", fallback=float(os.environ.get("STATION_LON", "0")))

    # WU
    wu_station = cfg.get("wu", "station_id", fallback=os.environ.get("WU_STATION_ID"))
    wu_key     = cfg.get("wu", "api_key", fallback=os.environ.get("WU_API_KEY"))
    wu_units   = cfg.get("wu", "units", fallback=os.environ.get("WU_UNITS", "e")).lower()
    if not wu_station or not wu_key:
        print("Missing WU station_id/api_key. Please set in config.ini [wu].")
        sys.exit(1)
    wu_url = ( "https://api.weather.com/v2/pws/observations/current"
               f"?stationId={wu_station}&format=json&units={wu_units}&apiKey={wu_key}" )

    # Device / Meta includes
    device_str        = build_device_string(cfg)
    dev_in_comment    = cfg.getboolean("device", "include_in_comment", fallback=False)
    dev_in_status     = cfg.getboolean("device", "include_in_status",  fallback=False)
    meta_in_comment   = cfg.getboolean("meta", "include_in_comment",    fallback=False)
    meta_in_status    = cfg.getboolean("meta", "include_in_status",     fallback=False)

    # AFSK options
    rf_on            = cfg.getboolean("afsk", "enabled", fallback=False)  # runtime toggle-able
    afsk_send_wx      = cfg.getboolean("afsk", "send_wx", fallback=True)
    afsk_send_status  = cfg.getboolean("afsk", "send_status", fallback=False)
    afsk_wx_interval  = cfg.getint("afsk", "wx_interval_sec", fallback=wx_interval)
    afsk_st_interval  = cfg.getint("afsk", "status_interval_sec", fallback=max(0, st_interval))

    # Precompute blocks
    position_block = to_aprs_latlon(lat, lon)

    print(f"Starting WU2APRS: {callsign} | IS={'on' if is_on else 'off'} "
          f"@ {server}:{port} every {wx_interval}s ; RF={'on' if rf_on else 'off'}")

    # Hotkey flags
    hk_wx = threading.Event()
    hk_status = threading.Event()
    hk_tog_rf = threading.Event()
    hk_tog_is = threading.Event()

    def _hotkey_thread():
        if not hotkeys_enabled:
            return
        if hotkeys_hint:
            print("[hotkeys] W=WX now, S=Status now, R=toggle RF, I=toggle IS (Q=quit console mode).")
        mode = hotkeys_mode
        if mode == "auto":
            try:
                import keyboard  # type: ignore
                keyboard.add_hotkey('w', lambda: hk_wx.set())
                keyboard.add_hotkey('s', lambda: hk_status.set())
                keyboard.add_hotkey('r', lambda: hk_tog_rf.set())
                keyboard.add_hotkey('i', lambda: hk_tog_is.set())
                print("[hotkeys] Using 'keyboard' library for global hotkeys.")
                keyboard.wait()
                return
            except Exception:
                mode = "console"
        if mode == "keyboard":
            try:
                import keyboard  # type: ignore
                keyboard.add_hotkey('w', lambda: hk_wx.set())
                keyboard.add_hotkey('s', lambda: hk_status.set())
                keyboard.add_hotkey('r', lambda: hk_tog_rf.set())
                keyboard.add_hotkey('i', lambda: hk_tog_is.set())
                print("[hotkeys] Using 'keyboard' library for global hotkeys.")
                keyboard.wait()
                return
            except Exception as e:
                print("[hotkeys] Failed to init 'keyboard':", e)
                return
        if mode == "console":
            if os.name == 'nt':
                try:
                    import msvcrt  # Windows console
                    print("[hotkeys] Console mode active: press keys in this window (W/S/R/I/Q).")
                    while True:
                        if msvcrt.kbhit():
                            ch = msvcrt.getwch().lower()
                            if ch == 'w':   hk_wx.set()
                            elif ch == 's': hk_status.set()
                            elif ch == 'r': hk_tog_rf.set()
                            elif ch == 'i': hk_tog_is.set()
                            elif ch == 'q': break
                        time.sleep(0.05)
                except Exception as e:
                    print("[hotkeys] Console mode not available:", e)
            else:
                print("[hotkeys] Console mode only supported on Windows.")

    t = threading.Thread(target=_hotkey_thread, daemon=True)
    t.start()

    now = time.time()
    next_wx = now
    next_st = None
    next_wx_rf = now if rf_on and afsk_send_wx else None
    next_st_rf = now if rf_on and afsk_send_status and st_enabled else None

    # Initial status (scheduled start)
    if st_enabled and st_on_start:
        do_send_status(cfg, device_str, meta_in_status, dev_in_status,
                       is_on, rf_on, server, port, callsign, passcode, sw_tag, tocall, log_preview, log_maxlen)
        if st_interval > 0:
            next_st = now + st_interval
        if afsk_st_interval and afsk_st_interval > 0:
            next_st_rf = now + afsk_st_interval

    while True:
        try:
            now = time.time()

            # Handle toggles
            if hk_tog_is.is_set():
                hk_tog_is.clear()
                is_on = not is_on
                print(f"[{_nowz()}] TOGGLE IS: {'ON' if is_on else 'OFF'}")
                # push next_wx so we don't immediately tx if toggled off
                next_wx = now + max(30, wx_interval)

            if hk_tog_rf.is_set():
                hk_tog_rf.clear()
                rf_on = not rf_on
                print(f"[{_nowz()}] TOGGLE RF: {'ON' if rf_on else 'OFF'}")
                next_wx_rf = now + max(30, afsk_wx_interval)

            # Handle hotkeys (immediate TX)
            if hk_wx.is_set():
                hk_wx.clear()
                try:
                    do_send_wx(cfg, wu_url, position_block, device_str, dev_in_comment, meta_in_comment,
                               is_on, rf_on, server, port, callsign, passcode, sw_tag, tocall, log_preview, log_maxlen)
                except Exception as e:
                    print("ERROR hotkey WX:", e)
                next_wx = now + max(30, wx_interval)
                if next_wx_rf is not None:
                    next_wx_rf = now + max(30, afsk_wx_interval)

            if hk_status.is_set():
                hk_status.clear()
                try:
                    do_send_status(cfg, device_str, meta_in_status, dev_in_status,
                                   is_on, rf_on, server, port, callsign, passcode, sw_tag, tocall, log_preview, log_maxlen)
                except Exception as e:
                    print("ERROR hotkey Status:", e)
                if st_interval > 0:
                    next_st = now + st_interval
                if afsk_st_interval and afsk_st_interval > 0:
                    next_st_rf = now + afsk_st_interval

            # Scheduled WX → IS
            if is_on and now >= next_wx:
                try:
                    do_send_wx(cfg, wu_url, position_block, device_str, dev_in_comment, meta_in_comment,
                               is_on, rf_on, server, port, callsign, passcode, sw_tag, tocall, log_preview, log_maxlen)
                except Exception as e:
                    print("ERROR schedule WX (IS):", e)
                next_wx = now + max(30, wx_interval)
            elif not is_on:
                next_wx = now + max(30, wx_interval)

            # Scheduled Status → IS
            if is_on and st_enabled and st_interval > 0 and (next_st is None or now >= next_st):
                try:
                    do_send_status(cfg, device_str, meta_in_status, dev_in_status,
                                   is_on, rf_on, server, port, callsign, passcode, sw_tag, tocall, log_preview, log_maxlen)
                except Exception as e:
                    print("ERROR schedule Status (IS):", e)
                next_st = now + st_interval

            # Scheduled WX → RF
            if rf_on and afsk_send_wx and (next_wx_rf is None or now >= next_wx_rf):
                try:
                    do_send_wx(cfg, wu_url, position_block, device_str, dev_in_comment, meta_in_comment,
                               is_on, rf_on, server, port, callsign, passcode, sw_tag, tocall, log_preview, log_maxlen)
                except Exception as e:
                    print("ERROR schedule WX (RF):", e)
                next_wx_rf = now + max(30, afsk_wx_interval)

            # Scheduled Status → RF
            if (rf_on and afsk_send_status and st_enabled and
                (afsk_st_interval and afsk_st_interval > 0) and
                (next_st_rf is None or now >= next_st_rf)):
                try:
                    do_send_status(cfg, device_str, meta_in_status, dev_in_status,
                                   is_on, rf_on, server, port, callsign, passcode, sw_tag, tocall, log_preview, log_maxlen)
                except Exception as e:
                    print("ERROR schedule Status (RF):", e)
                next_st_rf = now + afsk_st_interval

            time.sleep(0.05)

        except KeyboardInterrupt:
            print("Exiting.")
            break
        except Exception as e:
            print("ERROR main loop:", e)
            time.sleep(1)

if __name__ == "__main__":
    main()
