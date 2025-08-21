#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import time
import requests
import os
import sys
import configparser
from datetime import datetime, timezone

# ==============
# Utilities
# ==============
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
    # Table '/' + Symbol '_' (Weather station)
    return f"{lat_dm}{ns}/{lon_dm}{ew}_"

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
    if inches is None:
        v = 0.0
    else:
        v = max(0.0, float(inches))
    return f"{int(round(v * 100)):{'0'}{width}d}"
    
def build_device_string(cfg):
    """
    อ่านข้อมูลอุปกรณ์จาก [device] แล้วสร้างสตริงสั้น ๆ เช่น:
    'Dev: WXNode RPi4 FW:1.2.3 Sensors:WS-2902A'
    """
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
    if fw:      parts.append(f"FW: {fw}")
    if sensors: parts.append(f"Sensors: {sensors}")
    if custom:  parts.append(custom)

    if not parts:
        return ""
    return "Device: " + " ".join(parts)
    
def build_comment(website_url, extra_comment, device_str=""):
    parts = []
    if website_url:
        u = website_url.strip()
        if not u.startswith(("http://", "https://")):
            u = "https://" + u
        parts.append(u)
    if extra_comment:
        parts.append(extra_comment.strip())
    if device_str:
        parts.append(device_str.strip())

    if not parts:
        return ""
    # คอมเมนต์รวม (คั่นด้วย " | "), จำกัด ~100 ตัวอักษรเพื่อความเข้ากันได้
    return (" " + " | ".join(parts))[:100]
    
def build_wx_string(obs):
    """
    obs (units=e → imperial):
      winddir, windSpeed, windGust, temp, humidity, pressure, precipTotal, solarRadiation
    Returns APRS weather block e.g. ddd/sss ggg t r p P h b [L|l]
    """
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

    # r (1 ชม.) / p (24 ชม.) ไม่มีจาก WU โดยตรง → ใส่ 000 ไว้ก่อน
    r_field = "000"
    p_field = "000"
    P_field = inches_to_hundredths(precip_total_in, width=3)

    h_field = hum_to_h_field(humidity)
    h_part = f"h{h_field}" if h_field is not None else ""

    b_field = inhg_to_tenths_mbar(pressure_in)
    b_part = f"b{b_field}" if b_field is not None else ""

    # Solar radiation → Lxxx (0–999) หรือ lxxxx (>=1000)
    L_part = ""
    if solar_wm2 is not None:
        try:
            val = int(round(max(0.0, float(solar_wm2))))
            val = min(val, 1999)  # guard
            L_part = f"l{val:04d}" if val >= 1000 else f"L{val:03d}"
        except Exception:
            pass

    return f"{ddd}/{sss:03d}g{ggg:03d}t{t_field}r{r_field}p{p_field}P{P_field}{h_part}{b_part}{L_part}"

def build_position_weather_packet(callsign, position_block, wx_block, comment_text=""):
    # Timestamp (UTC): @DDHHMMz
    ts = datetime.now(timezone.utc).strftime("%d%H%Mz")
    info = f"@{ts}{position_block}{wx_block}{comment_text}"
    return f"{callsign}>APRS,TCPIP*:{info}"

def build_status_packet(callsign, status_text, include_timestamp=False):
    """
    สร้าง Status packet (ข้อมูลชนิด '>').
    ตามสเปกสามารถใส่ timestamp นำหน้าข้อความได้หรือไม่ก็ได้
    """
    if not status_text:
        status_text = ""
    status_text = status_text.strip()
    prefix = ""
    if include_timestamp:
        # ใช้ Zulu time แบบ DDHHMMz เพื่อความชัดเจน
        prefix = datetime.now(timezone.utc).strftime("%d%H%Mz")
    # จำกัดความยาวพอประมาณเพื่อความเข้ากันได้
    payload = f">{prefix}{status_text}"[:120]
    return f"{callsign}>APRS,TCPIP*:{payload}"

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
        "solarRadiation": o.get("solarRadiation"),  # W/m^2
    }

def send_to_aprs_is(server, port, callsign, passcode, software_tag, packet):
    with socket.create_connection((server, port), timeout=15) as s:
        login = f"user {callsign} pass {passcode} vers {software_tag}\r\n"
        s.sendall(login.encode("ascii", errors="ignore"))
        try:
            s.settimeout(5)
            _ = s.recv(1024)
        except socket.timeout:
            pass
        s.sendall((packet + "\r\n").encode("ascii", errors="ignore"))

def main():
    # อ่าน config
    cfg = configparser.ConfigParser()
    cfg.read("config.ini", encoding="utf-8")

    callsign = cfg.get("aprs", "callsign", fallback=os.environ.get("APRS_CALLSIGN", "N0CALL-13"))
    passcode = cfg.get("aprs", "passcode", fallback=os.environ.get("APRS_PASSCODE", "00000"))
    server   = cfg.get("aprs", "server", fallback=os.environ.get("APRS_SERVER", "asia.aprs2.net"))
    port     = cfg.getint("aprs", "port", fallback=int(os.environ.get("APRS_PORT", "14580")))
    sw_tag   = cfg.get("aprs", "software_tag", fallback="WU2APRS 1.3")

    # ช่วงเวลาส่ง
    wx_interval   = cfg.getint("aprs", "interval_sec", fallback=int(os.environ.get("APRS_INTERVAL", "600")))
    st_enabled    = cfg.getboolean("status", "enabled", fallback=False)
    st_on_start   = cfg.getboolean("status", "send_on_startup", fallback=True)
    st_interval   = cfg.getint("status", "interval_sec", fallback=0)
    st_text       = cfg.get("status", "text", fallback="WX bridge online")
    st_timestamp  = cfg.getboolean("status", "include_timestamp", fallback=False)

    # สถานี
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

    # ตัวเลือก Device
    dev_str            = build_device_string(cfg)
    dev_in_comment     = cfg.getboolean("device", "include_in_comment", fallback=True)
    dev_in_status      = cfg.getboolean("device", "include_in_status",  fallback=True)
    
    # คอมเมนต์ท้ายแพ็กเก็ต (URL + ข้อความอื่น + อุปกรณ์ตามตัวเลือก)
    website_url   = cfg.get("meta", "website_url",  fallback=os.environ.get("WEBSITE_URL", "")).strip()
    extra_comment = cfg.get("meta", "comment_text", fallback=os.environ.get("COMMENT_TEXT", "")).strip()
    comment_text  = build_comment(website_url, extra_comment, dev_str if dev_in_comment else "")

    # สร้างเนื้อหา Status (แนบ Device หรือไม่ตามตัวเลือก)
    def status_with_device(base):
        if dev_in_status and dev_str:
            return (base + " | " + dev_str)[:120]
        return base[:120]
        
    position_block = to_aprs_latlon(lat, lon)
    print(f"Starting WU2APRS: {callsign} → {server}:{port}  WX every {wx_interval}s  "
          f"Status={'on' if st_enabled else 'off'}")

    # ตัวจับเวลา
    now = time.time()
    next_wx = now
    next_st = None

    first_status_sent = False
    if st_enabled and st_on_start:
        # ส่งสถานะทันทีครั้งแรก
        pkt = build_status_packet(callsign, status_with_device(st_text), include_timestamp=st_timestamp)
        print(f"[{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}] TX (status/start): {pkt}")
        try:
            send_to_aprs_is(server, port, callsign, passcode, sw_tag, pkt)
        except Exception as e:
            print("ERROR sending startup status:", e)
        first_status_sent = True
        if st_interval > 0:
            next_st = now + st_interval

    if st_enabled and not first_status_sent and st_interval > 0:
        next_st = now + st_interval

    while True:
        try:
            now = time.time()
            # ส่ง Weather / Position
            if now >= next_wx:
                obs = fetch_wu_observation(wu_url)
                wx  = build_wx_string(obs)
                pkt = build_position_weather_packet(callsign, position_block, wx, comment_text=comment_text)
                print(f"[{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}] TX (wx): {pkt}")
                send_to_aprs_is(server, port, callsign, passcode, sw_tag, pkt)
                next_wx = now + max(30, wx_interval)

            # ส่ง Status ตามรอบ (ถ้าตั้งไว้)
            if st_enabled and st_interval > 0 and (next_st is None or now >= next_st):
                pkt = build_status_packet(callsign, status_with_device(st_text), include_timestamp=st_timestamp)
                print(f"[{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}] TX (status): {pkt}")
                send_to_aprs_is(server, port, callsign, passcode, sw_tag, pkt)
                next_st = now + st_interval

            # นอนจนกว่าจะถึงงานถัดไป
            sleep_to = min(next_wx, next_st) - now if (next_st is not None) else (next_wx - now)
            time.sleep(max(1.0, min(sleep_to, 5.0)))
        except Exception as e:
            print("ERROR:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
