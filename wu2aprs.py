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
        v = 0
    else:
        v = max(0.0, float(inches))
    return f"{int(round(v * 100)):{'0'}{width}d}"

def build_wx_string(obs):
    """
    obs: dict from WU (units=e → imperial)
      keys we use:
        winddir, windSpeed, windGust, temp, humidity, pressure, precipTotal, solarRadiation
    Returns APRS weather block e.g. ddd/sss ggg t r p P h b [L|l]
    """
    winddir = obs.get("winddir")
    windSpeed_mph = obs.get("windSpeed")
    windGust_mph  = obs.get("windGust")
    tempF         = obs.get("temp")
    humidity      = obs.get("humidity")
    pressure_in   = obs.get("pressure")
    precip_total_in = obs.get("precipTotal")  # since midnight (WU)
    solar_wm2     = obs.get("solarRadiation")

    ddd = f"{int(round(float(winddir)))%360:03d}" if winddir is not None else "000"
    sss = mph_to_knots(windSpeed_mph)
    ggg = mph_to_knots(windGust_mph)
    t_field = tempf_to_t_field(tempF)

    # Rain fields: r (last hour) & p (last 24h) ไม่ได้ให้มาจาก WU ตรงๆ → ใส่ 000 ไว้ก่อน
    r_field = "000"
    p_field = "000"
    P_field = inches_to_hundredths(precip_total_in, width=3)  # since midnight

    h_field = hum_to_h_field(humidity)
    h_part = f"h{h_field}" if h_field is not None else ""

    b_field = inhg_to_tenths_mbar(pressure_in)
    b_part = f"b{b_field}" if b_field is not None else ""

    # Solar radiation → Lxxx (0–999) หรือ lxxxx (>=1000)
    L_part = ""
    if solar_wm2 is not None:
        try:
            val = int(round(max(0.0, float(solar_wm2))))
            val = min(val, 1999)  # กันค่าเพี้ยน
            if val >= 1000:
                L_part = f"l{val:04d}"
            else:
                L_part = f"L{val:03d}"
        except Exception:
            pass

    return f"{ddd}/{sss:03d}g{ggg:03d}t{t_field}r{r_field}p{p_field}P{P_field}{h_part}{b_part}{L_part}"

def build_aprs_packet(callsign, position_block, wx_block, website_comment=None):
    # Timestamp (UTC): @DDHHMMz
    ts = datetime.now(timezone.utc).strftime("%d%H%Mz")
    comment = ""
    if website_comment:
        # ตัดความยาวคอมเมนต์ไม่ให้ยาวเกินไป (กันการตัดทอนโดยไคลเอนต์บางตัว)
        website_comment = website_comment.strip()
        # ใส่ช่องว่างนำหน้าคอมเมนต์ตามธรรมเนียม APRS
        comment = f" {website_comment}"[:80]  # จำกัด ~80 ตัวอักษรพอเหมาะ
    info = f"@{ts}{position_block}{wx_block}{comment}"
    # ตามแนวทาง client-originated → path ใช้ TCPIP*
    return f"{callsign}>APRS,TCPIP*:{info}"

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
        s.sendall(login.encode("ascii"))
        # อ่านข้อความต้อนรับ/ฟิลเตอร์ (optional)
        try:
            s.settimeout(5)
            _ = s.recv(1024)
        except socket.timeout:
            pass
        s.sendall((packet + "\r\n").encode("ascii"))

def main():
    # อ่าน config
    cfg = configparser.ConfigParser()
    cfg.read("config.ini", encoding="utf-8")

    callsign = cfg.get("aprs", "callsign", fallback=os.environ.get("APRS_CALLSIGN", "N0CALL-13"))
    passcode = cfg.get("aprs", "passcode", fallback=os.environ.get("APRS_PASSCODE", "00000"))
    server   = cfg.get("aprs", "server", fallback=os.environ.get("APRS_SERVER", "asia.aprs2.net"))
    port     = cfg.getint("aprs", "port", fallback=int(os.environ.get("APRS_PORT", "14580")))
    interval = cfg.getint("aprs", "interval_sec", fallback=int(os.environ.get("APRS_INTERVAL", "600")))
    sw_tag   = cfg.get("aprs", "software_tag", fallback="WU2APRS 1.1")

    lat = cfg.getfloat("station", "lat", fallback=float(os.environ.get("STATION_LAT", "0")))
    lon = cfg.getfloat("station", "lon", fallback=float(os.environ.get("STATION_LON", "0")))

    wu_station = cfg.get("wu", "station_id", fallback=os.environ.get("WU_STATION_ID"))
    wu_key     = cfg.get("wu", "api_key", fallback=os.environ.get("WU_API_KEY"))
    wu_units   = cfg.get("wu", "units", fallback=os.environ.get("WU_UNITS", "e")).lower()

    website_comment = cfg.get("meta", "website_url", fallback=os.environ.get("WEBSITE_URL", "")).strip()
    if website_comment and not website_comment.startswith(("http://", "https://")):
        # ไม่บังคับต้องมี http(s) แต่ใส่ให้จะดูเป็นลิงก์
        website_comment = "https://" + website_comment

    if not wu_station or not wu_key:
        print("Missing WU station_id/api_key. Please set in config.ini [wu].")
        sys.exit(1)

    wu_url = ( "https://api.weather.com/v2/pws/observations/current"
               f"?stationId={wu_station}&format=json&units={wu_units}&apiKey={wu_key}" )

    position_block = to_aprs_latlon(lat, lon)

    print(f"Starting WU2APRS: {callsign} → {server}:{port} every {interval}s")
    while True:
        try:
            obs = fetch_wu_observation(wu_url)
            wx  = build_wx_string(obs)
            pkt = build_aprs_packet(callsign, position_block, wx, website_comment=website_comment)
            utc_now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"[{utc_now}] TX: {pkt}")
            send_to_aprs_is(server, port, callsign, passcode, sw_tag, pkt)
        except Exception as e:
            print("ERROR:", e)
        time.sleep(max(30, interval))  # กันค่าต่ำเกินไป

if __name__ == "__main__":
    main()
