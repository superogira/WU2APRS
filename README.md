![Alt text](WU2APRS.png?raw=true)

# WU2APRS
wunderground.com to APRS Weather Station (IS & RF)

## **การใช้งาน**
- จะต้องมีสถานีตรวจอากาศบนเว็บ [wunderground.com](https://wunderground.com) ก่อน เพราะต้องใช้ในการสร้าง API เพื่อดึงข้อมูลมา จึงจะสามารถใช้งานได้ (สามารถสร้างสถานีตรวจอากาศง่าย ๆ ด้วย arduino กับบอร์ด esp8266 หรือ esp32 ก็ได้)
- รันผ่านไฟล์ .py โดยตรงเลยก็ได้ โดยตั้งค่าต่าง ๆ ใน config_2.3.ini ก่อนการรัน (แต่ต้องลง Python)
- หากไม่มี Python หรือเน้นง่าย ๆ ให้ใช้ตัว .exe แทน โดยตั้งค่าต่าง ๆ ใน config.ini ก่อนการเปิดโปรแกรม

---

## **Usage**
You must first have a weather station on [wunderground.com](https://wunderground.com) to generate an API key for fetching data. (You can easily create a weather station using an Arduino with an ESP8266 or ESP32 board).

---

### **Running with Python**

You can run the script directly using the `.py` file. Please configure the settings in `config_2.1.ini` before running. (Requires Python to be installed).

---

### **Running with the Executable (EXE)**

If you don't have Python or prefer a simpler method, use the `.exe` file instead. Make sure to configure the settings in `config_2.3.ini` before launching the program.

---

![Alt text](screenshots/wu2aprs_1.png?raw=true)
![Alt text](screenshots/wu2aprs_2.png?raw=true)
![Alt text](screenshots/wu2aprs_3.png?raw=true)
![Alt text](screenshots/wu2aprs_4.png?raw=true)

---

- หากสถานีตรวจอากาศของคุณสามารถเข้าหน้าเว็บ http://device_ip/record.html ได้ (ส่วนใหญ่จะเป็นอุปกรณ์ของ NicetyWeather) จะสามารถตั้งค่า source mode = local ได้ ซึ่งทำให้ไม่จำเป็นต้องดึงข้อมูลจาก wunderground และไม่ต้องต่อ Internet ทำงานเป็นแบบ Off Grid ไปเลยก็ได้
- If your weather station has a local web interface accessible at http://<device_ip>/record.html (which is common for devices from NicetyWeather), you can set the source_mode = local.
This allows the program to pull data directly from your device on the local network. It removes the need to fetch data from Wunderground and does not require an internet connection, enabling fully off-grid operation.
![Alt text](screenshots/wu2aprs_5.png?raw=true)
