import requests

# === CONFIG TELEGRAM ===


def send_telegram_message(text: str):
    TELEGRAM_TOKEN = "7386833040:AAE5Z05xuKDu1w9y3oar_CxKpoJMme-8zu0"
    CHAT_ID = "-4874762893"
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ Telegram error {r.status_code}: {r.text}")
        else:
            print("📨 Đã gửi Telegram thành công!")
    except Exception as e:
        print(f"⚠️ Lỗi khi gửi Telegram: {e}")