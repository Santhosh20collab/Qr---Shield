import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H
import os
import csv
import random
import shutil

# Wipe old dataset so we don't mix old + new runs
if os.path.exists("dataset"):
    shutil.rmtree("dataset")
os.makedirs("dataset/legitimate", exist_ok=True)
os.makedirs("dataset/phishing", exist_ok=True)

# --- LEGITIMATE payloads: wide variety, grouped by type ---
legitimate_urls = [
    "https://www.google.com", "https://www.wikipedia.org", "https://www.amazon.in",
    "https://www.flipkart.com", "https://www.github.com", "https://www.linkedin.com",
    "https://www.irctc.co.in", "https://www.icicibank.com", "https://www.hdfcbank.com",
    "https://www.swiggy.com", "https://www.zomato.com", "https://www.myntra.com",
]
legitimate_upi = [
    "upi://pay?pa=merchant@okaxis&pn=CoffeeShop&am=150",
    "upi://pay?pa=shopowner@ybl&pn=Grocery&am=899",
    "upi://pay?pa=cafe@oksbi&pn=CafeCorner&am=220",
    "upi://pay?pa=pharmacy@okicici&pn=MedStore&am=340",
    "upi://pay?pa=bookstore@okhdfcbank&pn=Books&am=599",
    "upi://pay?pa=restaurant@paytm&pn=Dinner&am=780",
]
legitimate_wifi = [
    "WIFI:S:HomeNetwork;T:WPA;P:mypassword123;;",
    "WIFI:S:OfficeWifi;T:WPA;P:office2024;;",
    "WIFI:S:LibraryWifi;T:WPA2;P:library2025;;",
    "WIFI:S:CafeGuest;T:WPA;P:welcome123;;",
]
legitimate_text = [
    "Contact: John Doe, +91-9876543210",
    "Event: Tech Fest 2026, Hall A, 10 AM",
    "Menu: Pasta - 250, Pizza - 350, Salad - 180",
    "Parking Ticket: Slot A4, Valid till 6 PM",
]

# --- PHISHING-style payloads: wide variety, grouped by type ---
phishing_urls = [
    "http://192.168.45.12/verify-account",
    "https://www.axiisbank-secure.com/login",
    "https://paypa1-verify.com/reset",
    "http://10.0.0.5:8080/update-kyc",
    "https://icici-bank-kyc.tk/verify",
    "https://secure-hdfc.xyz/login",
    "http://bit.ly/3xFakeLink",
    "https://amaz0n-rewards.com/claim",
    "https://www.faceb00k-security.com/confirm",
    "http://sbi-refund-claim.ml/process",
]
phishing_upi = [
    "upi://pay?pa=scammer123@fakebank&pn=URGENT&am=5000",
    "upi://pay?pa=refund-claim@xyz&pn=RBI_Refund&am=9999",
    "upi://pay?pa=winner-lottery@paytm&pn=ClaimPrize&am=1",
    "upi://pay?pa=kyc-update@fake&pn=KYCVerify&am=1",
    "upi://pay?pa=cashback-offer@scam&pn=Cashback&am=1",
]
phishing_wifi = [
    "WIFI:S:Free_Airport_WiFi;T:nopass;;",
    "WIFI:S:Free_Public_WiFi;T:nopass;;",
    "WIFI:S:Guest_Network_Open;T:nopass;;",
]
phishing_text = [
    "URGENT: Your account is locked. Verify now: bit.ly/3xFake",
    "WINNER! Claim your prize within 24hrs or lose it",
    "Your parcel is held. Pay customs fee: scam-link.com",
    "Bank Alert: Suspicious activity. Confirm identity now.",
]

legit_by_type = {
    "url": legitimate_urls,
    "upi": legitimate_upi,
    "wifi": legitimate_wifi,
    "text": legitimate_text,
}
phish_by_type = {
    "url": phishing_urls,
    "upi": phishing_upi,
    "wifi": phishing_wifi,
    "text": phishing_text,
}

payload_types = ["url", "upi", "wifi", "text"]
error_levels = [ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H]
error_level_names = {ERROR_CORRECT_L: "L", ERROR_CORRECT_M: "M", ERROR_CORRECT_Q: "Q", ERROR_CORRECT_H: "H"}

def generate_qr(payload, folder, index):
    box_size = random.choice([6, 8, 10, 12, 14])
    border = random.choice([2, 3, 4, 5])
    error_correction = random.choice(error_levels)

    # version=None + fit=True lets the library pick the correct smallest version automatically
    qr = qrcode.QRCode(version=None, error_correction=error_correction, box_size=box_size, border=border)
    qr.add_data(payload)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    filename = f"{folder}_{index}.png"
    filepath = os.path.join("dataset", folder, filename)
    img.save(filepath)

    return filename, qr.version, error_level_names[error_correction]

metadata = []
IMAGES_PER_TYPE_PER_CLASS = 40  # 40 x 4 types x 2 classes = 320 total, evenly balanced

for ptype in payload_types:
    for i in range(IMAGES_PER_TYPE_PER_CLASS):
        payload = random.choice(legit_by_type[ptype])
        filename, version, ec_level = generate_qr(payload, "legitimate", f"{ptype}_{i}")
        metadata.append([filename, "legitimate", ptype, payload, version, ec_level])

    for i in range(IMAGES_PER_TYPE_PER_CLASS):
        payload = random.choice(phish_by_type[ptype])
        filename, version, ec_level = generate_qr(payload, "phishing", f"{ptype}_{i}")
        metadata.append([filename, "phishing", ptype, payload, version, ec_level])

with open("dataset/metadata.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label", "payload_type", "payload", "qr_version", "error_correction"])
    writer.writerows(metadata)

print(f"Generated {len(metadata)} QR codes total.")
print(f"({IMAGES_PER_TYPE_PER_CLASS} images x 4 payload types x 2 classes)")
print("Metadata saved to dataset/metadata.csv")