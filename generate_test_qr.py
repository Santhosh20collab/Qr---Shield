import qrcode

# Create a QR code encoding a legitimate-looking URL
qr = qrcode.QRCode(version=1, box_size=10, border=4)
qr.add_data("https://www.example-bank.com/login")
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("test_qr.png")

print("QR code saved as test_qr.png")