"""
main.py — Smart Vision AI Entry Point
"""
import customtkinter as ctk
from ui import SmartVisionApp

def start():
    print("=" * 60)
    print("  SMART VISION AI — ENTERPRISE EDITION")
    print("  DB1: Persons (Permanent Face Registry)")
    print("  DB2: Objects (Visual DNA, no photos)")
    print("  DB3: Activity (Behaviour Patterns)")
    print("  SEC: AES-256 + SHA-256 Tamper Detection")
    print("=" * 60)

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app  = SmartVisionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    start()