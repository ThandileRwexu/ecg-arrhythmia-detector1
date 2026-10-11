"""
Simulated phone that streams ECG beats to the Streamlit app.

Usage:
    python phone_simulator.py <TOKEN>
    python phone_simulator.py <TOKEN> --url http://192.168.1.50:8765/ecg --count 50
    (or set ECG_API_TOKEN in the environment instead of passing <TOKEN>)
"""
import argparse
import os
import random
import time

import numpy as np
import requests

NORMAL = [0.1, 0.2, 0.5, 0.8, 1.0, 0.9, 0.4, 0.2, 0.1, 0.05]
ABNORMAL = [0.1, 0.6, 0.3, 1.2, 0.7, 1.5, 0.2, 0.4, 0.8, 0.1]


def generate_ecg():
    base = ABNORMAL if random.random() < 0.2 else NORMAL
    beat_10 = [v + (random.random() - 0.5) * 0.05 for v in base]
    beat_180 = np.interp(np.linspace(0, 1, 180), np.linspace(0, 1, 10), beat_10)
    return beat_180.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("token", nargs="?", default=os.environ.get("ECG_API_TOKEN"))
    p.add_argument("--url", default="http://localhost:8765/ecg")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()

    if not args.token:
        p.error("token required (positional arg or ECG_API_TOKEN env var)")

    print(f"📱 Simulating phone → {args.url}")
    print(f"🔐 Token: {args.token[:6]}...  Press Ctrl+C to stop.\n")

    try:
        for n in range(1, args.count + 1):
            try:
                r = requests.post(args.url, json={"ecg": generate_ecg()},
                                  headers={"X-Auth-Token": args.token}, timeout=5)
                if r.status_code == 401:
                    print("❌ 401 Unauthorised — copy the CURRENT token from the app sidebar.")
                    return
                print(f"📦 Packet {n} | HTTP {r.status_code} | {r.text[:200]}")
            except requests.RequestException as e:
                print(f"❌ Connection error: {e}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n✅ Stopped by user.")
    print("✅ Simulation complete.")


if __name__ == "__main__":
    main()