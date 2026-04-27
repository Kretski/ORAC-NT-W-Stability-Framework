"""
ORAC-NT Live Listener
Connects to NASA/LIGO GCN via Kafka and listens for gravitational wave events.
"""

from gcn_kafka import Consumer
import json
import time

# =====================================================
# 1. GCN AUTHENTICATION
# =====================================================
# Взети директно от твоя профил в NASA:
CLIENT_ID = '1hqfpp5nqq1hbirp40i2cnonfq'
CLIENT_SECRET = '1lnc4et60a8lh5nqrqgp4brku0cni3b41cko4cmvdqqgvjmj4mh7'

print("📡 Initializing connection to NASA GCN...")

try:
    consumer = Consumer(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    
   # Subscribe to real events only (mock topic is currently unavailable on NASA's end)
    consumer.subscribe(['igwn.gwalert'])
    print("✅ Connection successful! ORAC-NT is now listening...")
    print("Press Ctrl+C to stop the listener.")
except Exception as e:
    print(f"❌ Connection error: {e}")
    exit()

# =====================================================
# 2. ENDLESS LISTENING LOOP (24/7)
# =====================================================
try:
    while True:
        # Using timeout=1 as per official documentation for safe interrupts
        for message in consumer.consume(timeout=1):
            if message.error():
                print(f"Kafka error: {message.error()}")
                continue
                
            try:
                # Decode the JSON payload from NASA
                payload = json.loads(message.value())
                
                # We are only interested in new INITIAL detections
                alert_type = payload.get('alert_type')
                
                if alert_type == 'INITIAL':
                    event_id = payload.get('superevent_id', 'UNKNOWN')
                    event_time = payload['event'].get('time') # GPS time
                    
                    # LIGO sends initial classification and property estimates
                    classification = payload['event'].get('classification', {})
                    bns_prob = classification.get('BNS', 0.0) # Binary Neutron Star probability
                    bbh_prob = classification.get('BBH', 0.0) # Binary Black Hole probability
                    
                    print("\n" + "="*60)
                    print(f"🚨🚨 SPACE ALERT: EVENT {event_id} DETECTED! 🚨🚨")
                    print("="*60)
                    print(f"⏱️ GPS Time: {event_time}")
                    print(f"📊 Probabilities: {bbh_prob*100:.1f}% Black Holes | {bns_prob*100:.1f}% Neutron Stars")
                    print("⚙️ Saving data for ORAC-NT pipeline...")
                    
                    # Save the event data to a local file for the Streamlit UI to read
                    with open("latest_event.json", "w") as f:
                        json.dump({
                            "id": event_id,
                            "gps_time": event_time,
                            "bbh_prob": bbh_prob,
                            "bns_prob": bns_prob
                        }, f)
                    
                    print("✅ Data saved to latest_event.json! You can now run the triangulation.")
                        
            except json.JSONDecodeError:
                pass # Ignore malformed packets
            except Exception as e:
                print(f"⚠️ Error processing message: {e}")

except KeyboardInterrupt:
    print("\n🛑 Listener stopped manually.")
finally:
    consumer.close()