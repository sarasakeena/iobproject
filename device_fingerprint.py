import hashlib
import json
import random

class DeviceFingerprint:
    def __init__(self):
        self.device_data = {}
        self.user_data = {}  # Simulated user data storage

    def capture_device_info(self):
        # Simulate capturing device information
        device_info = {
            'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'screen_resolution': '1920x1080',
            'timezone': 'UTC',
            'language': 'en-US',
            'plugins': ['Chrome PDF Viewer', 'Shockwave Flash'],
            'ip_address': '192.168.1.1',  # Example IP
            'device_type': 'Desktop',
            'operating_system': 'macOS',
            'mac_address': '00:1A:2B:3C:4D:5E'  # Example MAC address
        }
        return device_info

    def generate_fingerprint(self, device_info):
        # Create a unique fingerprint using a hash function
        fingerprint_string = json.dumps(device_info, sort_keys=True)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()

    def store_fingerprint(self, user_id, fingerprint):
        self.user_data[user_id] = {'fingerprint': fingerprint, 'mac_address': fingerprint['mac_address']}

    def is_recognized_device(self, user_id, new_fingerprint):
        if user_id in self.user_data:
            return self.user_data[user_id]['fingerprint'] == new_fingerprint
        return False

    def is_risky_location(self, ip_address):
        # Simulate risky location check (for example, using a predefined list of risky IPs)
        risky_ips = ['192.168.1.100', '10.0.0.1']  # Example risky IPs
        return ip_address in risky_ips

    def verify_mac_address(self, user_id, mac_address):
        if user_id in self.user_data:
            return self.user_data[user_id]['mac_address'] == mac_address
        return False

    def evaluate_risk(self, login_attempt):
        risk_score = 0
        
        if not self.is_recognized_device(login_attempt['user_id'], login_attempt['fingerprint']):
            risk_score += 50  # High risk for new devices
        
        if self.is_risky_location(login_attempt['ip_address']):
            risk_score += 30  # Medium risk for risky locations
        
        if not self.verify_mac_address(login_attempt['user_id'], login_attempt['mac_address']):
            risk_score += 20  # Additional risk for unrecognized MAC addresses
        
        return risk_score

    def handle_login(self, login_attempt):
        # Grant access regardless of device fingerprint
        print(f"Access granted to user {login_attempt['user_id']}.")

# Example usage
device_fingerprint = DeviceFingerprint()

# Simulate storing a user's device fingerprint
user_id = 'user123'
device_info = device_fingerprint.capture_device_info()
fingerprint = device_fingerprint.generate_fingerprint(device_info)
device_fingerprint.store_fingerprint(user_id, device_info)

# Simulate a new login attempt from an unrecognized device
login_attempt = {
    'user_id': 'sri',
    'fingerprint': 'new_fingerprint',  # New device fingerprint
    'ip_address': '192.168.1.1',  # Same IP
    'mac_address': '00:1A:2B:3C:4D:5E'  # Same MAC
}

# Handle the login attempt
device_fingerprint.handle_login(login_attempt)