import cv2
from collections import Counter
from module import findnameoflandmark, findpostion
from pymavlink import mavutil
import time
import sys

# Constants
COMMAND_COOLDOWN = 5.0  # Minimum seconds between commands
CONNECTION_TIMEOUT = 10  # Seconds to wait for connection

def connect_to_sitl():
    """Establish connection to SITL with error handling"""
    try:
        print("Connecting to SITL...")
        master = mavutil.mavlink_connection('tcp:192.168.1.5:5762', autoreconnect=True)
        
        # Wait for heartbeat with timeout
        print("Waiting for heartbeat...")
        start_time = time.time()
        while time.time() - start_time < CONNECTION_TIMEOUT:
            msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if msg:
                print(f"Heartbeat received (system {master.target_system}, component {master.target_component})")
                return master
            print(".", end="", flush=True)
        
        raise ConnectionError("Heartbeat timeout")
    
    except Exception as e:
        print(f"\nConnection failed: {str(e)}")
        return None

def send_command(master, command_type, params, timeout=3):
    """Generic command sender with error handling"""
    try:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            command_type,
            0,  # confirmation
            *params
        )
        
        # Wait for acknowledgment
        start_time = time.time()
        while time.time() - start_time < timeout:
            ack_msg = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
            if ack_msg and ack_msg.command == command_type:
                if ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    return True
                else:
                    print(f"Command failed with result: {ack_msg.result}")
                    return False
        print("Command acknowledgment timeout")
        return False
        
    except Exception as e:
        print(f"Command error: {str(e)}")
        return False

def main():
    """Main program execution"""
    # Establish connection
    master = connect_to_sitl()
    if not master:
        print("Failed to connect to SITL", file=sys.stderr)
        return 1

    # Camera setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera", file=sys.stderr)
        master.close()
        return 1

    tip = [8, 12, 16, 20]
    tipname = [8, 12, 16, 20]
    last_command_time = 0

    try:
        while True:
            # Frame processing
            ret, frame = cap.read()
            if not ret:
                print("Camera frame error", file=sys.stderr)
                time.sleep(0.1)
                continue
                
            frame = cv2.resize(frame, (640, 480))
            
            # Gesture detection
            a = findpostion(frame)
            b = findnameoflandmark(frame)
            
            if not (a and b):  # Check if both are non-empty
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
                
            # Finger counting logic
            finger = [1 if a[0][1:] < a[4][1:] else 0]
            fingers = [1 if a[tip[id]][2:] < a[tip[id]-2][2:] else 0 for id in range(4)]
            
            up = sum(fingers + finger)
            print(f"Fingers up: {up}, down: {5 - up}")

            # Command execution with cooldown
            current_time = time.time()
            if current_time - last_command_time < COMMAND_COOLDOWN:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
                
            try:
                if up == 5:
                    print('Disarming')
                    if send_command(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, [0, 0, 0, 0, 0, 0, 0]):
                        last_command_time = current_time
                        
                elif up == 4:
                    print('Arming')
                    if (send_command(master, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 
                                    [mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 
                                     master.mode_mapping()['GUIDED'], 0, 0, 0, 0, 0]) and
                        send_command(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, [1, 0, 0, 0, 0, 0, 0])):
                        last_command_time = current_time
                        
                elif up == 3:
                    print('Takeoff')
                    if (send_command(master, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 
                                    [mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 
                                     master.mode_mapping()['GUIDED'], 0, 0, 0, 0, 0]) and
                        send_command(master, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, [0, 0, 0, 0, 0, 0, 10])):
                        last_command_time = current_time
                        
                elif up == 2:
                    print('Yaw 90° right')
                    if send_command(master, mavutil.mavlink.MAV_CMD_CONDITION_YAW, [90, 30, 1, 0, 0, 0, 0]):
                        last_command_time = current_time
                        
                elif up == 1:
                    print('Circle mode')
                    if (send_command(master, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 
                                    [mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 
                                     master.mode_mapping()['CIRCLE'], 0, 0, 0, 0, 0]) and
                        send_command(master, mavutil.mavlink.MAV_CMD_DO_GO_AROUND, [15, 0, 0, 0, 0, 0, 0])):
                        last_command_time = current_time
                        
                elif up == 0:
                    print('Landing')
                    if send_command(master, mavutil.mavlink.MAV_CMD_NAV_LAND, [0, 0, 0, 0, 0, 0, 0]):
                        last_command_time = current_time
                
            except Exception as e:
                print(f"Command execution error: {str(e)}", file=sys.stderr)
                
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Runtime error: {str(e)}", file=sys.stderr)
    finally:
        print("Cleaning up...")
        try:
            # Land and disarm if still armed
            if master.motors_armed():
                send_command(master, mavutil.mavlink.MAV_CMD_NAV_LAND, [0, 0, 0, 0, 0, 0, 0])
                time.sleep(5)
                send_command(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, [0, 0, 0, 0, 0, 0, 0])
        except Exception as e:
            print(f"Cleanup error: {str(e)}", file=sys.stderr)
        
        cap.release()
        cv2.destroyAllWindows()
        master.close()
        print("Shutdown complete")
        return 0

if __name__ == "__main__":
    sys.exit(main())