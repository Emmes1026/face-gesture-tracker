import time
from time import sleep
import threading

try:
    from adafruit_servokit import ServoKit # type: ignore
    SERVOKIT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SERVOKIT_AVAILABLE = False
    print("Warning: Library 'adafruit_servokit' not found.")
except Exception as e:
    SERVOKIT_AVAILABLE = False
    print(f"Import error adafruit_servokit: {e}")

GPIOZERO_AVAILABLE = SERVOKIT_AVAILABLE


class ServoTracker:
    def __init__(self, pan_channel=0, tilt_channel=4, pan_gain=0.015, tilt_gain=0.015):
        self.kit = None
        self.active = SERVOKIT_AVAILABLE
        self.pan_channel = pan_channel
        self.tilt_channel = tilt_channel

        self.deadcenter = 60
        self.smooth_factor = 0.15

        if not self.active: return

        self.pan_min_angle = 10.0
        self.pan_max_angle = 170.0
        self.tilt_min_angle = 20.0
        self.tilt_max_angle = 120.0

        self.pan_center_angle = (self.pan_min_angle + self.pan_max_angle) / 2.0
        self.tilt_center_angle = 60

        self.current_pan = self.pan_center_angle
        self.current_tilt = self.tilt_center_angle

        self.target_pan = self.pan_center_angle
        self.target_tilt = self.tilt_center_angle

        self.pan_gain = pan_gain
        self.tilt_gain = tilt_gain
        self.running = True

        try:
            self.kit = ServoKit(channels=16)
            self.kit.servo[self.pan_channel].actuation_range = 180
            self.kit.servo[self.pan_channel].set_pulse_width_range(500, 2500)
            self.kit.servo[self.tilt_channel].actuation_range = 180
            self.kit.servo[self.tilt_channel].set_pulse_width_range(500, 2500)

            self._force_move(self.current_pan, self.current_tilt)
            sleep(0.5)

            self.thread = threading.Thread(target=self._smooth_movement_loop, daemon=True)
            self.thread.start()

        except Exception as e:
            print(f"Fatal error ServoKit: {e}")
            self.active = False

    def update(self, target_x, target_y, frame_width, frame_height):
        if not self.active or self.kit is None: return

        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0

        error_pan = target_x - frame_center_x
        error_tilt = target_y - frame_center_y

        if abs(error_pan) < self.deadcenter: error_pan = 0
        if abs(error_tilt) < self.deadcenter: error_tilt = 0

        desired_pan = self.target_pan + (error_pan * self.pan_gain)
        desired_tilt = self.target_tilt + (error_tilt * self.tilt_gain)

        self.target_pan = max(self.pan_min_angle, min(self.pan_max_angle, desired_pan))
        self.target_tilt = max(self.tilt_min_angle, min(self.tilt_max_angle, desired_tilt))

    def _smooth_movement_loop(self):
        last_written_pan = -100
        last_written_tilt = -100

        while self.running and self.active:
            diff_pan = self.target_pan - self.current_pan
            diff_tilt = self.target_tilt - self.current_tilt

            if abs(diff_pan) > 0.05: self.current_pan += diff_pan * self.smooth_factor
            if abs(diff_tilt) > 0.05: self.current_tilt += diff_tilt * self.smooth_factor

            should_update = False
            if abs(self.current_pan - last_written_pan) > 0.5:
                should_update = True
            if abs(self.current_tilt - last_written_tilt) > 0.5:
                should_update = True

            if should_update:
                self._force_move(self.current_pan, self.current_tilt)
                last_written_pan = self.current_pan
                last_written_tilt = self.current_tilt

            time.sleep(0.02)

    def _force_move(self, pan, tilt):
        try:
            self.kit.servo[self.pan_channel].angle = pan
            self.kit.servo[self.tilt_channel].angle = tilt
        except:
            pass

    def cleanup(self):
        if not self.active: return

        self.target_pan = self.pan_center_angle
        self.target_tilt = self.tilt_center_angle
        
        start_wait = time.time()
        while time.time() - start_wait < 2.0:
            dist_pan = abs(self.current_pan - self.target_pan)
            dist_tilt = abs(self.current_tilt - self.target_tilt)
            
            if dist_pan < 1.0 and dist_tilt < 1.0:
                break
            
            time.sleep(0.1)

        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

        try:
            self.kit.servo[self.pan_channel].angle = None
            self.kit.servo[self.tilt_channel].angle = None
        except:
            pass
