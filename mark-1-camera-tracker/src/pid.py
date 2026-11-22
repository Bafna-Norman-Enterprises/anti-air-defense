import time


class PIDController:
    """
    Precision PID controller for pan/tilt optical tracking.
    Includes deadband, integral anti-windup, derivative low-pass filtering,
    and output rate limiting for smooth, jitter-free servo motion.
    """

    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        deadband: float = 3.0,
        output_limit: float = 15.0,
        integral_limit: float = 50.0,
        d_filter_alpha: float = 0.35,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.deadband = deadband              # Ignore tiny errors inside this pixel radius
        self.output_limit = output_limit      # Max servo step per update (degrees)
        self.integral_limit = integral_limit  # Anti-windup cap
        self.d_filter_alpha = d_filter_alpha  # Smooths noisy derivative spikes (0..1)

        self.reset()

    def reset(self):
        """Clears controller memory. Call this whenever target lock is lost."""
        self.integral = 0.0
        self.prev_error = None
        self.filtered_d = 0.0
        self.last_time = None

    def update(self, error: float, dt: float = None, feedforward: float = 0.0) -> float:
        """
        Calculates the smooth servo adjustment (in degrees) from the current tracking error
        and optional velocity feedforward rate.
        
        feedforward: Pre-calculated forward adjustment (in degrees) based on target angular velocity.
                     Bypasses feedback lag so the camera slews at target speed.
        """
        now = time.perf_counter()
        if dt is None:
            if self.last_time is None:
                dt = 1.0 / 60.0
            else:
                dt = max(min(now - self.last_time, 0.1), 1e-4)
        self.last_time = now

        # 1. Deadband: if error is within deadband, feedback terms are zeroed, but feedforward still tracks
        in_deadband = abs(error) <= self.deadband
        if in_deadband:
            self.prev_error = error
            self.filtered_d *= 0.5  # Gently decay derivative while holding
            p_term = 0.0
            i_term = self.ki * self.integral
            d_term = 0.0
        else:
            # 2. Proportional term (immediate response)
            p_term = self.kp * error

            # 3. Integral term (eliminates steady-state drift) with anti-windup clamp
            self.integral += error * dt
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
            i_term = self.ki * self.integral

            # 4. Derivative term (brakes against overshoot) with low-pass noise filter
            if self.prev_error is None:
                raw_d = 0.0
            else:
                raw_d = (error - self.prev_error) / dt
            self.prev_error = error

            self.filtered_d = (
                self.d_filter_alpha * raw_d + (1.0 - self.d_filter_alpha) * self.filtered_d
            )
            d_term = self.kd * self.filtered_d

        # 5. Combine feedback terms with feedforward rate
        output = p_term + i_term + d_term + feedforward
        output = max(-self.output_limit, min(self.output_limit, output))
        return output


# ==========================================
# STANDALONE BENCHMARK TEST
# ==========================================
if __name__ == "__main__":
    # Simulate a single pan servo starting at 90.0° chasing a target at 120.0°
    pid = PIDController(kp=0.18, ki=0.05, kd=0.015, deadband=0.1, output_limit=5.0)
    servo_angle = 90.0
    target_angle = 120.0
    dt = 1.0 / 60.0  # Simulated 60 Hz loop

    print("[PID TEST] Starting at 90.0 deg -> Chasing target at 120.0 deg")
    print("-" * 58)
    for frame in range(1, 61):
        error = target_angle - servo_angle
        step = pid.update(error, dt=dt)
        servo_angle += step

        if frame in (1, 5, 10, 15, 20, 30, 45, 60):
            print(
                f"Frame {frame:02d} ({frame*dt:0.2f}s) | "
                f"Angle: {servo_angle:6.2f} deg | Error: {error:+6.2f} deg | Step: {step:+5.2f} deg"
            )
            