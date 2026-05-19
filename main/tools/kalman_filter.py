import numpy as np

class KalmanDepthTracker:
    """
    Kalman Filter for smoothing noisy depth measurements.

    State Model
    -----------
    state =
        [ depth,
          depth_velocity ]

    The filter predicts future depth using velocity and
    corrects predictions using incoming measurements.
    """

    def __init__(self):

        # =========================================================
        # STATE VECTOR
        # =========================================================
        # [ depth,
        #   depth_velocity ]
        # =========================================================
        self.state_vector = np.array([
            [0.0],   # depth
            [0.0]    # velocity
        ])

        # =========================================================
        # STATE UNCERTAINTY (Covariance Matrix)
        # =========================================================
        # Large initial uncertainty because no measurements
        # have been processed yet.
        # =========================================================
        self.state_covariance = np.eye(2) * 1000

        # =========================================================
        # TIME STEP
        # =========================================================
        # Time between frames.
        # For 30 FPS:
        # dt = 1 / 30
        # =========================================================
        delta_time = 1.0

        # =========================================================
        # STATE TRANSITION MATRIX
        # =========================================================
        # Predicts next state using:
        #
        # new_depth = old_depth + velocity * dt
        # new_velocity = old_velocity
        # =========================================================
        self.motion_model = np.array([
            [1, delta_time],
            [0, 1]
        ])

        # =========================================================
        # MEASUREMENT MATRIX
        # =========================================================
        # Maps state -> measurable quantity.
        #
        # We only measure depth.
        # Velocity is hidden/internal.
        # =========================================================
        self.measurement_model = np.array([
            [1, 0]
        ])

        # =========================================================
        # MEASUREMENT NOISE
        # =========================================================
        # Higher value:
        #   smoother output
        #   slower response
        #
        # Lower value:
        #   faster response
        #   noisier output
        # =========================================================
        self.measurement_noise = np.array([
            [25]
        ])

        # =========================================================
        # PROCESS NOISE
        # =========================================================
        # Represents uncertainty in motion prediction.
        #
        # Higher:
        #   adapts quickly to motion changes
        #
        # Lower:
        #   smoother but slower adaptation
        # =========================================================
        self.process_noise = np.array([
            [0.05, 0],
            [0, 0.05]
        ])

        # =========================================================
        # FILTER INITIALIZATION FLAG
        # =========================================================
        self.has_received_first_measurement = False

    def update(self, measured_depth):

        # =========================================================
        # PREDICTION STEP
        # =========================================================

        # Predict next state
        self.state_vector = (
            self.motion_model
            @ self.state_vector
        )

        # Predict next uncertainty
        self.state_covariance = (
            self.motion_model
            @ self.state_covariance
            @ self.motion_model.T
            + self.process_noise
        )

        # =========================================================
        # HANDLE MISSING MEASUREMENT
        # =========================================================
        # If object tracking fails temporarily,
        # return predicted depth only.
        # =========================================================
        if measured_depth is None:

            predicted_depth = float(
                self.state_vector[0, 0]
            )

            return predicted_depth

        # =========================================================
        # FIRST VALID MEASUREMENT
        # =========================================================
        if not self.has_received_first_measurement:

            self.state_vector[0, 0] = measured_depth
            self.state_vector[1, 0] = 0.0

            self.has_received_first_measurement = True

            return measured_depth

        # =========================================================
        # MEASUREMENT VECTOR
        # =========================================================
        measurement_vector = np.array([
            [measured_depth]
        ])

        # =========================================================
        # INNOVATION / RESIDUAL
        # =========================================================
        # Difference between:
        # actual measurement
        # and predicted measurement
        # =========================================================
        prediction_error = (
            measurement_vector
            - self.measurement_model @ self.state_vector
        )

        # =========================================================
        # INNOVATION COVARIANCE
        # =========================================================
        prediction_uncertainty = (
            self.measurement_model
            @ self.state_covariance
            @ self.measurement_model.T
            + self.measurement_noise
        )

        # =========================================================
        # KALMAN GAIN
        # =========================================================
        # Controls balance between:
        # prediction vs measurement
        # =========================================================
        kalman_gain = (
            self.state_covariance
            @ self.measurement_model.T
            @ np.linalg.inv(prediction_uncertainty)
        )

        # =========================================================
        # STATE UPDATE
        # =========================================================
        self.state_vector = (
            self.state_vector
            + kalman_gain @ prediction_error
        )

        # =========================================================
        # COVARIANCE UPDATE
        # =========================================================
        identity_matrix = np.eye(2)

        self.state_covariance = (
            identity_matrix
            - kalman_gain @ self.measurement_model
        ) @ self.state_covariance

        # =========================================================
        # RETURN FILTERED DEPTH
        # =========================================================
        filtered_depth = float(
            self.state_vector[0, 0]
        )

        return filtered_depth


# =========================================================
# EXAMPLE USAGE
# =========================================================

# depth_tracker = KalmanDepthTracker()

# raw_depth = 125.7

# filtered_depth = depth_tracker.update(raw_depth)

# print(filtered_depth)