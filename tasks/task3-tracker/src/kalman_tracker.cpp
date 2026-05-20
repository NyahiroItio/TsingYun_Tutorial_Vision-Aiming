#include "kalman_tracker.hpp"

#include <stdexcept>

namespace hw
{
    KalmanTracker::KalmanTracker() = default;

    bool KalmanTracker::isTracking() const
    {
        return tracking_;
    }

    void KalmanTracker::reset()
    {
        tracking_ = false;
        x_ = AxisFilter{};
        y_ = AxisFilter{};
        z_ = AxisFilter{};
    }

    void KalmanTracker::AxisFilter::reset(double measured_position)
    {
        position = measured_position;
        velocity = 0.0;
        p00 = 1.0;
        p01 = 0.0;
        p10 = 0.0;
        p11 = 1.0;
    }

    void KalmanTracker::AxisFilter::predict(double dt, double process_noise)
    {
        // TODO(student): Implement the constant-velocity Kalman predict step.
        // dt = max(dt, 0)
        dt = dt > 0 ? dt : 0;
        position = position + velocity * dt;
        // F = [[1, dt],
        //      [0, 1]]
        // Q = process_noise * [[dt^4 / 4, dt^3 / 2],
        //                      [dt^3 / 2, dt^2]]
        // P = F * P * F^T + Q
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        double dt4 = dt3 * dt;

        double q00 = process_noise * (dt4 / 4.0);
        double q01 = process_noise * (dt3 / 2.0);
        double q10 = process_noise * (dt3 / 2.0);
        double q11 = process_noise * dt2;

        double p00_new = p00 + dt * (p10 + p01) + dt2 * p11 + q00;
        double p01_new = p01 + dt * p11 + q01;
        double p10_new = p10 + dt * p11 + q10;
        double p11_new = p11 + q11;

        p00 = p00_new;
        p01 = p01_new;
        p10 = p10_new;
        p11 = p11_new;
        // store the updated position, velocity, and covariance
        return;
        throw std::logic_error("NotImplementedError: KalmanTracker::AxisFilter::predict");
    }

    void KalmanTracker::AxisFilter::update(double measured_position, double measurement_noise)
    {
        // TODO(student): Implement the 1D position measurement update step.
        double residual = measured_position - position;
        // H = [1, 0]
        // S = H * P * H^T + measurement_noise
        double S = p00 + measurement_noise;
        // if S is not positive:
        //     return without updating
        if (S <= 0)
        {
            return;
        }
        // K = P * H^T / S
        double K0 = p00 / S;
        double K1 = p10 / S;
        // position = position + K[0] * residual
        // velocity = velocity + K[1] * residual
        position += K0 * residual;
        velocity += K1 * residual;
        // P = (I - K * H) * P
        double p00_new = (1.0 - K0) * p00;
        double p01_new = (1.0 - K0) * p01;
        double p10_new = p10 - K1 * p00;
        double p11_new = p11 - K1 * p01;

        p00 = p00_new;
        p01 = p01_new;
        p10 = p10_new;
        p11 = p11_new;
        return;
        throw std::logic_error("NotImplementedError: KalmanTracker::AxisFilter::update");
    }

    TrackState KalmanTracker::update(const Vec3 &measurement, double dt)
    {
        // TODO(student): Update tracker state from one measured 3D point.
        // if tracker is not initialized:
        //     initialize x, y, z filters with measurement components
        //     set all velocities to zero
        //     mark tracker as active
        //     return current state
        if(!tracking_){
            x_.reset(measurement.x);
            y_.reset(measurement.y);
            z_.reset(measurement.z);
            tracking_ = true;
            return stateFromFilters();
        }
        // predict each axis filter using dt
        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);
        // update each axis filter with its measured coordinate
        x_.update(measurement.x, measurement_noise_);
        y_.update(measurement.y, measurement_noise_);
        z_.update(measurement.z, measurement_noise_);
        // return position, velocity, and tracking flag
        return stateFromFilters();
        throw std::logic_error("NotImplementedError: KalmanTracker::update");
    }

    TrackState KalmanTracker::predict(double dt)
    {
        // TODO(student): Predict target state when a detection is missing.
        // if tracker is not active:
        //     return a non-tracking state
        if(!tracking_){
            return TrackState{false, {{0.0}, {0.0}, {0.0}}, {{0.0}, {0.0}, {0.0}}};
        }
        // predict x, y, z filters with dt
        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);
        // return predicted position and velocity
        return stateFromFilters();
        throw std::logic_error("NotImplementedError: KalmanTracker::predict");
    }

    TrackState KalmanTracker::stateFromFilters() const
    {
        return {
            true,
            {x_.position, y_.position, z_.position},
            {x_.velocity, y_.velocity, z_.velocity},
        };
    }
} // namespace hw
