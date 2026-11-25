"""Run one Stewart platform trajectory using the kinematics in kin.py.

Node:      /trajectory
Publishes: /joint_states  sensor_msgs/JointState
           /pose          geometry_msgs/PoseStamped
           /twist         geometry_msgs/TwistStamped
           /condition     std_msgs/Float64
           /tf            base_link -> desired

Start robot_state_publisher with stewart.urdf before running this node.
"""

from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.task import Future
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Header
from tf2_ros import TransformBroadcaster

from utils.TransformHelpers import (
    Pose_from_T, T_from_Rp, Transform_from_T, Twist_from_vw, crossmat,
)

if __package__:
    from .kin import JointType, StewartChains
else:
    from kin import JointType, StewartChains


@dataclass
class Sample:
    desired: np.ndarray
    actual: np.ndarray
    twist: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    condition: float


def desired_motion(t, duration, center):
    """Compute the pose and base-frame twist for a circle with a pitch oscillation."""
    u = t / duration
    # Smooth timing starts and ends with zero velocity and acceleration.
    phase = 2 * np.pi * u**3 * (10 - 15*u + 6*u**2)
    phase_rate = 60 * np.pi * u**2 * (1 - u)**2 / duration

    radius = 0.012  # m
    pitch = np.deg2rad(3)
    c, s = np.cos(phase), np.sin(phase)
    p = center[:3, 3] + radius * np.array([c, s, 0.0])
    R = Rotation.from_rotvec([0.0, pitch*s, 0.0]).as_matrix() @ center[:3, :3]
    v = radius * phase_rate * np.array([-s, c, 0.0])
    w = np.array([0.0, pitch * phase_rate * c, 0.0])
    return T_from_Rp(R, p), np.r_[v, w]


def plan_trajectory(model, times):
    """Solve and verify the full trajectory before sending joint states."""
    leg_joints = [
        [step for step in chain.chain if step.dof is not None]
        for chain in model.legchains
    ]
    joint_names = [step.name for joints in leg_joints for step in joints]
    slider_indices = [
        next(step.dof for step in joints if step.type == JointType.LINEAR)
        for joints in leg_joints
    ]

    # A mid-stroke pose leaves room to move in either slider direction.
    sliders = 0.5 * (model.slider_lower + model.slider_upper)
    center, theta = model.fkin(sliders)
    samples = []
    for t in times:
        desired, twist = desired_motion(t, times[-1], center)
        try:
            sliders, theta = model.ikin(desired, sliders, theta)
            actual, theta = model.fkin(sliders, theta)
            position_error = np.linalg.norm(actual[:3, 3] - desired[:3, 3])
            angle_error = Rotation.from_matrix(
                desired[:3, :3].T @ actual[:3, :3]
            ).magnitude()
            if position_error > model.position_tolerance or angle_error > model.angle_tolerance:
                raise ValueError(f"FK pose error is {position_error:.3g} m and {angle_error:.3g} rad")

            position = np.empty((6, 6))
            velocity = np.empty((6, 6))
            slider_jacobian = np.empty((6, 6))
            for i, slider_index in enumerate(slider_indices):
                position[i] = np.insert(theta[i], slider_index, sliders[i])
                leg_tip, Jv, Jw = model.leg_fkin(i, sliders, theta)

                # The leg tip moves with TOP1. Shift the linear velocity
                # from the platform origin to the attachment point.
                offset = leg_tip[:3, 3] - actual[:3, 3]
                tip_twist = np.eye(6)
                tip_twist[:3, 3:] = -crossmat(offset)
                joint_rate_map = np.linalg.solve(np.vstack((Jv, Jw)), tip_twist)
                velocity[i] = joint_rate_map @ twist
                slider_jacobian[i] = joint_rate_map[slider_index]

            # Normalize linear velocity by the same characteristic length
            # used in the closure residual before computing the condition.
            slider_jacobian[:, :3] *= model.orientation_weight
            condition = float(np.linalg.cond(slider_jacobian))
            if not np.isfinite(condition):
                raise ValueError("Singular platform Jacobian")
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"Trajectory failed at t={t:.3f} s: {exc}") from exc

        samples.append(Sample(desired, actual, twist, position.ravel(), velocity.ravel(), condition))

    return joint_names, samples


class TrajectoryNode(Node):
    def __init__(self, name, future):
        super().__init__(name)
        self.future = future
        self.timer = None

        self.pubjoint = self.create_publisher(JointState, "/joint_states", 10)
        self.pubpose = self.create_publisher(PoseStamped, "/pose", 10)
        self.pubtwist = self.create_publisher(TwistStamped, "/twist", 10)
        self.pubcond = self.create_publisher(Float64, "/condition", 10)
        self.tfbroad = TransformBroadcaster(self)

    def start(self):
        self.model = StewartChains(self)
        self.dt = 0.05
        duration = 6.0
        times = np.linspace(0.0, duration, round(duration / self.dt) + 1)

        self.get_logger().info("Solving the trajectory before starting playback...")
        self.jointnames, self.samples = plan_trajectory(self.model, times)
        self.index = 0
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info(f"Publishing {len(self.samples)} samples at {1/self.dt:g} Hz")

    def update(self):
        sample = self.samples[self.index]
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id="base_link")

        self.pubjoint.publish(JointState(
            header=header,
            name=self.jointnames,
            position=sample.joint_position.tolist(),
            velocity=sample.joint_velocity.tolist(),
        ))
        self.pubpose.publish(PoseStamped(header=header, pose=Pose_from_T(sample.actual)))
        self.pubtwist.publish(TwistStamped(
            header=header,
            twist=Twist_from_vw(sample.twist[:3], sample.twist[3:]),
        ))
        self.tfbroad.sendTransform(TransformStamped(
            header=header,
            child_frame_id="desired",
            transform=Transform_from_T(sample.desired),
        ))
        self.pubcond.publish(Float64(data=sample.condition))

        self.index += 1
        if self.index == len(self.samples):
            self.timer.cancel()
            self.future.set_result("Trajectory has ended")


def main(args=None):
    rclpy.init(args=args)
    future = Future()
    trajectory = TrajectoryNode("trajectory", future)
    try:
        trajectory.start()
        rclpy.spin_until_future_complete(trajectory, future)
        if future.done():
            trajectory.get_logger().info(f"Stopping: {future.result()}")
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        trajectory.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
