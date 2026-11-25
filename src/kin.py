"""Serial-chain and Stewart-platform kinematics for stewart.urdf.

StewartChains reads /robot_description by default. Pass urdf_xml to evaluate
the same model without a running ROS node or a physics simulator.
"""

import enum
import rclpy
import numpy as np

from rclpy.qos                  import QoSProfile, DurabilityPolicy
from std_msgs.msg               import String
from urdf_parser_py.urdf        import Robot
from scipy.optimize             import least_squares
from scipy.spatial.transform    import Rotation

from utils.TransformHelpers import (
    R_from_T, Reye, Rotn, T_from_Rp, T_from_URDF_origin, Teye,
    n_from_URDF_axis, p_from_T, pzero,
)


######################################################################
########################  HELPER FUNCTIONS  ##########################
######################################################################
#
#   Helper Functions: Report info and trigger an error
#
#   This uses the given Node's logger to post information (print) or
#   trigger an error.
#
def info(node, string):
    if node is not None:
        node.get_logger().info("KinematicChain: " + string)

def error(node, string):
    if node is not None:
        node.get_logger().error("KinematicChain: " + string)
    raise ValueError(string)


#
#   Helper Function: Read URDF XML from /robot_description
#
#   This uses the given Node to read the URDF XML description from
#   the standard /robot_description topic.
#
def read_HTML(node):
    # Report the action.
    info(node, "Waiting for the URDF to be published...")

    # Define the variable to receive the html and a callback function
    # to place the message content there.
    html = None
    def callback(msg):
        nonlocal html
        html = msg.data

    # Create a temporary subscriber to receive the URDF.  We use the
    # TRANSIENT_LOCAL durability, so that we see the last message
    # already published (if any).
    topic   = '/robot_description'
    quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
    sub     = node.create_subscription(String, topic, callback, quality)

    # Wait for the message.
    try:
        while html is None and rclpy.ok(context=node.context):
            rclpy.spin_once(node)
    finally:
        node.destroy_subscription(sub)
    if html is None:
        raise RuntimeError("ROS shut down before /robot_description arrived")
    return html


#
#   Helper Function: Parse URDF XML into the kinematic chain steps
#
#   Parse the given URDF XML to find the kinematic chain steps from
#   baseframe to tipframe.  It uses the given node to report problems
#   and errors.
#
def parse_HTML(node, html, baseframe, tipframe):
    return _parse_chain(node, Robot.from_xml_string(html), baseframe, tipframe)


def _parse_chain(node, robot, baseframe, tipframe):
    info(node, f"Processing URDF for robot '{robot.name}'")
    info(node, f"Building chain from '{baseframe}' to '{tipframe}'")

    # Prepare the empty chain (list of kinematic steps).
    chain = []

    # Parse the Robot object into a list of URDF steps from the base
    # frame to the tip frame.  Search backwards, from the tip to the
    # base, as the robot could be a tree structure: while a parent may
    # have multiple children, every child has only one parent.  The
    # resulting chain of steps is unique.
    frame = tipframe
    while (frame != baseframe):
        # Look for the URDF joint to the parent frame.
        joints = [j for j in robot.joints if j.child == frame]
        if len(joints) == 0:
            error(node, f"Unable find joint connecting to '{frame}'")
        elif len(joints) != 1:
            error(node, f"Unable find unique joint connecting to '{frame}'")
        joint = joints[0]

        # Find the parent frame.
        if (joint.parent == frame):
            error(node, f"Joint '{joint.name}' connects '{frame}' to itself")
        frame = joint.parent

        # Check the type (use the above enumeration)
        if  (joint.type == 'revolute' or
             joint.type == 'continuous'):  type = JointType.REVOLUTE
        elif joint.type == 'prismatic':    type = JointType.LINEAR
        elif joint.type == 'fixed':        type = JointType.FIXED
        else:
            error(node, f"Joint '{joint.name}' has unknown type '{joint.type}'")

        # Check that the axis is normalized (of axis is expected).
        if type is JointType.FIXED:
            nlocal = None
        else:
            nlocal = n_from_URDF_axis(joint.axis)
            mag = np.linalg.norm(nlocal)
            if not np.isfinite(mag) or mag == 0:
                error(node, f"Invalid axis for {joint.name}")
            if abs(mag - 1) > 1e-6:
                info(node, f"WARNING Joint '{joint.name}' axis normalized")
            nlocal = nlocal / mag

        # Grab the shift (or identity if none specified).
        if joint.origin is None:
            info(node, f"WARNING Joint '{joint.name}' has no <origin>")
            Tshift = Teye()
        else:
            Tshift = T_from_URDF_origin(joint.origin)

        # Convert the collected information into a single URDF step.
        # Note the axis (nlocal) is meaningless for a fixed joint.
        lower, upper = -np.inf, np.inf
        if joint.type in ("prismatic", "revolute") and joint.limit is not None:
            lower, upper = joint.limit.lower, joint.limit.upper
            # The export rounds full turns to 6.283185 rad. Use unwrapped
            # coordinates for these periodic joints to let them cross zero.
            if joint.type == "revolute" and upper - lower >= 2 * np.pi - 1e-6:
                lower, upper = -np.inf, np.inf
        chain.insert(0, URDFStep(
            name=joint.name, type=type, Tshift=Tshift, nlocal=nlocal,
            lower=lower, upper=upper))

    # Return the chain.
    return chain


######################################################################
############################  MAIN CODE  #############################
######################################################################
#
#   Single URDF Step
#
#   This captures a single step from one frame to the next.  It can be of type:
#
#     FIXED     Just a fixed T-matrix shift, nothing moving, not an active DOF.
#     REVOLUTE  A fixed T-matrix shift, followed by a rotation about an axis.
#     LINEAR    A fixed T-matrix shift, followed by a transation along an axis.
#
#   It contains several pieces of permanent data (coming from the URDF):
#
#     name      String showing the name
#     type      One of the above
#     Tshift    Fixed shift: Transform of this frame w.r.t. previous frame
#     nlocal    Joint axis (if applicable) in this local frame
#
#   We also add information how this relates to the active joints:
#
#     dof       If an active dof (not FIXED), the dof number
#
#   And this provides repr() and str() methods to display this information.
#

# Define the joint types.
class JointType(enum.Enum):
    FIXED    = 0
    REVOLUTE = 1
    LINEAR   = 2

# Define a single step in the URDF (kinematic chain).
class URDFStep():
    def __init__(self, name, type, Tshift, nlocal, lower=-np.inf, upper=np.inf):
        # Store the permanent/fixed/URDF data.
        self.name   = name      # Joint name
        self.type   = type      # Joint type (per above enumeration)
        self.Tshift = Tshift    # Fixed transform w.r.t. previous frame
        self.nlocal = nlocal    # Joint axis in this local frame

        self.lower = lower
        self.upper = upper

        # Match against the joint numbers
        self.dof    = None      # Joint DOF number (or None if FIXED)

    # Return a compact representation
    def __repr__(self):
        return ((f"{self.type.name:8} ") +
                (" "*6 if self.dof is None else f"DOF #{self.dof}") +
                (f" '{self.name}'"))

    # Return a complete multi-line description
    def __str__(self, prefix="step."):
        space = " " * (len(prefix) + 8)
        return (
            "-" * (len(prefix) + 31) + "\n" +
            f"{prefix}name    '{self.name}'\n" +
            f"{prefix}type    {self.type.name}\n" +
            f"{prefix}Tshift  {str(self.Tshift).replace("\n", "\n"+space)}\n" +
            f"{prefix}nlocal  {str(self.nlocal)}\n" +
            f"{prefix}dof     {str(self.dof)}\n")


#
#   Kinematic Chain Object
#
#   This stores the information provided by the URDF in the form of
#   a serial kinematic chain (list of steps). 
#   In particular, see the fkin()
#   function, as it walks up the chain to determine the transforms.
#

# Define a full kinematic chain
class KinematicChain():
    # Initialization - load the URDF and set up the chain.
    def __init__(self, node, baseframe, tipframe, expectedjointnames, *, robot=None):
        # Store the node (so we can properly report errors later).
        self.node = node

        if robot is None:
            robot = Robot.from_xml_string(read_HTML(node))
        self.chain = _parse_chain(node, robot, baseframe, tipframe)
        self.steps = len(self.chain)

        # Set/count the active DOF numbers, walking up the chain.
        dof = 0
        for step in self.chain:
            if step.type == JointType.FIXED:
                step.dof = None         # Skip fixed steps.
            else:
                step.dof = dof          # Mark the active dof number
                dof += 1                # Increment the dof number
        self.dofs = dof
        info(node, "URDF has %d steps, %d active DOFs:" %
             (self.steps, self.dofs))

        # Report the steps we found.
        for (i, step) in enumerate(self.chain):
            info(node, "Step #%d %s" % (i, repr(step)))

        # Confirm the active joint names matches the expectation.
        jointnames = [s.name for s in self.chain if s.type != JointType.FIXED]
        if expectedjointnames is not None and jointnames != list(expectedjointnames):
            error(node, "Chain does not match the expected names: " +
                  str(expectedjointnames))


    # Report the details of the chain (of each step).
    def report(self):
        info(self.node, ("The chain steps are:\n" +
                         "".join(str(step) for step in self.chain)))



    # Compute the forward kinematics!
    def fkin(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dofs,):
            error(self.node, f"Expected {self.dofs} joint coordinates in chain order")

        ### INITIALIZE ###
        # We will build up three lists.  For each DOF (non-fixed, active
        # step) collect the type, position (pi), axis (ni) w.r.t. the base.
        type = []
        p    = []
        n    = []

        # Initialize the T matrix to walk up the chain, w.r.t. the base frame!
        T = Teye()

        ### PHASE 1: WALK UP THE CHAIN ###
        # We walk the chain, one URDF step at a time, adjusting T as we
        # go.  Each step could be a fixed or active URDF joint.
        for step in self.chain:

            """
              The T varable is the T matrix of a step w.r.t. world.
              As the loop iterates over steps, adjust the T matrix to
              incorporate the step (move up the chain).

              Note the step is of type URDFStep (see above) and contains:
                step.Tshift     Fixed transform w.r.t. the previous frame
                step.nlocal     Joint axis in the local (post-shift) frame
                step.dof        Joint number
              which also tell us
                q[step.dof]     Joint position (angle or displacement)
            """

            # Take action based on the joint type: Move the transform T
            # up the kinematic chain (remaining w.r.t. the base frame).
            if step.type is JointType.REVOLUTE:
                # Revolute is a shift followed by a rotation:

                T = T @ step.Tshift @ T_from_Rp(Rotn(step.nlocal, q[step.dof]), pzero())

            elif step.type is JointType.LINEAR:
                # Linear is a shift followed by a translation:

                T = T @ step.Tshift @ T_from_Rp(Reye(), q[step.dof] * step.nlocal)

            else:
                # Fixed is only a shift.

                T = T @ step.Tshift


            # For active joints (our DOFs), store the type, positon (pi),
            # and axis (ni) info, w.r.t. the base frame.
            if step.type != JointType.FIXED:
                type.append(step.type)
                p.append(p_from_T(T))
                n.append(R_from_T(T) @ step.nlocal)

        # Collect the tip information.
        ptip = p_from_T(T)
        Rtip = R_from_T(T)

        ### PHASE 2: USE ABOVE INFOMATION TO BUILD THE JACOBIAN ###
        # Collect the Jacobian for each active joint.
        Jv = np.zeros((3,self.dofs))
        Jw = np.zeros((3,self.dofs))
        for i in range(self.dofs):

            """
              Calculate the Jacobian.

              This loops over the dofs, effectively ignoring the fixed
              URDF joints.  They do not enter into the Jacobian.

              After having walked up the chain above, we now have:
              type[i]   Joint type
              p[i]      Joint position w.r.t. world
              n[i]      Joint axis w.r.t. world
            """

            # Fill in the appropriate Jacobian column based on the
            # type.  The Jacobian (like the data) is w.r.t. the base.
            if type[i] is JointType.REVOLUTE:
                # Revolute is a rotation:

                Jv[:,i] = np.cross(n[i], ptip - p[i])
                Jw[:,i] = n[i]

            elif type[i] is JointType.LINEAR:
                # Linear is a translation:

                Jv[:,i] = n[i]
                Jw[:,i] = np.zeros(3)


        # Return the info
        return (ptip, Rtip, Jv, Jw)



class StewartChains:
    """Close the five open loops of the Stewart platform URDF.

    Leg arrays are ordered by Slider_13 through Slider_18. Each row of
    passive angles follows the base-to-tip URDF order with the slider removed.
    Full-turn revolute coordinates are returned unwrapped in radians.
    """

    leg_tips = ("J1B_1", "J2B1", "J3B1", "J4B1", "J5B1", "J6B1")
    # Leg 6 already connects to TOP1. The sixth outgoing top chain is
    # indicator1. J6T1 is upstream of TOP1, so it is absent from this list.
    top_tips = ("J1T1", "J2T1", "J3T_1", "J4T1", "J5T_1", "indicator1")
    orientation_weight = 0.2  # meters per radian
    position_tolerance = 1e-7  # meters
    angle_tolerance = 1e-6  # radians

    def __init__(
        self, node=None, baseframe="base_link", legtipframes=leg_tips,
        topframe="TOP1", toptipframes=top_tips, explegjointnames=None,
        exptopjointnames=None, *, urdf_xml=None,
    ):
        if urdf_xml is None:
            if node is None:
                raise ValueError("Provide a ROS node or urdf_xml")
            urdf_xml = read_HTML(node)
        robot = Robot.from_xml_string(urdf_xml)

        leg_names = [None] * 6 if explegjointnames is None else explegjointnames
        top_names = [None] * 6 if exptopjointnames is None else exptopjointnames
        if any(len(items) != 6 for items in (legtipframes, toptipframes, leg_names, top_names)):
            raise ValueError("Expected six leg and six top chains")

        self.node = node
        self.legchains = [
            KinematicChain(node, baseframe, tip, names, robot=robot)
            for tip, names in zip(legtipframes, leg_names)
        ]
        self.topchains = [
            KinematicChain(node, topframe, tip, names, robot=robot)
            for tip, names in zip(toptipframes, top_names)
        ]
        if any(chain.dofs for chain in self.topchains):
            raise ValueError("Expected rigid top chains")

        self.basetopchain = KinematicChain(node, baseframe, topframe, leg_names[5], robot=robot)
        if self.basetopchain.dofs != 6:
            raise ValueError("Expected TOP1 to be rigidly attached to the sixth leg")

        # In the current URDF each slider is the third moving joint. Read
        # its index from the chain so the packing follows the URDF order.
        self._slider_indices = []
        self._joint_lower = np.empty((6, 6))
        self._joint_upper = np.empty((6, 6))
        for i, chain in enumerate(self.legchains):
            joints = [step for step in chain.chain if step.dof is not None]
            sliders = [step for step in joints if step.type == JointType.LINEAR]
            if len(joints) != 6 or len(sliders) != 1 or sliders[0].name != f"Slider_{13 + i}":
                raise ValueError(f"Expected Slider_{13 + i} and five revolute joints on leg {i + 1}")
            self._slider_indices.append(sliders[0].dof)
            self._joint_lower[i] = [step.lower for step in joints]
            self._joint_upper[i] = [step.upper for step in joints]

        leg6_prefix = self.basetopchain.chain[:self.legchains[5].steps]
        if [step.name for step in leg6_prefix] != [step.name for step in self.legchains[5].chain]:
            raise ValueError("Expected the sixth leg to lie on the base-to-TOP1 chain")

        self._passive_joints = np.ones((6, 6), dtype=bool)
        self._passive_joints[np.arange(6), self._slider_indices] = False
        self._theta_lower = self._joint_lower[self._passive_joints].reshape(6, 5)
        self._theta_upper = self._joint_upper[self._passive_joints].reshape(6, 5)
        self.slider_lower = self._joint_lower[~self._passive_joints]
        self.slider_upper = self._joint_upper[~self._passive_joints]

        # The zero pose is the reference assembly. TOP1 is approximately
        # (0.040872, -0.194619, 0.360449) m in base_link at zero.
        s0, theta0 = np.zeros(6), np.zeros((6, 5))
        self.home_pose = self.T_base_top(s0, theta0)

        # A_i is TOP1 -> J*B*, the leg-side frame. The J*T* origins sit
        # about 2 mm away on legs 1–5 (1 mm on leg 6), so matching the
        # frames directly would shift the assembly relative to the CAD export.
        self.As = [
            np.linalg.solve(self.home_pose, self.leg_fkin(i, s0, theta0)[0])
            for i in range(6)
        ]

    def _leg_coordinates(self, i, slider, angles):
        q = np.empty(6)
        q[self._slider_indices[i]] = slider
        q[self._passive_joints[i]] = angles
        return q

    def leg_fkin(self, i, s, theta):
        """Return the leg-side transform and serial Jacobians for leg i."""
        q = self._leg_coordinates(i, s[i], theta[i])
        p, R, Jv, Jw = self.legchains[i].fkin(q)
        return T_from_Rp(R, p), Jv, Jw

    def T_base_top(self, s, theta):
        """Return the platform transform along the connected sixth leg."""
        q = self._leg_coordinates(5, s[5], theta[5])
        p, R, _, _ = self.basetopchain.fkin(q)
        return T_from_Rp(R, p)

    def _pose_error(self, actual, target):
        position_error = actual[:3, 3] - target[:3, 3]
        relative_rotation = target[:3, :3].T @ actual[:3, :3]
        rotation_error = Rotation.from_matrix(relative_rotation).as_rotvec()
        return np.r_[position_error, self.orientation_weight * rotation_error]

    def residual(self, s, theta):
        """Return 30 closure residuals for the five disconnected legs."""
        # Leg 6 defines T through its fixed connection, closing its own loop.
        T = self.T_base_top(s, theta)
        return np.concatenate([
            self._pose_error(self.leg_fkin(i, s, theta)[0], T @ self.As[i])
            for i in range(5)
        ])

    @staticmethod
    def _validate_array(value, shape, name):
        array = np.asarray(value, dtype=float)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite array of shape {shape}")
        return array

    def _solve(self, residual, initial, lower, upper, description):
        def errors(values):
            blocks = residual(values).reshape(-1, 6)
            position = np.max(np.linalg.norm(blocks[:, :3], axis=1))
            angle = np.max(np.linalg.norm(blocks[:, 3:], axis=1)) / self.orientation_weight
            return position, angle

        initial = np.clip(initial, lower, upper)
        position, angle = errors(initial)
        if position <= self.position_tolerance and angle <= self.angle_tolerance:
            return initial

        result = least_squares(
            residual, initial, bounds=(lower, upper),
            ftol=1e-12, xtol=1e-12, gtol=None,
        )
        position, angle = errors(result.x)
        if position > self.position_tolerance or angle > self.angle_tolerance:
            raise ValueError(
                f"{description} did not close (position error {position:.3g} m, "
                f"angle error {angle:.3g} rad): {result.message}"
            )
        return result.x

    def fkin(self, s, theta=None):
        """Return (base-to-TOP1 transform, passive angles) for six slider readings.

        s is in meters from URDF zero. Optionally pass the previous (6, 5)
        passive angles in radians to follow the same assembly mode.
        """
        s = self._validate_array(s, (6,), "s")
        if np.any(s < self.slider_lower) or np.any(s > self.slider_upper):
            raise ValueError("Slider positions are outside their URDF limits")
        theta = np.zeros((6, 5)) if theta is None else self._validate_array(theta, (6, 5), "theta")

        angles = self._solve(
            lambda x: self.residual(s, x.reshape(6, 5)),
            theta.ravel(), self._theta_lower.ravel(), self._theta_upper.ravel(),
            "Forward kinematics",
        ).reshape(6, 5)
        return self.T_base_top(s, angles), angles

    def ikin(self, T, s=None, theta=None):
        """Return (slider positions, passive angles) for a base-to-TOP1 pose.

        The target is a 4x4 rigid transform. Optional s and theta seed the
        six independent leg solves to track the previous configuration.
        """
        T = self._validate_array(T, (4, 4), "T")
        R = T[:3, :3]
        if (not np.allclose(T[3], [0, 0, 0, 1])
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(R), 1, atol=1e-7)):
            raise ValueError("T must be a homogeneous rigid transform")

        s = np.zeros(6) if s is None else self._validate_array(s, (6,), "s")
        theta = np.zeros((6, 5)) if theta is None else self._validate_array(theta, (6, 5), "theta")
        sliders, angles = np.empty(6), np.empty((6, 5))
        for i, chain in enumerate(self.legchains):
            target = T @ self.As[i]

            def residual(q):
                p, R, _, _ = chain.fkin(q)
                return self._pose_error(T_from_Rp(R, p), target)

            q = self._solve(
                residual, self._leg_coordinates(i, s[i], theta[i]),
                self._joint_lower[i], self._joint_upper[i], f"Inverse kinematics for leg {i + 1}",
            )
            sliders[i] = q[self._slider_indices[i]]
            angles[i] = q[self._passive_joints[i]]

        return sliders, angles
