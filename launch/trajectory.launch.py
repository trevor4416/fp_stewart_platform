"""Launch test trajectory for Stewart platform.

   ros2 launch fp_stewart_platform kin.launch.py

This should start
  1) RVIZ, ready to view the robot
  2) The robot_state_publisher to broadcast the robot model
  3) The trajectory code
  
"""

import os

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir('utils'), 'rviz', 'viewurdfplus.rviz')

    # Locate the URDF file.
    urdf = os.path.join(pkgdir('fp_stewart_platform'), 'urdf', 'stewart.urdf')

    # Load the robot's URDF file (XML).
    with open(urdf, 'r') as file:
        robot_description = file.read().replace(
            'package://stewart/meshes/', 'package://fp_stewart_platform/meshes/'
        )


    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for the robot_state_publisher.
    node_robot_state_publisher = Node(
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}])

    # Configure a node for RVIZ
    node_rviz = Node(
        name       = 'rviz', 
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg, '-f', 'base_link'],
        on_exit    = Shutdown())
    
    # Configure a node for the joint trajectory
    node_trajectory = Node(
        name       = 'trajectory', 
        package    = 'fp_stewart_platform',
        executable = 'trajectory',
        output     = 'screen')


    ######################################################################
    # RETURN THE ELEMENTS IN ONE LIST
    return LaunchDescription([
        # Start the robot_state_publisher, RVIZ, and the trajectory.
        node_robot_state_publisher,
        node_rviz,
        node_trajectory,
    ])
