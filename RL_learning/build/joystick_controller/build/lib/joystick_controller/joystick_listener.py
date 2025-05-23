# joystick_listener.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

class JoystickReader(Node):
    def __init__(self):
        super().__init__('joystick_reader')
        self.subscription = self.create_subscription(
            Joy,
            'joy',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        axes = msg.axes
        buttons = msg.buttons
        self.get_logger().info(f'Axes: {axes} | Buttons: {buttons}')

def main(args=None):
    rclpy.init(args=args)
    node = JoystickReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


