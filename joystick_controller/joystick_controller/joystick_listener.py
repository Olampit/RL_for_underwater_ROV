import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import aiohttp
import asyncio

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
        desired_speed = axes[1]  # Example: forward axis
        asyncio.create_task(self.post_desired_speed(desired_speed))

    async def post_desired_speed(self, desired_speed):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8080/joystick",
                    json={"desired_speed": desired_speed}
                ) as response:
                    if response.status == 200:
                        self.get_logger().info("Posted desired speed successfully.")
                    else:
                        self.get_logger().warn(f"Failed to post: {response.status}")
        except Exception as e:
            self.get_logger().error(f"HTTP post error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = JoystickReader()

    # Spin with asyncio integration
    try:
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, executor.spin)
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
