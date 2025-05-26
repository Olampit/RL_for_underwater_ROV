import asyncio
from aiohttp import web
import signal

class ROVController:
    def __init__(self):
        self.state = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "thrust": 0.0,
            "yaw": 0.0,
            "depth": 0.0,
            "pressure_abs": 1013.25,
            "desired_speed": 0.0  # ← added to track joystick input
        }

    async def update_state(self, request):
        data = await request.json()
        print(f"[RECEIVED COMMAND] /update {data}")
        self.state.update(data)
        print(f"[STATE] {self.state}")
        return web.json_response({"status": "ok", "received": data})

    async def joystick_input(self, request):
        data = await request.json()
        print(f"[RECEIVED JOYSTICK] /joystick {data}")
        if "desired_speed" in data:
            self.state["desired_speed"] = data["desired_speed"]
        print(f"[STATE] {self.state}")
        return web.json_response({"status": "ok", "received": data})

    async def get_state(self, request):
        return web.json_response(self.state)

    def routes(self):
        return [
            web.post('/update', self.update_state),
            web.post('/joystick', self.joystick_input),  # ← new route
            web.get('/state', self.get_state)
        ]

async def main():
    controller = ROVController()
    app = web.Application()
    app.add_routes(controller.routes())
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()

    print("ROV Controller API running on http://localhost:8080")

    stop_event = asyncio.Event()

    def handle_sigint():
        print("Received SIGINT, shutting down...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, handle_sigint)
    loop.add_signal_handler(signal.SIGTERM, handle_sigint)

    await stop_event.wait()
    await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
