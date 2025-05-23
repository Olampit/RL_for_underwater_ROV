import asyncio
from aiohttp import web

class ROVController:
    def __init__(self):
        self.state = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "thrust": 0.0,
            "yaw": 0.0,
            "depth": 0.0,
            "pressure_abs": 1013.25
        }

    async def update_state(self, request):
        data = await request.json()
        print(f"[RECEIVED COMMAND] {data}")
        self.state.update(data)
        print(f"[STATE] {self.state}")
        return web.json_response({"status": "ok", "received": data})

    async def get_state(self, request):
        return web.json_response(self.state)

    def routes(self):
        return [
            web.post('/update', self.update_state),
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

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        print("Shutting down server...")
    finally:
        await runner.cleanup()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server interrupted by user.")

