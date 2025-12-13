"""
WebSocket server to connect the frontend UI to the Zentrax voice/gesture control backend.
Run this alongside main.py to enable web UI control.
"""

import asyncio
import websockets
import json
import threading
import sys
import os

# Add project root to path to import main
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

try:
    from main import VoiceGestureControl
except ImportError as e:
    print(f"Warning: Could not import VoiceGestureControl from main.py: {e}")
    print("WebSocket server will run in standalone mode")
    VoiceGestureControl = None


class ZentraxWebSocketServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.controller = None
        self.controller_thread = None
        
    async def register(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial status
        await self.send_to_client(websocket, {
            'type': 'status',
            'status': 'sleeping' if not (self.controller and self.controller.is_awake) else 'awake',
            'mode': self.controller.active_mode if self.controller else None
        })
        
    async def unregister(self, websocket):
        """Unregister a disconnected client"""
        self.clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def send_to_client(self, websocket, message):
        """Send message to a specific client"""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            print(f"Error sending to client: {e}")
            
    async def broadcast(self, message):
        """Broadcast message to all connected clients"""
        if self.clients:
            await asyncio.gather(
                *[self.send_to_client(client, message) for client in self.clients],
                return_exceptions=True
            )
            
    async def handle_client(self, websocket):
        """Handle incoming client connections and messages"""
        await self.register(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_command(data)
                except json.JSONDecodeError:
                    await self.send_to_client(websocket, {
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    })
                except Exception as e:
                    await self.send_to_client(websocket, {
                        'type': 'error',
                        'message': str(e)
                    })
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
            
    async def handle_command(self, data):
        """Process commands from frontend"""
        command = data.get('command')
        params = data.get('params', {})
        
        print(f"Received command: {command}, params: {params}")
        
        if command == 'wake':
            if not self.controller or not self.controller.running:
                # Start the controller
                self.start_controller()
                await self.broadcast({
                    'type': 'log',
                    'message': 'Starting Zentrax...',
                    'level': 'info'
                })
            
            if self.controller:
                self.controller.is_awake = True
                self.controller.active_mode = 'voice'
                await self.broadcast({
                    'type': 'status',
                    'status': 'awake',
                    'mode': 'voice'
                })
                await self.broadcast({
                    'type': 'log',
                    'message': 'Zentrax is now awake in voice mode',
                    'level': 'success'
                })
                
        elif command == 'sleep':
            if self.controller:
                self.controller.is_awake = False
                await self.broadcast({
                    'type': 'status',
                    'status': 'sleeping',
                    'mode': None
                })
                await self.broadcast({
                    'type': 'log',
                    'message': 'Zentrax is going to sleep',
                    'level': 'info'
                })
                
        elif command == 'switch_mode':
            mode = params.get('mode')
            if self.controller and self.controller.is_awake:
                self.controller.active_mode = mode
                await self.broadcast({
                    'type': 'status',
                    'status': 'awake',
                    'mode': mode
                })
                await self.broadcast({
                    'type': 'log',
                    'message': f'Switched to {mode} mode',
                    'level': 'success'
                })
            else:
                await self.broadcast({
                    'type': 'log',
                    'message': 'Zentrax must be awake to switch modes',
                    'level': 'warning'
                })
                
        elif command == 'start_game':
            if self.controller and self.controller.is_awake:
                try:
                    self.controller.start_hill_climb()
                    await self.broadcast({
                        'type': 'log',
                        'message': 'Hill Climb game started',
                        'level': 'success'
                    })
                except Exception as e:
                    await self.broadcast({
                        'type': 'log',
                        'message': f'Failed to start game: {str(e)}',
                        'level': 'error'
                    })
            else:
                await self.broadcast({
                    'type': 'log',
                    'message': 'Zentrax must be awake to start the game',
                    'level': 'warning'
                })
                
        elif command == 'stop':
            if self.controller:
                self.controller.running = False
                await self.broadcast({
                    'type': 'log',
                    'message': 'Zentrax stopped',
                    'level': 'info'
                })
                
    def start_controller(self):
        """Start the VoiceGestureControl in a separate thread"""
        if not self.controller or not self.controller.running:
            print("Starting VoiceGestureControl...")
            self.controller = VoiceGestureControl(use_whisper=True, whisper_model="base")
            self.controller_thread = threading.Thread(
                target=self.controller.run,
                daemon=True
            )
            self.controller_thread.start()
            print("VoiceGestureControl started")
            
    async def start(self):
        """Start the WebSocket server"""
        print(f"Starting Zentrax WebSocket Server on {self.host}:{self.port}")
        print("Open frontend/index.html in your browser to access the UI")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever


def main():
    """Main entry point"""
    server = ZentraxWebSocketServer(host='localhost', port=8765)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nShutting down server...")
        if server.controller:
            server.controller.running = False
        print("Server stopped")


if __name__ == "__main__":
    main()
