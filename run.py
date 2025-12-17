"""
Zentrax Unified Launcher
========================
Single command to start the entire Zentrax AI Assistant:
- Voice & Gesture Control Backend
- WebSocket Server for UI communication
- Frontend HTTP Server
- Opens the UI in default browser

Usage: python run.py [options]
  --headless     Run without camera window
  --no-browser   Don't auto-open browser
  --port PORT    Frontend port (default: 8080)
  --ws-port PORT WebSocket port (default: 8765)
"""

import subprocess
import threading
import webbrowser
import time
import sys
import os
import signal
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    """Print Zentrax ASCII banner"""
    banner = f"""{Colors.CYAN}
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗███████╗███╗   ██╗████████╗██████╗  █████╗ ██╗  ██╗ ║
║   ╚══███╔╝██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗╚██╗██╔╝ ║
║     ███╔╝ █████╗  ██╔██╗ ██║   ██║   ██████╔╝███████║ ╚███╔╝  ║
║    ███╔╝  ██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗██╔══██║ ██╔██╗  ║
║   ███████╗███████╗██║ ╚████║   ██║   ██║  ██║██║  ██║██╔╝ ██╗ ║
║   ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ║
║                                                               ║
║              🎤 AI Voice & Gesture Assistant 🖐️               ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)

def log(message, level="INFO"):
    """Pretty print log messages"""
    colors = {
        "INFO": Colors.CYAN,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "HEADER": Colors.HEADER
    }
    color = colors.get(level, Colors.CYAN)
    timestamp = time.strftime("%H:%M:%S")
    print(f"{Colors.BOLD}[{timestamp}]{Colors.ENDC} {color}[{level}]{Colors.ENDC} {message}")

class FrontendServer(threading.Thread):
    """HTTP Server for Frontend UI"""
    
    def __init__(self, port=8080):
        super().__init__(daemon=True)
        self.port = port
        self.server = None
        self.running = False
        
    def run(self):
        """Start the HTTP server"""
        frontend_dir = os.path.join(PROJECT_ROOT, 'frontend')
        
        if not os.path.exists(frontend_dir):
            log(f"Frontend directory not found: {frontend_dir}", "ERROR")
            return
            
        os.chdir(frontend_dir)
        
        handler = partial(SimpleHTTPRequestHandler, directory=frontend_dir)
        
        try:
            self.server = HTTPServer(('localhost', self.port), handler)
            self.running = True
            log(f"Frontend server started at http://localhost:{self.port}", "SUCCESS")
            self.server.serve_forever()
        except OSError as e:
            if "address already in use" in str(e).lower() or e.errno == 10048:
                log(f"Port {self.port} already in use. Frontend may already be running.", "WARNING")
            else:
                log(f"Failed to start frontend server: {e}", "ERROR")
                
    def stop(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            self.running = False
            log("Frontend server stopped", "INFO")


class ZentraxLauncher:
    """Main launcher for all Zentrax components"""
    
    def __init__(self, args):
        self.args = args
        self.frontend_server = None
        self.main_process = None
        self.running = False
        
    def start_frontend(self):
        """Start the frontend HTTP server"""
        log("Starting frontend server...", "INFO")
        self.frontend_server = FrontendServer(port=self.args.port)
        self.frontend_server.start()
        time.sleep(1)  # Give server time to start
        
    def start_backend(self):
        """Start the main voice/gesture control backend"""
        log("Starting Zentrax backend...", "INFO")
        
        main_script = os.path.join(PROJECT_ROOT, 'main.py')
        
        if not os.path.exists(main_script):
            log(f"main.py not found at {main_script}", "ERROR")
            return False
            
        cmd = [sys.executable, main_script]
        
        if self.args.headless:
            cmd.append('--headless')
            
        try:
            # Run main.py in the current terminal (not as subprocess)
            # This allows proper keyboard interrupt handling
            os.chdir(PROJECT_ROOT)
            
            # Import and run directly for better integration
            log("Initializing voice & gesture control...", "INFO")
            
            # Add the headless flag to sys.argv if needed
            original_argv = sys.argv.copy()
            sys.argv = ['main.py']
            if self.args.headless:
                sys.argv.append('--headless')
            
            # Import main module
            import importlib.util
            spec = importlib.util.spec_from_file_location("main_module", main_script)
            main_module = importlib.util.module_from_spec(spec)
            
            # Restore argv
            sys.argv = original_argv
            
            # Load and run
            spec.loader.exec_module(main_module)
            
            return True
            
        except KeyboardInterrupt:
            log("Received shutdown signal", "WARNING")
            return False
        except Exception as e:
            log(f"Failed to start backend: {e}", "ERROR")
            return False
            
    def open_browser(self):
        """Open the UI in default browser"""
        if not self.args.no_browser:
            url = f"http://localhost:{self.args.port}"
            log(f"Opening browser at {url}", "INFO")
            time.sleep(2)  # Wait for servers to be ready
            webbrowser.open(url)
            
    def run(self):
        """Run all components"""
        self.running = True
        
        print_banner()
        log("Starting Zentrax AI Assistant...", "HEADER")
        print()
        
        # Start frontend server
        self.start_frontend()
        
        # Open browser
        if not self.args.no_browser:
            threading.Thread(target=self.open_browser, daemon=True).start()
        
        # Print status
        print()
        log("=" * 50, "INFO")
        log(f"🌐 Frontend UI: http://localhost:{self.args.port}", "SUCCESS")
        log(f"🔌 WebSocket:   ws://localhost:{self.args.ws_port}", "SUCCESS")
        log("=" * 50, "INFO")
        print()
        log("Say 'Hey Zentrax' to wake up the assistant!", "INFO")
        log("Press Ctrl+C to stop all services", "INFO")
        print()
        
        # Start backend (this blocks until exit)
        self.start_backend()
        
    def stop(self):
        """Stop all components"""
        log("Shutting down Zentrax...", "WARNING")
        self.running = False
        
        if self.frontend_server:
            self.frontend_server.stop()
            
        if self.main_process:
            self.main_process.terminate()
            
        log("Zentrax stopped. Goodbye! 👋", "INFO")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Zentrax AI Assistant - Voice & Gesture Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    Start with UI and browser
  python run.py --headless         Run without camera window
  python run.py --no-browser       Don't auto-open browser
  python run.py --port 3000        Use custom frontend port
        """
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without displaying camera window'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not automatically open browser'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Frontend HTTP server port (default: 8080)'
    )
    
    parser.add_argument(
        '--ws-port',
        type=int,
        default=8765,
        help='WebSocket server port (default: 8765)'
    )
    
    args = parser.parse_args()
    
    launcher = ZentraxLauncher(args)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print()
        launcher.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        launcher.run()
    except KeyboardInterrupt:
        launcher.stop()
    except Exception as e:
        log(f"Fatal error: {e}", "ERROR")
        launcher.stop()
        sys.exit(1)


if __name__ == '__main__':
    main()
