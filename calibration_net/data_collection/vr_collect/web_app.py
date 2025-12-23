import flask
from flask import Flask, render_template, jsonify, request
import subprocess
import sys
import os
import zmq
import time
import socket

app = Flask(__name__)

# Configuration
BASE_VISER_PORT = 8080
BASE_ZMQ_PORT = 6000 # Starting port for control sockets
BASE_VUER_PORT = 5555 # Starting port for Vuer data streams

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

def free_port(port):
    """Kills any process listening on the given port."""
    try:
        cmd = f"lsof -i :{port} -t"
        try:
            pid_bytes = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
            pids = pid_bytes.decode().strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"Port {port} is in use by PID {pid}. Killing it...")
                    subprocess.call(f"kill -9 {pid}", shell=True)
            if pids:
                time.sleep(0.5) # Wait for release
        except subprocess.CalledProcessError:
            pass # No process found
    except Exception as e:
        print(f"Warning: Failed to clean up port {port}: {e}")

def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

class DeviceManager:
    def __init__(self):
        self.robot_process = None
        self.robot_config = None
        self.robot_zmq_port = None
        self.robot_viser_port = None
        self.robot_socket = None
        
        self.input_devices = {}
        self.next_id = 0
        self.context = zmq.Context()
        self.is_calibrated = False

    def start_robot(self, config):
        if self.robot_process:
            self.stop_robot()
            
        self.robot_config = config
        self.robot_viser_port = int(config.get('viser_port', BASE_VISER_PORT))
        self.robot_zmq_port = BASE_ZMQ_PORT
        free_port(self.robot_viser_port)
        free_port(self.robot_zmq_port)
        cmd = [
            sys.executable, 
            os.path.join(os.path.dirname(__file__), 'robot_node.py'),
            '--hand-type', config.get('hand_type', 'inspire'),
            '--viser-port', str(self.robot_viser_port),
            '--zmq-port', str(self.robot_zmq_port)
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = WORKSPACE_ROOT + os.pathsep + env.get('PYTHONPATH', '')
        
        print(f"Starting Robot Node with command: {' '.join(cmd)}")
        self.robot_process = subprocess.Popen(cmd, env=env)
        
        # Connect ZMQ
        self.robot_socket = self.context.socket(zmq.REQ)
        self.robot_socket.connect(f"tcp://localhost:{self.robot_zmq_port}")
        
        return True

    def stop_robot(self):
        if self.robot_process:
            try:
                self.robot_socket.send_json({'cmd': 'quit'})
            except:
                pass
            self.robot_process.terminate()
            self.robot_process = None
            self.robot_socket = None
            
    def create_socket(self, port, mode=zmq.REQ):
        socket = self.context.socket(mode)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect(f"tcp://localhost:{port}")
        return socket

    def add_input_device(self, config):
        self.is_calibrated = False
        device_id = str(self.next_id)
        self.next_id += 1
        
        # Assign ports
        # Control port: BASE + 1 + id
        zmq_port = BASE_ZMQ_PORT + 1 + int(device_id)
        # Vuer port (if needed): BASE_VUER + id
        vuer_zmq_port = BASE_VUER_PORT + int(device_id)
            
        # Viser port for input device
        viser_port = BASE_VISER_PORT + 100 + int(device_id)
        
        free_port(viser_port)
        free_port(zmq_port)
        free_port(vuer_zmq_port)
        
        cmd = [
            sys.executable, 
            os.path.join(os.path.dirname(__file__), 'input_node.py'),
            '--output-dir', config.get('output_dir', 'data'),
            '--hand-type', self.robot_config.get('hand_type', 'inspire') if self.robot_config else 'inspire',
            '--collector-type', config.get('collector_type', 'avp'),
            '--zmq-port', str(zmq_port),
            '--device-id', device_id,
            '--vuer-zmq-port', str(vuer_zmq_port),
            '--viser-port', str(viser_port),
            '--sides', config.get('side', 'both')
        ]
        
        if config.get('vuer_port') and config['collector_type'] == 'vuer':
            free_port(config['vuer_port'])
            cmd.extend(['--vuer-port', str(config['vuer_port'])])
        if config.get('ip'):
            cmd.extend(['--ip', config['ip']])
        if config.get('camera_index'):
            cmd.extend(['--camera-index', str(config['camera_index'])])
        if config.get('left_glove_port'):
            cmd.extend(['--left-glove-port', config['left_glove_port']])
        if config.get('right_glove_port'):
            cmd.extend(['--right-glove-port', config['right_glove_port']])
        if config.get('calibrate'):
            cmd.append('--calibrate')
            
        env = os.environ.copy()
        env['PYTHONPATH'] = WORKSPACE_ROOT + os.pathsep + env.get('PYTHONPATH', '')
        
        print(f"Starting Input Device {device_id} with command: {' '.join(cmd)}")
        
        process = subprocess.Popen(cmd, env=env)
        
        socket = self.create_socket(zmq_port, mode=zmq.REQ)
        
        self.input_devices[device_id] = {
            'process': process,
            'config': config,
            'zmq_port': zmq_port,
            'vuer_zmq_port': vuer_zmq_port,
            'vuer_port': config.get('vuer_port'),
            'viser_port': viser_port,
            'socket': socket,
            'status': 'starting'
        }
        
        return device_id

    def remove_input_device(self, device_id):
        if device_id in self.input_devices:
            device = self.input_devices[device_id]
            try:
                device['socket'].send_json({'cmd': 'quit'})
            except:
                pass
            device['process'].terminate()
            del self.input_devices[device_id]
            return True
        return False

    def set_input_output_dir(self, device_id, output_dir):
        if device_id in self.input_devices:
            device = self.input_devices[device_id]
            try:
                device['socket'].send_json({
                    'cmd': 'set_output_dir',
                    'output_dir': output_dir
                })
                reply = device['socket'].recv_json()
                if reply.get('status') == 'success':
                    device['config']['output_dir'] = output_dir
                    self.is_calibrated = False
                    return True
            except Exception as e:
                print(f"Error setting output dir for device {device_id}: {e}")
        return False

    def send_robot_command(self, cmd):
        if self.robot_socket:
            try:
                self.robot_socket.send_json({'cmd': cmd})
                return self.robot_socket.recv_json()
            except zmq.ZMQError as e:
                return {'status': 'error', 'message': str(e)}
        return {'status': 'error', 'message': 'Robot not running'}

    def trigger_collection(self):
        # Get current pose from robot
        state = self.send_robot_command('get_state')
        if state['status'] != 'success':
            return state
            
        idx = state['idx']
        pose_name = state['pose_name']
        
        results = {}
        # Send requests concurrently
        for dev_id, dev in self.input_devices.items():
            try:
                dev['socket'].send_json({
                    'cmd': 'collect',
                    'idx': idx,
                    'pose_name': pose_name
                })
            except Exception as e:
                results[dev_id] = {'status': 'error', 'message': str(e)}
        
        # Wait for all replies
        for dev_id, dev in self.input_devices.items():
            if dev_id in results: continue
            try:
                # Set a 10s timeout for collection
                dev['socket'].setsockopt(zmq.RCVTIMEO, 10000)
                reply = dev['socket'].recv_json()
                results[dev_id] = reply
            except zmq.Again:
                results[dev_id] = {'status': 'error', 'message': 'Timeout: Please check all the devices can see the hands properly'}
            except Exception as e:
                results[dev_id] = {'status': 'error', 'message': str(e)}
            finally:
                # create a new socket to avoid issues
                dev['socket'].setsockopt(zmq.RCVTIMEO, -1)
                if results[dev_id]['status'] != 'success':
                    dev['socket'].close()
                    new_socket = self.create_socket(dev['zmq_port'], mode=zmq.REQ)
                    dev['socket'] = new_socket
                
        return {'status': 'success', 'results': results}

    def trigger_calibration(self):
        results = {}
        # Send requests concurrently
        for dev_id, dev in self.input_devices.items():
            try:
                dev['socket'].send_json({
                    'cmd': 'calibrate'
                })
            except Exception as e:
                results[dev_id] = {'status': 'error', 'message': str(e)}
        self.is_calibrated = True
        # Wait for all replies
        for dev_id, dev in self.input_devices.items():
            if dev_id in results: continue
            try:
                # Set a 60s timeout for calibration (10s collection + fitting)
                dev['socket'].setsockopt(zmq.RCVTIMEO, 60000)
                reply = dev['socket'].recv_json()
                results[dev_id] = reply
            except zmq.Again:
                results[dev_id] = {'status': 'error', 'message': 'Timeout: Calibration took too long (>60s)'}
            except Exception as e:
                results[dev_id] = {'status': 'error', 'message': str(e)}
            finally:
                # Reset timeout
                dev['socket'].setsockopt(zmq.RCVTIMEO, -1)
                # create a new socket to avoid issues
                if results[dev_id]['status'] != 'success':
                    dev['socket'].close()
                    new_socket = self.create_socket(dev['zmq_port'], mode=zmq.REQ)
                    dev['socket'] = new_socket
                    self.is_calibrated = False
        print("Calibration results:", results)
        return {'status': 'success' if self.is_calibrated else 'error', 'results': results}

    def get_status(self):
        return {
            'ip': get_ip_address(),
            'is_calibrated': self.is_calibrated,
            'robot': {
                'running': self.robot_process is not None,
                'config': self.robot_config,
                'viser_port': self.robot_viser_port
            },
            'inputs': {
                dev_id: {
                    'config': dev['config'],
                    'zmq_port': dev['zmq_port'],
                    'vuer_zmq_port': dev['vuer_zmq_port'],
                    'viser_port': dev.get('viser_port'),
                    'vuer_port': dev.get('vuer_port'),
                } for dev_id, dev in self.input_devices.items()
            }
        }

manager = DeviceManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(manager.get_status())

@app.route('/api/robot', methods=['POST'])
def start_robot():
    config = request.json
    if manager.start_robot(config):
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'})

@app.route('/api/robot/command', methods=['POST'])
def robot_command():
    cmd = request.json.get('cmd')
    return jsonify(manager.send_robot_command(cmd))

@app.route('/api/inputs', methods=['POST'])
def add_input():
    config = request.json
    dev_id = manager.add_input_device(config)
    return jsonify({'status': 'success', 'device_id': dev_id})

@app.route('/api/inputs/<dev_id>', methods=['DELETE'])
def remove_input(dev_id):
    if manager.remove_input_device(dev_id):
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 404

@app.route('/api/inputs/<dev_id>/output_dir', methods=['POST'])
def set_input_output_dir(dev_id):
    output_dir = request.json.get('output_dir')
    if manager.set_input_output_dir(dev_id, output_dir):
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 400

@app.route('/api/collect', methods=['POST'])
def collect():
    return jsonify(manager.trigger_collection())

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    return jsonify(manager.trigger_calibration())

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    data_dir = os.path.join(WORKSPACE_ROOT, 'data') # Assuming default data dir
    # Or use the output_dir from config if available, but for now let's search recursively
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith('.pkl'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, data_dir)
                files.append({'path': full_path, 'name': rel_path})
    return jsonify(files)

@app.route('/api/playback', methods=['POST'])
def start_playback():
    file_path = request.json.get('path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': 'File not found'})
    
    viser_port = 8090 # Fixed port for playback for now
    
    cmd = [
        sys.executable, 
        os.path.join(os.path.dirname(__file__), 'playback_node.py'),
        '--file-path', file_path,
        '--viser-port', str(viser_port)
    ]
    
    env = os.environ.copy()
    env['PYTHONPATH'] = WORKSPACE_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    
    # Kill existing playback if any (simple approach)
    # Ideally manager should track this
    subprocess.Popen(['pkill', '-f', 'playback_node.py'])
    time.sleep(0.5)
    
    subprocess.Popen(cmd, env=env)
    
    return jsonify({'status': 'success', 'viser_port': viser_port})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()
    
    try:
        app.run(host='0.0.0.0', port=args.port, debug=True, use_reloader=False)
    finally:
        manager.stop_robot()
        for dev_id in list(manager.input_devices.keys()):
            manager.remove_input_device(dev_id)
