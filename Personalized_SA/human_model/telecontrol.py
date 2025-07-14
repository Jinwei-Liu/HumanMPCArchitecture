import numpy as np
import sys
import os
import time
import pygame
import threading
from typing import Optional, Tuple, Dict, Any

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the QuadrotorRaceEnv from your existing code
from Personalized_SA.env.quadrotor_env import QuadrotorRaceEnv

class RemoteController:
    """
    Class to interface with a quadrotor remote controller using pygame joystick.
    This simulates a typical quadrotor remote with two sticks:
    - Left stick: throttle (up/down) and yaw (left/right)
    - Right stick: pitch (up/down) and roll (left/right)
    """
    def __init__(self):
        # Initialize pygame for joystick input
        pygame.init()
        pygame.joystick.init()
        
        # Check if any joysticks/controllers are connected
        self.joystick_count = pygame.joystick.get_count()
        if self.joystick_count == 0:
            print("No joystick detected. Please connect a controller.")
            self.controller = None
            self.initialized = False
        else:
            # Initialize the first joystick
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            self.initialized = True
            print(f"Initialized controller: {self.controller.get_name()}")
            print(f"Number of axes: {self.controller.get_numaxes()}")
            print(f"Number of buttons: {self.controller.get_numbuttons()}")
            
        # Initialize controller state
        self.throttle = 0.0  # Vertical thrust (0-1)
        self.yaw = 0.0       # Rotation around vertical axis (-1 to 1)
        self.pitch = 0.0     # Forward/backward tilt (-1 to 1)
        self.roll = 0.0      # Left/right tilt (-1 to 1)
        
        # Mapping of controller values to quadrotor actions
        self.thrust_scale = 20.0  # Max thrust value
        self.rotation_scale = 3.0  # Max rotation value
        
        # Running flag for the controller thread
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the controller input thread"""
        if not self.initialized:
            print("Controller not initialized. Cannot start.")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def stop(self):
        """Stop the controller input thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        pygame.quit()
    
    def _update_loop(self):
        """Main loop for updating controller values"""
        while self.running:
            # Process pygame events
            pygame.event.pump()
            
            # Read joystick values
            if self.controller:
                # Different controllers might have different axis mappings
                # Common mapping for quadrotor-like controllers:
                # Left stick vertical (throttle)
                self.throttle = self.controller.get_axis(2) * 0.5 + 0.5  # Normalize to 0-1
                # Left stick horizontal (yaw)
                self.yaw = -self.controller.get_axis(3)
                # Right stick vertical (pitch)
                self.pitch = -self.controller.get_axis(0)
                # Right stick horizontal (roll)
                self.roll = self.controller.get_axis(1)
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.01)
    
    def get_action(self) -> np.ndarray:
        """
        Convert controller inputs to quadrotor action values
        Returns:
            np.ndarray: [thrust, roll, pitch, yaw]
        """
        if not self.initialized:
            # Return neutral action if controller not initialized
            return np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Convert controller values to action space
        thrust = self.throttle * self.thrust_scale
        roll_cmd = self.roll * self.rotation_scale
        pitch_cmd = self.pitch * self.rotation_scale
        yaw_cmd = self.yaw * self.rotation_scale
        
        return np.array([thrust, roll_cmd, pitch_cmd, yaw_cmd], dtype=np.float32)
    
    def get_raw_input(self) -> Dict[str, float]:
        """Return raw controller input values for debugging"""
        return {
            "throttle": self.throttle,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll
        }

def test_controller():
    """
    Function to test the remote controller without running the simulator.
    Displays the raw controller values and calculated actions.
    """
    controller = RemoteController()
    if not controller.start():
        print("Failed to start controller. Exiting test.")
        return
    
    try:
        print("Testing controller. Press Ctrl+C to exit.")
        print("Move the controller sticks to see the values change.")
        
        while True:
            # Get and display raw controller values
            raw_input = controller.get_raw_input()
            action = controller.get_action()
            
            # Clear terminal and display values
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Controller Test Mode")
            print("-------------------")
            print(f"Throttle: {raw_input['throttle']:.2f}")
            print(f"Yaw:      {raw_input['yaw']:.2f}")
            print(f"Pitch:    {raw_input['pitch']:.2f}")
            print(f"Roll:     {raw_input['roll']:.2f}")
            print("\nQuadrotor Actions:")
            print(f"Thrust:   {action[0]:.2f}")
            print(f"Roll:     {action[1]:.2f}")
            print(f"Pitch:    {action[2]:.2f}")
            print(f"Yaw:      {action[3]:.2f}")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nTest ended by user.")
    finally:
        controller.stop()
        print("Controller stopped.")


def control_quadrotor():
    """
    Main function to control the quadrotor using the remote controller.
    Initializes the quadrotor environment and controller, then runs the control loop.
    """
    # Initialize the environment with visualization
    env = QuadrotorRaceEnv(dt=0.01, mode='control')
    obs, _ = env.reset(seed=42)
    
    # Initialize the controller
    controller = RemoteController()
    if not controller.start():
        print("Failed to start controller. Exiting.")
        env.close()
        return
    
    try:
        print("Controlling quadrotor with remote controller.")
        print("Press Ctrl+C to exit.")
        
        done = False
        total_reward = 0.0
        step_count = 0
        
        # Main control loop
        while not done:
            # Get action from controller
            action = controller.get_action()
            
            # Apply action to environment
            obs, reward, done, info = env.step(action)
            
            # Render the environment
            env.render()
            
            # Track statistics
            total_reward += reward
            step_count += 1
            
            # Display information
            if step_count % 10 == 0:  # Update display every 10 steps to reduce clutter
                print(f"Step {step_count}: Reward = {reward:.2f}")
                print(f"Action: {action}")
                print(f"Position: {env.quad.get_position()}")
                print(f"Gate {env.current_gate_idx + 1}/{env.num_gates}")
                if info:
                    print(f"Info: {info}")
            
            # Small delay to maintain simulation speed
            time.sleep(0.03)
    
    except KeyboardInterrupt:
        print("\nControl ended by user.")
    finally:
        # Clean up
        controller.stop()
        env.close()
        print(f"Final stats - Total reward: {total_reward:.2f}, Steps: {step_count}")


def keyboard_control():
    """
    Alternative control method using keyboard inputs instead of a physical controller.
    Useful for testing when no controller is available.
    """
    # Initialize pygame for keyboard input
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Quadrotor Keyboard Control")
    
    # Initialize the environment with visualization
    env = QuadrotorRaceEnv(dt=0.01, mode='human')
    obs, _ = env.reset(seed=42)
    print(obs)
    
    # Control parameters
    thrust = 10.0  # Default thrust to hover
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    
    # Control sensitivity
    thrust_delta = 1.0
    rotation_delta = 2.0
    
    try:
        print("Controlling quadrotor with keyboard.")
        print("Controls:")
        print("  W/S - Increase/decrease thrust")
        print("  A/D - Roll left/right")
        print("  Up/Down - Pitch forward/backward")
        print("  Left/Right - Yaw left/right")
        print("  R - Reset environment")
        print("  ESC - Exit")
        
        done = False
        total_reward = 0.0
        step_count = 0
        clock = pygame.time.Clock()
        
        # Main control loop
        running = True
        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        obs, _ = env.reset(seed=42)
                        done = False
                        total_reward = 0.0
                        step_count = 0
            
            # Get keyboard state
            keys = pygame.key.get_pressed()
            
            # Update control values based on keys
            if keys[pygame.K_w]:
                thrust = min(thrust + thrust_delta, 20.0)
            if keys[pygame.K_s]:
                thrust = max(thrust - thrust_delta, 0.0)
            if keys[pygame.K_a]:
                roll = -rotation_delta
            elif keys[pygame.K_d]:
                roll = rotation_delta
            else:
                roll = 0.0
            if keys[pygame.K_UP]:
                pitch = -rotation_delta
            elif keys[pygame.K_DOWN]:
                pitch = rotation_delta
            else:
                pitch = 0.0
            if keys[pygame.K_LEFT]:
                yaw = -rotation_delta
            elif keys[pygame.K_RIGHT]:
                yaw = rotation_delta
            else:
                yaw = 0.0
            
            # Create action
            action = np.array([thrust, roll, pitch, yaw], dtype=np.float32)
            
            if not done:
                # Apply action to environment
                obs, reward, done, info = env.step(action)
                
                # Render the environment
                env.render()
                
                # Track statistics
                total_reward += reward
                step_count += 1
                
                # Display information on pygame screen
                screen.fill((0, 0, 0))
                font = pygame.font.Font(None, 36)
                text_lines = [
                    f"Step: {step_count}",
                    f"Reward: {reward:.2f}",
                    f"Total Reward: {total_reward:.2f}",
                    f"Thrust: {thrust:.2f}",
                    f"Roll: {roll:.2f}",
                    f"Pitch: {pitch:.2f}",
                    f"Yaw: {yaw:.2f}",
                    f"Gate: {env.current_gate_idx + 1}/{env.num_gates}"
                ]
                
                for i, line in enumerate(text_lines):
                    text = font.render(line, True, (255, 255, 255))
                    screen.blit(text, (20, 20 + i * 30))
                
                pygame.display.flip()
            elif running:
                # Reset if done
                print("Episode finished. Press 'R' to reset or ESC to exit.")
                if keys[pygame.K_r]:
                    obs, _ = env.reset(seed=42)
                    done = False
                    total_reward = 0.0
                    step_count = 0
            
            # Control frame rate
            clock.tick(60)
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        env.close()
        pygame.quit()
        print(f"Final stats - Total reward: {total_reward:.2f}, Steps: {step_count}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quadrotor Remote Controller')
    parser.add_argument('--mode', type=str, default='control', 
                        choices=['test', 'control', 'keyboard'],
                        help='Mode to run: test (controller only), control (with quadrotor), keyboard')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_controller()
    elif args.mode == 'control':
        control_quadrotor()
    elif args.mode == 'keyboard':
        keyboard_control()
