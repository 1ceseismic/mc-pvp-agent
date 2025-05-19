import time
import gym
import numpy as np
from gym import spaces
from malmo import MalmoPython
from malmoenv import Env
class MalmoPvPEnv(gym.Env):
    def __init__(self, mission_file, agent_role, port=10000, host='127.0.0.1'):
        super().__init__()
        self.agent_role = agent_role  # 'agent_1' or 'agent_2'
        self.agent_host = MalmoPython.AgentHost()
        # Connect to the local Malmo server
        self.agent_host.setClientPool([MalmoPython.ClientInfo(host, port)])
        # Load mission spec
        with open(mission_file, 'r') as f:
            self.mission_xml = f.read()
        self._setup_spaces()

    def _setup_spaces(self):
        # Discrete actions: move (no-op, forward, back, left, right), jump, attack
        self.action_space = spaces.Discrete(1 + 4 + 1 + 1)
        # Observation is an 84×84 RGB image
        self.observation_space = spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)

    def reset(self):
        mission = MalmoPython.MissionSpec(self.mission_xml, True)
        mission.requestVideo(84, 84)
        mission_record = MalmoPython.MissionRecordSpec()
        self.agent_host.startMission(mission, mission_record)
        # Wait for mission to start
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
        return self._get_observation(world_state)

    def step(self, action):
        # Map discrete action to commands
        cmds = {
            0: [],                # no-op
            1: ['move 1'],        # forward
            2: ['move -1'],       # back
            3: ['strafe 1'],      # right
            4: ['strafe -1'],     # left
            5: ['jump 1'],        # jump
            6: ['attack 1'],      # attack
        }.get(action, [])
        for cmd in cmds:
            self.agent_host.sendCommand(cmd)
        # Step one tick
        time.sleep(0.1)
        world_state = self.agent_host.getWorldState()
        # Get latest frame
        img = None
        for frame in world_state.video_frames:
            img = np.frombuffer(frame.pixels, dtype=np.uint8).reshape((84, 84, 3))
        # Get reward
        reward = 0.0
        for r in world_state.rewards:
            reward += r.getValue()
        done = not world_state.is_mission_running
        return img, reward, done, {}

    def _get_observation(self, world_state):
        # Similar to step: return initial frame
        for frame in world_state.video_frames:
            return np.frombuffer(frame.pixels, dtype=np.uint8).reshape((84, 84, 3))
        return np.zeros((84,84,3), dtype=np.uint8)
