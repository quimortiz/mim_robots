import pybullet as p
import numpy as np
from dynamic_graph_head import ThreadHead, SimHead, HoldPDController
from datetime import datetime
# import dynamic_graph_manager_cpp_bindings
from mim_robots.robot_loader import load_bullet_wrapper, load_pinocchio_wrapper
from mim_robots.robot_loader import load_bullet_wrapper, load_mujoco_model, load_pinocchio_wrapper, load_mujoco_wrapper
from mim_robots.pybullet.env import BulletEnvWithGround
from mim_robots.robot_list import MiM_Robots
import pathlib
import os
# os.sys.path.insert(1, str(python_path))
import time


class RandomGoalController:
    pass
    



# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Choose experiment, load config and import controller  #  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
SIM           = True
EXP_NAME      = 'random_move'
controller_class = RandomGoalController

    
    
    
    
# # # # # # # # # # # #
# Import robot model  #
# # # # # # # # # # # #
pin_robot   = load_pinocchio_wrapper('iiwa_pusher')

config = {}

config['q0'] =  [ 0.5604080525397099, 
    0.8670522246951389,
    0.10696889788203193,  
    -1.8091919335298106, 
    -0.17904272753850936, 
    0.47469638420974486, 
    0.7893769555580691]

config['dq0'] = [ 0,0,0,0,0,0,0 ]

config['ctrl_freq'] = 100


def time_to_str_with_ms(_now):
    return ("Time : %s.%s\n" % (time.strftime('%x %X',time.localtime(_now)),
          str('%.3f'%_now).split('.')[1])) # Rounds to nearest millisecond


class EnvMJWrapper:
    """
    """
    def __init__(self, robot_simulator, viewer=False):
        import mujoco
        import mujoco.viewer
        self.time_calls = []
        self.time_calls_raw = []
        self.dt = 0.001
        self.last_viewer_update = time.time()
        self.viewer_update_interval = 1. / 20.
        self.robot_simulator = robot_simulator
        if viewer:
            self.viewer = mujoco.viewer.launch_passive(self.robot_simulator.mjmodel, self.robot_simulator.mjdata)
        else:
            self.viewer = None
        time.sleep(.1)

    def step(self, sleep=None):
        """
        """
        self.time_calls.append(time_to_str_with_ms(time.time()))
        self.time_calls_raw.append(time.time())
        self.robot_simulator.step_simulation()
        current_time = time.time()
        if sleep:
            time.sleep(sleep)
        if self.viewer:
            if current_time - self.last_viewer_update > self.viewer_update_interval:
                self.last_viewer_update = current_time
                self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()




# # # # # # # # # # # # #
# Setup control thread  #
# # # # # # # # # # # # #
if SIM:
    # Sim env + set initial state 
    config['T_tot'] = 30

    robot_simulator = load_mujoco_wrapper('iiwa')
    env = EnvMJWrapper(robot_simulator, viewer=False)

    # env = BulletEnvWithGround(p.GUI)
    # robot_simulator = load_bullet_wrapper('iiwa')
    # env.add_robot(robot_simulator)

    # env.add_robot(robot_simulator)
    q_init = np.asarray(config['q0'] )
    v_init = np.asarray(config['dq0'])
    robot_simulator.reset_state(q_init, v_init)
    robot_simulator.forward_robot(q_init, v_init)

    # <<<<< Customize your PyBullet environment here if necessary
    head = SimHead(robot_simulator, with_sliders=False)
# !!!!!!!!!!!!!!!!
# !! REAL ROBOT !!
# !!!!!!!!!!!!!!!!
else:
    config['T_tot'] = 400              
    path = MiM_Robots['iiwa'].dgm_path  
    print(path)
    head = dynamic_graph_manager_cpp_bindings.DGMHead(path)
    target = None
    env = None

ctrl = HoldPDController(head, 50., 0.5, with_sliders=False)

# MPCController(head, pin_robot, config, run_sim=SIM)

thread_head = ThreadHead(
    1./config['ctrl_freq'],                                         # dt.
    HoldPDController(head, 50., 0.5, with_sliders=False),           # Safety controllers.
    head,                                                           # Heads to read / write from.
    [], 
    env                                                             # Environment to step.
)

thread_head.switch_controllers(ctrl)





# # # # # # # # #
# Data logging  #
# # # # # # # # # <<<<<<<<<<<<< Choose data save path & log config here (cf. launch_utils)
# prefix     = "/home/skleff/data_sqp_paper_croc2/constrained/circle/"
prefix     = "/tmp/"
suffix     = "_"+config['SOLVER'] +'_CODE_SPRINT'
LOG_FIELDS = launch_utils.get_log_config(EXP_NAME) 
# print(LOG_FIELDS)
# LOG_FIELDS = launch_utils.LOGS_NONE 
# LOG_FIELDS = launch_utils.SSQP_LOGS_MINIMAL 
# LOG_FIELDS = launch_utils.CSSQP_LOGS_MINIMAL 







# # # # # # # # # # # 
# Launch experiment #
# # # # # # # # # # # 
if SIM:
    thread_head.start_logging(int(config['T_tot']), prefix+EXP_NAME+"_SIM_"+str(datetime.now().isoformat())+suffix+".mds", LOG_FIELDS=LOG_FIELDS)
    thread_head.sim_run_timed(int(config['T_tot']))
    thread_head.stop_logging()
else:
    thread_head.start()
    thread_head.start_logging(50, prefix+EXP_NAME+"_REAL_"+str(datetime.now().isoformat())+suffix+".mds", LOG_FIELDS=LOG_FIELDS)
    
print("experiment done")

if type(env) == EnvMJWrapper:
    env.close() # I want to close the mujoco viewer if opened

import matplotlib.pyplot as plt


if type(env) == EnvMJWrapper:
    times = np.array(env.time_calls_raw)
    times -= times[0]
    plt.plot(times, '.')
    plt.show()


_history_measurements = head._history_measurements,
import pdb; pdb.set_trace()


# robot_simulator.viewer.close() 
# thread_head.plot_timing() # <<<<<<<<<<<<< Comment out to skip timings plot


from mim_data_utils import DataLogger, DataReader
from plots.plot_utils import SimpleDataPlotter
r = DataReader(thread_head.log_filename)

s = SimpleDataPlotter(dt=1./config['ctrl_freq'])

ee_lb = r.data['lb'] 
ee_ub = r.data['ub'] 

from croco_mpc_utils.pinocchio_utils import get_p_

frameId     = pin_robot.model.getFrameId('contact')
nq = pin_robot.model.nq ; 
nv = pin_robot.model.nv

p_mea = get_p_(r.data['joint_positions'], pin_robot.model, pin_robot.model.getFrameId('contact'))
p_des = get_p_(r.data['x_des'][:,:nq], pin_robot.model, pin_robot.model.getFrameId('contact'))



N = r.data['absolute_time'].shape[0]

target_position = np.zeros((N,3))
target_position[:,0] = r.data['target_position_x'][:,0]
target_position[:,1] = r.data['target_position_y'][:,0]
target_position[:,2] = r.data['target_position_z'][:,0]




s.plot_ee_pos([p_mea, 
               p_des,
               target_position,
               ee_lb,
               ee_ub],  
            ['Measured', 
             'Predicted',
             'Reference',
             'lb',
             'ub'], 
            ['r', 'b', 'g', 'k', 'k'], 
            linestyle=['solid', 'solid', 'dotted', 'dotted', 'dotted'],
            ylims=[[-0.8,-0.5,0],[+0.8,+0.5,1.5]])

plt.show()
