import numpy as np
import math
import argparse
import os
from src.helper_funcs import Params

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/idm', help="Directory to save idm data, and find config file")

def one_step_idm(state, para):
    # state: [dx, dv, v]
    # para: v0, T, s0, a_max, b
    dx = state[0]; dv = state[1]; v = state[2]
    root = math.sqrt(para[3] * para[4])
    temp = v * para[1]-0.5 * v * dv / root

    desired_s_n = para[2] + temp
    a = para[3] * (1 - (v / para[0])**4 - (desired_s_n / dx)**2)
    #if a < -2:
    #    a = -2    
    return a

class ParaGen():
    def __init__(self, para_mean=[10, 1, 1, 1, 1],
                 para_std = [1, 0.1, 0.1, 0.1, 0.1],
                    variant_bound = [0.01, 100]):
        '''
        para: v0, T, s0, a, b
        '''
        self.para_mean = para_mean
        self.para_std = para_std
        self.bound = variant_bound
        self.r = np.random.RandomState(1)
        
    def get_para(self):
        count = 0
        while True:
            count += 1
            new_para = [self.r.normal(self.para_mean[i], self.para_std[i]) 
                       for i in range(len(self.para_mean))]
            new_para = np.array(new_para)
            if sum(new_para<self.bound[0]) > 0:
                print(new_para)
                continue
            if sum(new_para>self.bound[1]) > 0:
                continue
            break
        return new_para
    

class StateGen():
    def __init__(self, dx_lim=[10,50], dv_lim=[-8,8], v_lim=[2,10]):
        self.dx_lim = dx_lim
        self.dv_lim = dv_lim
        self.v_lim = v_lim
        self.r = np.random.RandomState(2)
    def get_state(self):
        state = []
        for item in [self.dx_lim, self.dv_lim, self.v_lim]:
            state.append( self.r.uniform( item[0],item[1],1 )[0] )
        return state

class StateAccGenerator():
    def __init__(self, StateGen_conf, ParaGen_conf, external_sigma = 0.05):
        self.stategen = StateGen(dx_lim = StateGen_conf["dx_lim"],
                                  dv_lim = StateGen_conf["dv_lim"],
                                  v_lim = StateGen_conf["v_lim"])
        self.paragen = ParaGen(para_mean = ParaGen_conf["para_mean" ], 
                               para_std = ParaGen_conf["para_std"],                     #data.append(para+state+[a])sss
                               variant_bound = ParaGen_conf["variant_bound"])
        self.external_sigma = external_sigma
        self.r = np.random.RandomState(3)
    def get_data(self, N_s, N_a_for_one_s, dt = 0.1, action_as = "a"):
        # action_as:
        # a: use acceleration as action
        # v: use velocity as action
        data = []
        for _ in range(N_s):
            while True:
                A = []
                state = self.stategen.get_state()
                for _ in range(N_a_for_one_s):
                    para = self.paragen.get_para()
                    a = one_step_idm(state, para)
                    a = self.r.normal(a, self.external_sigma,1)[0]
                    #if (a > 2) | (a < -4):
                    #    A = []
                    #    break
                    A.append(a)
                    #data.append(para+state+[a])
                if min(A) > -2:
                    if action_as == "a":
                        data.append(state + A)
                    elif action_as == "v":
                        A = [state[-1] + dt*a for a in A]
                        data.append(state + A)
                    else:
                        raise ValueError("action_as value wrong")
                    break
                else:
                    pass
        data_arr = np.array(data)
        return data_arr
    

    
if __name__ == "__main__":
    args = parser.parse_args()
    data_config_path = os.path.join(args.data_dir, "data_para.json")
    assert os.path.isfile(data_config_path), f"file not found: {data_config_path}"

    # if data already exist
    assert not os.path.exists(os.path.join(args.data_dir, 'train_data.csv')),\
        f" train data already exists: {args.data_dir}"
    assert not os.path.exists(os.path.join(args.data_dir, 'validation_data.csv')),\
        f" validation data already exists: {args.data_dir}"
    assert not os.path.exists(os.path.join(args.data_dir, 'test_data.csv')),\
        f" test data already exists: {args.data_dir}"
    assert not os.path.exists(os.path.join(args.data_dir, 'collocation_data.csv')),\
        f" collocation data already exists: {args.data_dir}"

    params = Params(data_config_path)

    # parameters
    external_sigma = params.external_sigma
    StateGen_conf = {
        "dx_lim" : [params.state_range['dx']['lower_bound'],
                    params.state_range['dx']['upper_bound']],
        "dv_lim" : [params.state_range['dv']['lower_bound'],
                    params.state_range['dv']['upper_bound']],
        "v_lim" : [params.state_range['v']['lower_bound'],
                   params.state_range['v']['upper_bound']],
    }


    # para: v0, T, s0, a, b
    ParaGen_conf = {
        "para_mean" :  [params.para_mean_stddev['v0']['mean'], params.para_mean_stddev['T']['mean'],
                        params.para_mean_stddev['s0']['mean'], params.para_mean_stddev['a_max']['mean'],
                        params.para_mean_stddev['b']['mean']],
        "para_std"  : [params.para_mean_stddev['v0']['stddev'], params.para_mean_stddev['T']['stddev'],
                       params.para_mean_stddev['s0']['stddev'], params.para_mean_stddev['a_max']['stddev'],
                       params.para_mean_stddev['b']['stddev']],
        "variant_bound": [params.para_bounds['lower_bound'], params.para_bounds['upper_bound']]
    }

    sa_gen = StateAccGenerator(StateGen_conf, ParaGen_conf, external_sigma)
    train_data = sa_gen.get_data(params.train_pool_size, 1, dt=params.time_step, action_as=params.action_as)
    validation_data = sa_gen.get_data(params.validation_pool_size, 1, dt=params.time_step, action_as=params.action_as)
    test_data = sa_gen.get_data(params.test_pool_size, params.n_samples, dt=params.time_step, action_as=params.action_as)
    collocation_data = sa_gen.get_data(params.test_pool_size, 1, dt=params.time_step, action_as=params.action_as)


    train_feature    =        train_data[:,:3]
    train_label         =     train_data[:,3:]

    validation_feature  =       validation_data[:,:3]
    validation_label    =       validation_data[:,3:]

    test_feature     =        test_data[:,:3]
    test_label          =     test_data [:,3:]

    collocation_feature  =    collocation_data[:,:3]

    # save the data
    np.savetxt(os.path.join(args.data_dir, 'train_feature.csv'), train_feature, delimiter=',')
    np.savetxt(os.path.join(args.data_dir, 'train_label.csv'), train_label, delimiter=',')
    np.savetxt(os.path.join(args.data_dir, 'validation_feature.csv'), validation_feature, delimiter=',')
    np.savetxt(os.path.join(args.data_dir, 'validation_label.csv'), validation_label, delimiter=',')
    np.savetxt(os.path.join(args.data_dir, 'test_feature.csv'), test_feature, delimiter=',')
    np.savetxt(os.path.join(args.data_dir, 'test_label.csv'), test_label, delimiter=',')
    np.savetxt(os.path.join(args.data_dir, 'collocation_feature.csv'), collocation_feature, delimiter=',')



