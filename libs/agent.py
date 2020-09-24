import numpy as np
import cerebellum


class Agent():
    
    
    def __init__(self,ctrl_scheme='fel', beta_s=[1,1], mass = 650., mass_pole = 250., load_memory = False):
        
        self.t = 0
        self.dt = .01
        
        self.p  = 0.
        self.th = 0.
        self.p_prime = 0.
        self.th_prime = 0.
        
        self.mass = mass
        self.mass_pole = mass_pole

        self.Kp_v = 0.004
        self.Ki_v = 0.
        self.Kd_v = 0.
        self.Kp_th = 9000.
        self.Ki_th = 0.
        self.Kd_th = 0.
        
        self.err_v = 0.
        self.err_v_i = 0
        self.err_v_t = 0
       
        self.err_th_i = 0
        self.err_th_t = 0
        self.err_th = 0.

        self.delay_fb = 80
        self.sens_fb = np.zeros([self.delay_fb,4])
        self.del_fbk_idx = 0
        
        self.delay_hspc = 240
        self.del_ff_d_out = np.zeros(self.delay_hspc)
        self.del_hspc_idx = 0
        self.x_p_p = 0
        
        self.beta_1 = beta_s[0]
        self.beta_2 = beta_s[1]
        
        self.ctrl_scheme = ctrl_scheme
        self.load_memory = load_memory
        
        self.xpp_log = []
        self.ff_d_log = []
        self.err_v_log = []

    
    def plant(self,u):
        
        M_t = self.mass 
        m = self.mass_pole 
        l = 30.
        J = .06
        J_t = J+m*l**2
        g = 9.8
        c = .1
        gamma = .1
        

        aa = -m*l*np.sin(self.th)*self.th_prime**2
        bb = m*g*(m*l**2/J_t)*np.sin(self.th)*np.cos(self.th)
        cc = -c*self.p_prime
        dd = -(gamma/J_t)*l*m*np.cos(self.th)*self.th_prime
        ee = u
        ff = M_t-m*(m*l**2/J_t)*np.cos(self.th)**2
        
        
        self.p_prime = ((aa+bb+cc+dd+ee)/ff)*self.dt  # (e1n1 + e1n2 + e1n3 + e1n4 + e1n5)/e1d;

        
        gg = -m*l**2*np.sin(self.th)*np.cos(self.th)*self.th_prime**2
        hh = M_t*g*l*np.sin(self.th)
        ii = -c*l*np.cos(self.th)*self.p_prime
        ll = -gamma*(M_t/m)*self.th_prime
        mm = l*np.cos(self.th)*u
        nn = J_t*(M_t/m)-m*(l*np.cos(self.th))**2
    
        self.th_prime = ((gg+hh+ii+ll+mm)/nn)*self.dt
        
        self.p  += self.p_prime
        self.th += self.th_prime
        
        self.sens_fb[self.del_fbk_idx-1,:] = np.array([self.p,self.th,self.p_prime,self.th_prime])
        
        
    def feedback_control(self,s_fb, tgt, err_pred = [0,0]):

        v, th = s_fb[0], s_fb[1]

        self.err_v = -v
        self.err_v_log.append(self.err_v)
        
        err_v_pred = self.err_v  - err_pred[1]
        self.err_v_i += err_v_pred
        err_v_d = err_v_pred - self.err_v_t
        self.err_v_t = err_v_pred
        th_tgt = -self.Kp_v*err_v_pred+self.Ki_v*self.err_v_i+self.Kd_v*err_v_d

        self.err_th = th_tgt-th
        err_th_pred = self.err_th + err_pred[0]
        self.err_th_i += err_th_pred
        err_th_d = err_th_pred - self.err_th_t
        self.err_th_t = err_th_pred
        u_fb = self.Kp_th*err_th_pred+self.Ki_th*self.err_th_i+self.Kd_th*err_th_d

        return u_fb
    
    
    def update(self, f, x_d, x_p):
        
        ff_out = 0
        
        if self.ctrl_scheme == 'hspc':
            u_hspc,ff_out = self.hSPC(x_d, x_p, self.sens_fb[self.del_fbk_idx,:2], 0)
            u = f+u_hspc
        
        elif self.ctrl_scheme == 'fel':
            u_fel,ff_out = self.FEL(x_d, x_p, self.sens_fb[self.del_fbk_idx,:2], 0)
            u = f+u_fel
        
        else:
            u_fbk = self.feedback_control(self.sens_fb[self.del_fbk_idx,:],0)
            u = f+u_fbk
            
        self.plant(u)
        self.del_fbk_idx = self.t%self.delay_fb
        
        self.t+=1
        
        return np.array([self.p,self.th,self.p_prime,self.th_prime,ff_out])
    
    
    def plant_reset(self):
    
        self.p  = 0.
        self.th = 0.
        self.p_prime = 0.
        self.th_prime = 0. 
        
        self.err_v = 0.
        self.err_v_i = 0
        self.err_v_t = 0
       
        self.err_th_i = 0
        self.err_th_t = 0
        self.err_th = 0.
        
        self.t = 0
        
        self.xpp_log = []
        self.ff_d_log = []
        
        self.sens_fb = np.zeros([self.delay_fb,4])
        self.x_p_p = 0
        
        
    def ff_reset(self):

        self.ff_d = cerebellum.Cerebellum()
        self.ff_p = cerebellum.Cerebellum()
        
        
        if self.load_memory == False:
            
            self.ff_d.config("./libs/basis_apas_sim.cfg")
            self.ff_d.plug(1,     #
                      1000,  #
                      1,     #
                      0.0,   #
                      self.beta_1,    # 
                      0.2,   #
                      100)   #

            self.ff_p.config("./libs/basis_apas_sim.cfg")
            self.ff_p.plug(1,     #
                      1000,  #
                      1,     #
                      0.0,   #
                      self.beta_2,    #
                      0.2,   # 
                      100)   #
        
        
            self.ff_d.loadMemory('./libs/proto_memory.cfg')
            self.ff_p.loadMemory('./libs/proto_memory.cfg')
            
            print 'proto memory loaded'
            
        if self.load_memory == True:
            
            if self.ctrl_scheme == 'fel':
                self.ff_d.loadMemory('fel_ff_d.cfg')
                self.ff_p.loadMemory('fel_ff_p.cfg')
                
            elif self.ctrl_scheme == 'hspc':
                self.ff_d.loadMemory('hspc_ff_d.cfg')
                self.ff_p.loadMemory('hspc_ff_p.cfg')
                
            print 'trained memory loaded'

        
        
    def ff_start(self):

        self.ff_d.initTrial()
        self.ff_p.initTrial()
        
        
    def ff_update(self):

        self.ff_d.endTrial()
        self.ff_p.endTrial()

        self.t_hspc = 0.
        

    def FEL(self, x_d, x_p, sens_fb, tgt_fb):

        u_fb = self.feedback_control(sens_fb,tgt_fb)

        ff_d_out = self.ff_d.input([x_d,u_fb])[0]
        ff_p_out = self.ff_p.input([x_p,u_fb])[0]

        u_ff_out = ff_d_out+ff_p_out
        u_fel = u_ff_out+u_fb

        return u_fel,u_ff_out
    
    
    def hSPC(self, x_d, x_p, sens_fb, tgt_fb):

        ff_d_out = self.ff_d.input([x_d,self.x_p_p])[0]
        self.del_ff_d_out[self.del_hspc_idx-1] = ff_d_out
        self.x_p_p = (x_p - self.del_ff_d_out[self.del_hspc_idx])
        
        self.xpp_log.append(self.x_p_p)
        self.ff_d_log.append(ff_d_out)

        ff_p_out = np.array([0.,self.ff_p.input([max(self.x_p_p+ff_d_out,0),self.err_v])[0]])
        u_hspc = self.feedback_control(sens_fb,tgt_fb,ff_p_out)

        self.del_hspc_idx = self.t%self.delay_hspc

        return u_hspc,ff_p_out[1]
