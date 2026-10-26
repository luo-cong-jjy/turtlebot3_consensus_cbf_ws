#!/usr/bin/env python3
"""ROS-free adapters for the validated MPC: timed reference and observed ellipses."""
from collections import deque
import numpy as np
from mpc_dcbf_static_obstacles import reference_covariance_matrices, reference_axis_inflation


def default_config():
    return dict(horizon=10, dt=.1, margin=.12, gamma=.3, omega_max=2.84,
        plan_v_max=.22, delta_limit=False, delta_v=.06, delta_w=.5,
        q_state=[35.,35.,.6], q_terminal=[80.,80.,1.], q_plan=[20.,20.],
        q_coupling=[30.,30.], r_control=[.15,.04], r_plan=[.04,.04],
        r_delta=[.05,.16], r_delta_plan=[.03,.03], max_iter=120,
        fallback_kv=1.8, fallback_kw=3., continuous_guidance=True,
        guide_velocity_weight=600., guide_heading_weight=2., guide_delta_scale=8.)


class TimedReference:
    def __init__(self, robot_id, dt=.1, horizon=10, radius=.5, robot_count=5):
        self.dt, self.n, self.radius, self.robot_id = dt, horizon, radius, robot_id
        self.robot_count=robot_count
        self.history = deque(maxlen=11)

    def offset(self):
        angle = self.robot_id*2*np.pi/self.robot_count
        return self.radius*np.array([np.cos(angle),np.sin(angle)])

    def observe(self, logical_time, position, velocity):
        if self.history and logical_time <= self.history[-1][0]+1e-7:
            return
        self.history.append((logical_time,np.asarray(position).copy(),np.asarray(velocity).copy()))

    def goals(self, age, current_heading):
        t,pos,vel = self.history[-1]
        coefficients=None
        weight=0.
        if len(self.history)==11:
            times=np.array([s[0]-t for s in self.history])
            design=np.column_stack([np.ones(11),times,times**2])
            coefficients=np.linalg.lstsq(design,np.array([s[1] for s in self.history]),rcond=None)[0]
            weight=np.clip((t-3.)/.5,0.,1.)
        def target(tau):
            # Legacy leader message carries p_L(t+dt); the unchanged observer
            # integrates once more. Account for both, plus real message age.
            s=tau+age-2*self.dt
            value=pos+s*vel
            if coefficients is not None:
                value=(1-weight)*value+weight*(np.array([1.,s,s*s])@coefficients)
            return value+self.offset()
        result=np.zeros((self.n+1,3));previous=target(-self.dt)
        angle=current_heading
        for j in range(self.n+1):
            result[j,:2]=target(j*self.dt)
            delta=result[j,:2]-previous
            if np.linalg.norm(delta)>1e-8:angle=np.arctan2(delta[1],delta[0])
            result[j,2]=angle;previous=result[j,:2]
        return result


class ObservedEnvelopes:
    """Apply upstream covariance recursion to measured ellipses, not world truth."""
    def __init__(self, dt=.1, horizon=10):
        self.dt,self.n=dt,horizon
        self.A,self.Q,self.R,self.H,self.P0=reference_covariance_matrices(dt)
        self.tracks={}

    def update(self, ellipses, stamp):
        output=[]
        for raw in ellipses:
            key=raw['id'];previous=self.tracks.get(key)
            p=self.P0.copy() if previous is None else previous[1]
            # Unlike the known-geometry simulation, a new ID already carries
            # a real measurement. Condition P0 on that observation immediately;
            # otherwise every ID reappearance injects a .3m prior-only envelope.
            if previous is None or stamp>previous[0]+1e-6:
                predicted=self.A@p@self.A.T+self.Q
                measurement=self.R.copy()
                variance=np.clip(raw.get('variance',0.),.005,.05)
                scale=3000.**(np.log10(variance/.005))
                measurement[0,0]*=scale;measurement[1,1]*=scale
                gain=np.linalg.solve(self.H@predicted@self.H.T+measurement,self.H@predicted).T
                p=predicted-gain@self.H@predicted;p=(p+p.T)/2
            self.tracks[key]=(stamp,p.copy())
            forecast=p.copy();axes=[]
            for j in range(self.n+1):
                forecast=self.A@forecast@self.A.T+self.Q
                inflation=reference_axis_inflation(forecast)
                axes.append([raw['a']+inflation,raw['b']+inflation])
            obs=dict(raw);obs.update(a=axes[0][0],b=axes[0][1],
                prediction_axes=np.asarray(axes),name='laser_'+str(key))
            output.append(obs)
        self.tracks={k:v for k,v in self.tracks.items() if stamp-v[0]<2.}
        return output


def peer_obstacle(peer_id, state, velocity, dt, n, radius=.20):
    state=np.asarray(state,float).copy();centers=[state[:2].copy()]
    for j in range(n):
        v,w=velocity;state[2]+=dt*w
        state[:2]+=dt*v*np.array([np.cos(state[2]),np.sin(state[2])])
        centers.append(state[:2].copy())
    return dict(center=np.asarray(centers[0]),a=radius,b=radius,theta=0.,
        prediction_centers=np.asarray(centers),name='peer_'+str(peer_id),peer=True,guidance=False)


def padded_obstacles(obstacles, count):
    if len(obstacles)>count:raise ValueError('Obstacle slot capacity exceeded')
    result=list(obstacles)
    for i in range(count-len(result)):
        result.append(dict(center=np.array([1000.+i,1000.]),a=.1,b=.1,theta=0.,name='inactive',guidance=False))
    return result
