#!/usr/bin/env python3
"""ROS-free adapters for the validated MPC: timed reference and observed ellipses."""
from collections import deque
import numpy as np
from mpc_dcbf_static_obstacles import reference_covariance_matrices, reference_axis_inflation


METRICS_SCHEMA_VERSION=2


def metric_definitions():
    return dict(version=METRICS_SCHEMA_VERSION,frame='world',sample_time='input odometry stamp',
        position_error='measured position - leader reference at the same stamp - nominal offset',
        heading_error='wrap(leader heading - measured heading), radians',
        plan_execution_error='previous accepted auxiliary plan at the measured stamp - measured position',
        plan_reference='integral of accepted plan_controls; never the newly anchored plan node zero',
        observer_error='control observer reference aligned to the measured stamp - leader reference',
        reference_quality={0:'unavailable',1:'Hermite interpolation',2:'bounded constant-velocity extrapolation',3:'exact packet node'},
        legacy_reference_fields='reference_x/y and RobotHistoryData.xr/yr remain nominal compatibility fields',
        missing='NaN with validity flag; never a substituted zero')


def wrap_angle(angle):
    return float(np.arctan2(np.sin(angle),np.cos(angle)))


class LeaderMetricTimeline:
    """Pair unchanged leader packets with clocks; packets describe t+dt.

    This is evaluation only. Neither controller references nor observer updates
    use this timeline. Exact floating logical stamps pair both ROS messages.
    """
    def __init__(self,dt=.1,capacity=200):
        self.dt,self.capacity=dt,capacity
        self.states={};self.clocks={};self.samples=[]

    def reset(self):
        self.states.clear();self.clocks.clear();self.samples.clear()

    def observe_clock(self,logical,stamp):
        if self.samples and (stamp+self.dt<self.samples[-1][0]-.5 or logical+self.dt<self.samples[-1][1]-.5):self.reset()
        self.clocks[float(logical)]=float(stamp);self._pair(float(logical))

    def observe_state(self,logical,position,velocity,heading):
        if self.samples and logical+self.dt<self.samples[-1][1]-.5:self.reset()
        self.states[float(logical)]=(np.asarray(position,float).copy(),np.asarray(velocity,float).copy(),float(heading))
        self._pair(float(logical))

    def _pair(self,key):
        if key in self.states and key in self.clocks:
            stamp=self.clocks[key]+self.dt
            row=(stamp,key+self.dt,*self.states[key])
            self.samples=[s for s in self.samples if abs(s[0]-stamp)>1e-9]
            self.samples.append(row);self.samples.sort(key=lambda s:s[0]);self.samples=self.samples[-self.capacity:]
        for cache in (self.states,self.clocks):
            while len(cache)>self.capacity:del cache[min(cache)]

    def evaluate(self,stamp):
        empty=dict(position=np.full(2,np.nan),velocity=np.full(2,np.nan),heading=np.nan,logical_time=np.nan,quality=0)
        if not self.samples or not np.isfinite(stamp):return empty
        times=np.array([s[0] for s in self.samples]);index=int(np.searchsorted(times,stamp))
        if index<len(times) and abs(times[index]-stamp)<1e-9:
            _,logical,pos,vel,heading=self.samples[index]
            return dict(position=pos.copy(),velocity=vel.copy(),heading=heading,logical_time=logical,quality=3)
        if 0<index<len(times):
            left,right=self.samples[index-1],self.samples[index];h=right[0]-left[0]
            if h>2.5*self.dt:return empty
            s=(stamp-left[0])/h;p0,v0=left[2:4];p1,v1=right[2:4]
            pos=(2*s**3-3*s*s+1)*p0+(s**3-2*s*s+s)*h*v0+(-2*s**3+3*s*s)*p1+(s**3-s*s)*h*v1
            vel=(6*s*s-6*s)*p0/h+(3*s*s-4*s+1)*v0+(-6*s*s+6*s)*p1/h+(3*s*s-2*s)*v1
            heading=float(np.arctan2(vel[1],vel[0])) if np.linalg.norm(vel)>1e-8 else wrap_angle(left[4]+s*wrap_angle(right[4]-left[4]))
            return dict(position=pos,velocity=vel,heading=heading,logical_time=left[1]+s*(right[1]-left[1]),quality=1)
        sample=self.samples[0 if index==0 else -1];delta=stamp-sample[0]
        if abs(delta)>1.5*self.dt+1e-9:return empty
        if sample[1]+delta<0:return empty  # No prescribed trajectory before its start.
        return dict(position=sample[2]+delta*sample[3],velocity=sample[3].copy(),heading=sample[4],
                    logical_time=sample[1]+delta,quality=2)


def timeline_from_leader_history(rows,dt=.1):
    """Decode the unchanged leader's history: row i stores packet i-1 values.

    leader_node publishes its previous self.x0 before assigning the just-computed
    t+dt state. Hence history row i belongs to row i-1's stamp plus dt, not
    automatically row i's publication stamp. Skip the initializer row.
    """
    timeline=LeaderMetricTimeline(dt,capacity=max(200,len(rows)+1))
    for previous,current in zip(rows,rows[1:]):
        logical=float(previous['trajectory_time']);stamp=float(previous['ros_time'])
        gap=float(current['trajectory_time'])-logical
        if not .5*dt<=gap<=1.5*dt:continue  # A lost row makes the predecessor ambiguous.
        timeline.observe_clock(logical,stamp)
        timeline.observe_state(logical,[float(current['x0']),float(current['y0'])],
            [float(current['u0x']),float(current['u0y'])],float(current['theta0']))
    return timeline


def accepted_plan_reference(origin,position,plan_controls,dt,issued_at,cycle):
    controls=np.asarray(plan_controls,float)
    if controls.ndim!=2 or controls.shape[1]!=2 or not np.isfinite(controls).all():raise ValueError('Invalid auxiliary plan')
    positions=np.vstack((position,np.asarray(position)+dt*np.cumsum(controls,axis=0)))
    return dict(origin=float(origin),issued_at=float(issued_at),dt=float(dt),positions=positions,cycle=int(cycle))


def sample_plan_reference(plan,stamp):
    empty=dict(position=np.full(2,np.nan),valid=0,age=np.nan,cycle=-1)
    if plan is None:return empty
    age=stamp-plan['origin'];dt=plan['dt']
    if stamp<plan['issued_at']-1e-9 or age<=1e-8 or age>min(1.5*dt,(len(plan['positions'])-1)*dt)+1e-9:return empty
    times=plan['origin']+dt*np.arange(len(plan['positions']))
    position=np.array([np.interp(stamp,times,plan['positions'][:,i]) for i in (0,1)])
    return dict(position=position,valid=1,age=float(age),cycle=plan['cycle'])


def default_config():
    return dict(horizon=10, dt=.1, margin=.12, gamma=.3, omega_max=2.84,
        plan_v_max=.22, delta_limit=False, delta_v=.06, delta_w=.5,
        q_state=[35.,35.,.6], q_terminal=[80.,80.,1.], q_plan=[20.,20.],
        q_coupling=[30.,30.], r_control=[.15,.04], r_plan=[.04,.04],
        r_delta=[.05,.16], r_delta_plan=[.03,.03], max_iter=120,
        fallback_kv=1.8, fallback_kw=3., continuous_guidance=True,
        guide_velocity_weight=600., guide_heading_weight=2., guide_delta_scale=8.,
        prediction_risk=False,risk_weight=3.,risk_clearance=.35,risk_discount=.5,risk_tail_seconds=0.,
        risk_external_only=True,suppress_guidance=False,constraint_mode='dcbf',
        risk_penalty='clearance_quartic',wang_alpha=1.,wang_mu=.1,wang_iota=.02,wang_radius=1.)


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
