#!/usr/bin/env python3
"""Parameterised MPC with unchanged states/cost/CBFs and selective multistart.

No ROS or simulation globals. Four static-obstacle uncertainty envelopes are
inputs, not refitted here. The nominal target and hard safety constraints are
unchanged. Optional tangential guidance adds soft velocity/heading costs.
"""
import time
from collections import deque,namedtuple
import math

import casadi as ca
import numpy as np

from mpc_dcbf_static_obstacles import obstacle_axes


def _clip_scalar(value,lower,upper):
    return value if math.isnan(value) else min(upper,max(lower,value))


class PreviewBudgetExceeded(RuntimeError):
    pass


def _check_preview_budget(config):
    limit=config.get('_preview_deadline')
    if limit is not None and time.perf_counter()>=limit:raise PreviewBudgetExceeded('Reference preview budget exhausted')


def rollout(state, controls, dt):
    """Unwrapped prediction headings satisfy the NLP dynamics exactly."""
    controls = np.asarray(controls, dtype=float)
    states = np.empty((len(controls)+1, 3))
    states[0] = state
    for j, (v, w) in enumerate(controls):
        states[j+1, 2] = states[j, 2]+dt*w
        states[j+1, :2] = states[j, :2]+dt*v*np.array([
            np.cos(states[j+1, 2]), np.sin(states[j+1, 2])])
    return states


def obstacle_angles(obstacle, count):
    """Static angles remain constant; robot envelopes may rotate at each node."""
    if 'prediction_angles' not in obstacle:
        return np.full(count, obstacle['theta'])
    angles = np.asarray(obstacle['prediction_angles'], dtype=float)
    if angles.ndim != 1 or len(angles) < count or not np.isfinite(angles[:count]).all():
        raise ValueError('Expected one finite obstacle angle per prediction node')
    return angles[:count]


BarrierBatch=namedtuple('BarrierBatch','centers axes cosine sine gamma')


def obstacle_batch(obstacles,count,gamma=.3):
    """Pack one immutable snapshot; never retain a cache across sensor updates."""
    centers=[];axes=[];angles=[]
    for o in obstacles:
        c=np.asarray(o.get('prediction_centers',o['center']),float)
        a=np.asarray(o.get('prediction_axes',[o['a'],o['b']]),float)
        if c.ndim==1:c=np.broadcast_to(c,(count,2))
        if a.ndim==1:a=np.broadcast_to(a,(count,2))
        if len(c)<count or len(a)<count:raise IndexError('Obstacle prediction stage outside the MPC window')
        centers.append(c[:count]);axes.append(a[:count]);angles.append(obstacle_angles(o,count))
    angles=np.asarray(angles).reshape(len(obstacles),count)
    return BarrierBatch(np.asarray(centers).reshape(len(obstacles),count,2),
        np.asarray(axes).reshape(len(obstacles),count,2),np.cos(angles),np.sin(angles),
        np.array([o.get('gamma',gamma) for o in obstacles]))


def _batch_values(positions,batch,margin,start=0):
    stop=start+len(positions);delta=np.asarray(positions)[None,:,:]-batch.centers[:,start:stop]
    c,s=batch.cosine[:,start:stop],batch.sine[:,start:stop]
    local=np.stack((c*delta[:,:,0]+s*delta[:,:,1],-s*delta[:,:,0]+c*delta[:,:,1]),axis=-1)
    axes=batch.axes[:,start:stop]
    return axes[:,:,1]*(np.sqrt(np.sum((local/axes)**2,axis=-1)+1e-12)-1)-margin


def barrier_values(states, obstacles, margin):
    """Same stage-wise ellipse formula, broadcast over obstacles and nodes."""
    batch=obstacle_batch(obstacles,len(states))
    return _batch_values(np.asarray(states)[:,:2],batch,margin)


def first_step_values(state,commands,batch,margin,dt,stage=0):
    angle=state[2]+dt*commands[:,1]
    positions=state[:2]+dt*commands[:,0,None]*np.column_stack((np.cos(angle),np.sin(angle)))
    h0=_batch_values(np.asarray(state)[None,:2],batch,margin,stage)[:,0]
    j=stage+1;delta=positions[None,:,:]-batch.centers[:,j,None,:]
    c,s=batch.cosine[:,j,None],batch.sine[:,j,None]
    local=np.stack((c*delta[:,:,0]+s*delta[:,:,1],-s*delta[:,:,0]+c*delta[:,:,1]),axis=-1)
    axes=batch.axes[:,j,None,:]
    h1=axes[:,:,1]*(np.sqrt(np.sum((local/axes)**2,axis=-1)+1e-12)-1)-margin
    return h0,h1,positions,angle


def first_step_mask(state, commands, obstacles, bounds, previous, config,batch=None,stage=0):
    """Vectorized screening of the SAME first-step bounds and D-CBF tests.

    Screening is followed by CachedMPCProblem.validate before any command can
    be returned. This only makes the finite fallback search cheaper/denser.
    """
    tol=2e-6;u=np.asarray(commands,float);dt=config['dt']
    angle=state[2]+dt*u[:,1]
    positions=state[:2]+dt*u[:,0,None]*np.column_stack((np.cos(angle),np.sin(angle)))
    valid=(np.isfinite(u).all(axis=1)&(u[:,0]>=bounds[0]-tol)&(u[:,0]<=bounds[1]+tol)
           &(np.abs(u[:,1])<=config['omega_max']+tol))
    if config['delta_limit'] and previous is not None:
        valid&=np.all(np.abs(u-previous)<=np.array([config['delta_v'],config['delta_w']])+tol,axis=1)
    if obstacles:
        if batch is None:batch=obstacle_batch(obstacles,stage+2,config['gamma'])
        h0,h1,positions,angle=first_step_values(state,u,batch,config['margin'],dt,stage)
        initially_safe=h0.min()>=-tol
        valid&=np.all(h1-(1-batch.gamma[:,None])*h0[:,None]>=-tol,axis=0)
        if initially_safe:valid&=np.all(h1>=-tol,axis=0)
    return valid,positions,angle


def adapt_negative_recovery(state,obstacles,bounds,previous,config):
    """Construct a grid witness for a slower recovery ONLY when h0<0.

    Positive barriers keep their original gamma. A reduced gamma remains in
    [0,base], so negative h is required not to decrease in the frozen model.
    No grid witness leaves the original problem unchanged; not a claim of
    continuous infeasibility, recursive feasibility or physical safety.
    """
    cfg=config;tol=2e-6;packed=obstacle_batch(obstacles,2,cfg['gamma'])
    h0=_batch_values(np.asarray(state)[None,:2],packed,cfg['margin'])[:,0]
    negative=(h0<0)&(packed.gamma>0)
    info=dict(adapted=False,scale=1.,negative_barriers=int(negative.sum()),witness=None)
    if not negative.any():return list(obstacles),info
    v,w=np.meshgrid(np.linspace(*bounds,33),np.linspace(-cfg['omega_max'],cfg['omega_max'],33))
    choices=np.vstack((np.zeros(2),np.zeros(2) if previous is None else previous,np.column_stack((v.ravel(),w.ravel()))))
    valid,_,_=first_step_mask(state,choices,[],bounds,previous,cfg)
    _,h1,_,_=first_step_values(state,choices,packed,cfg['margin'],cfg['dt'])
    base_valid=valid&np.all(h1-(1-packed.gamma[:,None])*h0[:,None]>=-tol,axis=0)
    if h0.min()>=-tol:base_valid&=np.all(h1>=-tol,axis=0)
    if base_valid.any():return list(obstacles),info
    safe=~negative
    valid&=np.all(h1[safe]-(1-packed.gamma[safe,None])*h0[safe,None]>=-tol,axis=0)
    capacity=np.min((h1[negative]-h0[negative,None]-tol)/(-packed.gamma[negative,None]*h0[negative,None]),axis=0)
    capacity=np.where(valid,capacity,-np.inf);index=int(np.argmax(capacity));scale=float(min(1.,capacity[index]))
    if not np.isfinite(scale) or scale<0.:
        info['reason']='no_monotone_grid_recovery';return list(obstacles),info
    if scale>=1.:return list(obstacles),info
    gammas=packed.gamma.copy();gammas[negative]*=scale
    candidate=[dict(o,gamma=float(g)) for o,g in zip(obstacles,gammas)]
    mask,_,_=first_step_mask(state,choices[index:index+1],candidate,bounds,previous,cfg)
    if not mask[0]:
        info['reason']='witness_recheck_failed';return list(obstacles),info
    info.update(adapted=True,scale=scale,witness=choices[index].tolist(),
        barrier_names=[o.get('name',str(i)) for i,o in enumerate(obstacles)],
        base_gammas=packed.gamma.tolist(),applied_gammas=gammas.tolist(),
        initial_h=h0.tolist(),witness_next_h=h1[:,index].tolist())
    return candidate,info


def project_relative_velocity(velocity,frame_velocity,limit=.22):
    """Project onto ||v_relative + v_frame|| <= limit in velocity space."""
    world=np.asarray(velocity)+np.asarray(frame_velocity)
    return world*min(1.,limit/max(np.linalg.norm(world),1e-12))-frame_velocity


def _stationary_guidance(state, goals, obstacles, config, memory):
    """Smooth, side-persistent field used only as a soft MPC objective.

    A short geometric lookahead detects an obstructed approach before the
    1-second MPC window arrives at the barrier. It is not a replacement path,
    dynamics model, collision test, or applied control.
    """
    n, dt = config['horizon'], config['dt']
    zero = np.zeros((n, 2))
    if not config.get('continuous_guidance', False) or not obstacles:
        return zero, 0.
    position = state[:2]
    feed = (goals[1, :2]-goals[0, :2])/dt
    aim = goals[-1, :2]+2.0*feed
    frame=config.get('_guide_frame_velocities')

    def bounded(vector,stage):
        if frame is not None:return project_relative_velocity(vector,frame[stage])
        return vector*min(1.,.22/max(np.linalg.norm(vector),1e-9))

    def geometry(point, obs, stage):
        a, b = obstacle_axes(obs, stage)
        c, s = np.cos(obs['theta']), np.sin(obs['theta'])
        rotation = np.array([[c, -s], [s, c]])
        local = (point-obs['center'])@rotation
        radius = max(np.linalg.norm(local/[a, b]), 1e-9)
        normal = rotation@(local/np.array([a*a, b*b]))
        normal /= max(np.linalg.norm(normal), 1e-9)
        return b*(radius-1)-config['margin'], normal

    def blocked(obs):
        a, b = obstacle_axes(obs, n)
        scale = 1+(config['margin']+.04)/b
        c, s = np.cos(obs['theta']), np.sin(obs['theta'])
        rotation = np.array([[c, -s], [s, c]])
        start = (position-obs['center'])@rotation/(scale*np.array([a, b]))
        end = (aim-obs['center'])@rotation/(scale*np.array([a, b]))
        line = end-start
        fraction = np.clip(-np.dot(start, line)/max(np.dot(line, line), 1e-12), 0, 1)
        # Being inside the extra guidance buffer is not a reason to keep
        # circling when the nominal direction already points safely outward.
        return (np.linalg.norm(start+fraction*line) < 1.0
                and (fraction > 1e-6 or np.dot(start, line) < 0))

    def route_geometry(obs):
        a, b = obstacle_axes(obs, n)
        scale = 1+(config['margin']+.04)/b
        c, s = np.cos(obs['theta']), np.sin(obs['theta'])
        return scale*np.array([a, b]), np.array([[c, -s], [s, c]])

    def project(point, obs, axes, rotation):
        local = (point-obs['center'])@rotation
        radius = np.linalg.norm(local/axes)
        if radius >= 1.:
            return point.copy(), False
        if radius < 1e-8:
            local = (position-obs['center'])@rotation
            radius = max(np.linalg.norm(local/axes), 1e-8)
        return obs['center']+(local/radius)@rotation.T, True

    def visible(point, target, obs, axes, rotation):
        start = (point-obs['center'])@rotation/axes
        end = (target-obs['center'])@rotation/axes
        line = end-start
        fraction = np.clip(-np.dot(start, line)/max(np.dot(line, line), 1e-12), 0., 1.)
        # An outward segment from inside the *soft* guide buffer is allowed.
        # The actual safety envelope is still enforced separately by the NLP.
        if fraction < 1e-6 and np.dot(start, line) >= -1e-8:
            return True
        return np.linalg.norm(start+fraction*line) >= 1.-1e-6

    def routes(point, target, obs, axes, rotation):
        start = (point-obs['center'])@rotation/axes
        end = (target-obs['center'])@rotation/axes
        rs, rt = np.linalg.norm(start), np.linalg.norm(end)
        ps, pt = np.arctan2(start[1], start[0]), np.arctan2(end[1], end[0])
        start_offset = np.arccos(1/max(rs, 1.))
        end_offset = np.arccos(1/max(rt, 1.))
        answer = {}
        for side in (1, -1):
            begin = ps+side*start_offset
            finish = pt-side*end_offset
            sweep = float((side*(finish-begin))%(2*np.pi))
            # Tangent segments + true ellipse arc (sampled), in metres.
            angle = begin+side*np.linspace(0., sweep, 25)
            arc = np.column_stack((axes[0]*np.cos(angle), axes[1]*np.sin(angle)))@rotation.T+obs['center']
            length = np.linalg.norm(arc[0]-point)+np.linalg.norm(arc[-1]-target)
            length += np.linalg.norm(np.diff(arc, axis=0), axis=1).sum()
            answer[side] = (float(length), sweep)
        return answer

    index = memory.get('guide_index')
    if index is not None and index >= len(obstacles):
        index = None
    if index is not None and not obstacles[index].get('guidance', True):
        index = None
    eligible = []
    for j, obs in enumerate(obstacles):
        if not obs.get('guidance', True):
            continue
        h, normal = geometry(position, obs, 0)
        if h < .4 and blocked(obs):
            eligible.append((h, j))
    if index is None or memory.get('guide_level', 0.) < .025:
        choice = min(eligible)[1] if eligible else None
        if choice != index and choice is not None:
            _, normal = geometry(position, obstacles[choice], 0)
            tangent = np.array([-normal[1], normal[0]])
            attraction = feed+1.2*(goals[0, :2]-position)
            score = np.dot(tangent, attraction)
            if abs(score) < 1e-6:
                score = np.dot(tangent, [np.cos(state[2]), np.sin(state[2])])
            memory['guide_side'] = 1 if score >= 0 else -1
            memory['route_age'] = 0.
            memory.pop('guide_vector', None)
        index = choice
    target_level = 0.
    projections, projected_inside, route_data = None, None, None
    direct = True
    if index is not None:
        h, normal = geometry(position, obstacles[index], 0)
        obs = obstacles[index]
        axes, rotation = route_geometry(obs)
        projected = [project(g[:2], obs, axes, rotation) for g in goals]
        projections = np.array([item[0] for item in projected])
        projected_inside = any(item[1] for item in projected)
        target = projections[-1]
        direct = visible(position, target, obs, axes, rotation)
        route_data = routes(position, target, obs, axes, rotation)
        side = memory.get('guide_side', 1)
        alternative = -side
        memory['route_age'] = memory.get('route_age', 0.)+dt
        if not direct:
            preferred = min(route_data, key=lambda s: route_data[s][0])
            advantage = route_data[side][0]-route_data[alternative][0]
            threshold = max(.12, .15*route_data[side][0])
            if (memory.get('guide_level', 0.) < .025
                    or (memory['route_age'] >= .8 and preferred != side and advantage > threshold)):
                if preferred != side:
                    memory['route_switches'] = memory.get('route_switches', 0)+1
                memory['guide_side'] = preferred
                memory['route_age'] = 0.
        memory['route_cost_ccw'], memory['route_cost_cw'] = route_data[1][0], route_data[-1][0]
        memory['projected_follow'] = bool(projected_inside)
        if blocked(obstacles[index]):
            target_level = float(np.clip((.4-h)/.25, 0., 1.))
            if direct and not projected_inside:
                target_level = min(target_level, .3)
        # An infeasible nominal point needs a feasible local velocity reference,
        # even if the projected point can be approached without an obstacle arc.
        if projected_inside or not direct:
            target_level = max(target_level, float(np.clip((.4-h)/.25, 0., 1.)))
    alpha = dt/(.4+dt)
    level = memory.get('guide_level', 0.)+alpha*(target_level-memory.get('guide_level', 0.))
    memory['guide_level'], memory['guide_index'] = level, index
    if index is None or level < 1e-5:
        memory.pop('guide_vector', None)
        return zero, 0.
    guide = np.zeros((n, 2))
    point = position.copy()
    previous_vector = memory.get('guide_vector', feed.copy())
    previous_vector = bounded(previous_vector,0)
    for j in range(n):
        h, normal = geometry(point, obstacles[index], j)
        tangent = memory.get('guide_side', 1)*np.array([-normal[1], normal[0]])
        attraction = feed+1.2*(goals[j, :2]-point)
        attraction = bounded(attraction,j)
        local_feed = (projections[j+1]-projections[j])/dt
        if visible(point, projections[j+1], obs, axes, rotation):
            flow = local_feed+1.2*(projections[j+1]-point)
        else:
            side = memory.get('guide_side', 1)
            route = routes(point, projections[-1], obs, axes, rotation)[side]
            arc_progress = min(route[0], max(axes)*route[1])
            maximum=.20+(np.linalg.norm(frame[j]) if frame is not None else 0.)
            speed = np.clip(.9*arc_progress+max(np.dot(local_feed, tangent), 0.), 0., maximum)
            radial = np.clip(.8*(.06-h), -.08, .12)
            flow = speed*tangent+radial*normal
        desired = (1-level)*attraction+level*flow
        desired = bounded(desired,j)
        desired = previous_vector+.25*(desired-previous_vector)
        if frame is not None:desired=bounded(desired,j)
        previous_vector = desired
        guide[j] = desired
        point += dt*desired
    memory['guide_vector'] = guide[0].copy()
    return guide, float(level)


def _legacy_continuous_guidance(state,goals,obstacles,config,memory):
    """Use the existing soft bypass field in a predicted moving frame.

    Predicted obstacle displacement is removed from nominal goal positions,
    then added back to the desired velocity. The NLP still uses the original
    world-frame dynamics, future ellipses, speed bounds and D-CBF constraints.
    This breaks symmetric following/retreat equilibria; it is not a safety
    certificate or a substitute for nonlinear validation.
    """
    stationary,level=_stationary_guidance(state,goals,obstacles,config,memory)
    if not config.get('continuous_guidance',False):return stationary,level
    candidates=[]
    for obs in obstacles:
        if not obs.get('dynamic',False) or obs.get('peer',False):continue
        a,b=obstacle_axes(obs,0);c,s=np.cos(obs['theta']),np.sin(obs['theta'])
        local=(state[:2]-obs['center'])@np.array([[c,-s],[s,c]])
        h=b*(np.linalg.norm(local/[a,b])-1)-config['margin']
        if h<.5:candidates.append((h,obs['name'],obs))
    if not candidates:return stationary,level
    _,name,obs=min(candidates,key=lambda item:item[:2])
    if memory.get('moving_guide_name')!=name:
        memory['moving_guide_name']=name;memory['moving_guide_memory']={}
    relative_goals=np.array(goals,copy=True)
    centers=np.asarray(obs['prediction_centers'])
    relative_goals[:,:2]-=centers[:len(goals)]-centers[0]
    virtual=dict(obs,guidance=True,dynamic=False)
    velocities=np.diff(centers[:len(goals)],axis=0)/config['dt']
    shifted=bool(config.get('moving_frame_projection',False))
    moving_config=dict(config,_guide_frame_velocities=velocities) if shifted else config
    guide,moving_level=_stationary_guidance(state,relative_goals,[virtual],moving_config,memory['moving_guide_memory'])
    if moving_level<=level:return stationary,level
    velocity=(velocities
              if shifted or obs.get('prediction_model') in ('ca','singer','adaptive') and obs.get('stage_velocity_guidance',True)
              else (centers[1]-centers[0])/config['dt'])
    guide+=velocity
    speed=np.linalg.norm(guide,axis=1)
    guide*=np.minimum(1.,.22/np.maximum(speed,1e-12))[:,None]
    return guide,moving_level


def progress_centers(obstacle,times,dt):
    """Existing predictor means at longer times, for soft decisions ONLY."""
    times=np.asarray(times);motion=obstacle.get('guidance_motion')
    if motion is not None:
        result=np.zeros((len(times),2))
        for weight,model in zip(motion['weights'],motion['models']):
            p=np.asarray(model['position']);v=np.asarray(model['velocity']);a=np.asarray(model['acceleration'])
            if model['mode']=='frozen':value=np.tile(p,(len(times),1))
            else:
                value=p+times[:,None]*v
                if model['mode']=='ca':value=value+.5*times[:,None]**2*a
                elif model['mode']=='singer':
                    tau=model['tau'];value=value+tau*tau*(times/tau+np.expm1(-times/tau))[:,None]*a
            result+=weight*value
        return result
    centers=np.asarray(obstacle.get('prediction_centers',[obstacle['center']]))
    if centers.ndim==1:return np.tile(centers,(len(times),1))
    nodes=np.arange(len(centers))*dt
    result=np.column_stack([np.interp(times,nodes,centers[:,j]) for j in (0,1)])
    if len(centers)>1:
        velocity=(centers[-1]-centers[-2])/dt
        result+=np.maximum(0.,times-nodes[-1])[:,None]*velocity
    return result


def progress_obstacles(obstacles,times,config):
    """Keep the original safety-window nodes; hold axes beyond that window.

    Long means / held axes are a decision approximation, not new hard CBFs.
    """
    result=[];n=config['horizon'];dt=config['dt']
    for obs in obstacles:
        item=dict(obs);item['prediction_centers']=progress_centers(obs,times,dt)
        item['prediction_axes']=np.array([obstacle_axes(obs,min(int(round(t/dt)),n)) for t in times])
        angles=np.asarray(obs.get('prediction_angles',[obs['theta']]))
        item['prediction_angles']=angles[np.minimum(np.rint(times/dt).astype(int),len(angles)-1)]
        result.append(item)
    return result


def progress_targets(goals,times,dt):
    nodes=np.arange(len(goals))*dt
    result=np.column_stack([np.interp(times,nodes,goals[:,j]) for j in (0,1)])
    result+=np.maximum(0.,np.asarray(times)-nodes[-1])[:,None]*(goals[-1,:2]-goals[-2,:2])/dt
    return result


def relative_obstruction(position,aim,centers,axes,angles,times,margin,speed_limit):
    """Sample relative approach, including objects closing from behind.

    This straight robot ray / held long axes is a soft trigger, not an
    additional hard prediction horizon or a collision-free trajectory proof.
    """
    velocity=(np.asarray(aim)-position)/times[-1]
    velocity*=min(1.,speed_limit/max(np.linalg.norm(velocity),1e-12))
    robot=np.asarray(position)+times[:,None]*velocity
    delta=robot[None,:,:]-centers;c,s=np.cos(angles),np.sin(angles)
    local=np.stack((c*delta[:,:,0]+s*delta[:,:,1],-s*delta[:,:,0]+c*delta[:,:,1]),axis=-1)
    expanded=axes*(1.+(margin+.04)/axes[:,1])[:,None]
    radius=np.linalg.norm(local/expanded,axis=-1)
    future=radius[:,1:].min(axis=1)
    return (future<1.)&(future<radius[:,0]-1e-6)


def bypass_velocity(tangent,normal,h,frame_velocity=None):
    relative=.20*tangent+_clip_scalar(.8*(.06-h),-.08,.12)*normal
    return relative if frame_velocity is None else relative+frame_velocity


def progress_preview(state,goals,obstacles,selected,mode,config,previous,bounds):
    """A bounded unicycle reference primitive; never an applied command."""
    _check_preview_budget(config)
    dt=config['dt'];steps=max(config['horizon'],int(round(config.get('progress_horizon',4.)/dt)))
    times=np.arange(steps+1)*dt;targets=progress_targets(goals,times,dt)
    feeds=np.diff(targets,axis=0)/dt;feed=feeds[0]
    forecast=progress_obstacles(obstacles,times,config);obs=forecast[selected]
    centers=obs['prediction_centers'];axes=obs['prediction_axes'];angles=obs['prediction_angles']
    motion=obstacles[selected].get('guidance_motion',{})
    hypothesis_centers=[centers]
    if config.get('progress_hypotheses',True):
        for model in motion.get('models',[]):
            alternative=dict(obstacles[selected],guidance_motion=dict(stamp=motion['stamp'],weights=[1.],models=[model]))
            hypothesis_centers.append(progress_centers(alternative,times,dt))
    hypothesis_centers=np.asarray(hypothesis_centers)
    predictive=config.get('predictive_obstruction',False)
    if predictive:
        ahead=max(1,int(round(config.get('obstruction_horizon',2.)/dt)))
        ahead_times=np.arange(ahead+1)*dt;extended=np.arange(steps+ahead+1)*dt
        look_centers=[progress_centers(obstacles[selected],extended,dt)]
        if config.get('progress_hypotheses',True):
            for model in motion.get('models',[]):
                alternative=dict(obstacles[selected],guidance_motion=dict(stamp=motion['stamp'],weights=[1.],models=[model]))
                look_centers.append(progress_centers(alternative,extended,dt))
        look_centers=np.asarray(look_centers);held=np.minimum(np.arange(steps+ahead+1),len(axes)-1)
        look_axes=axes[held];look_angles=angles[held]
    direction=feed/max(np.linalg.norm(feed),1e-12);normal_line=np.array([-direction[1],direction[0]])
    release=float('inf')
    if mode=='yield':
        future=np.arange(int(round(config.get('progress_wait_horizon',6.)/dt))+1)*dt
        c=progress_centers(obstacles[selected],future,dt)
        a,b=obstacle_axes(obstacles[selected],config['horizon']);theta=obstacles[selected]['theta']
        radius=np.hypot(a*np.dot([np.cos(theta),np.sin(theta)],normal_line),
                        b*np.dot([-np.sin(theta),np.cos(theta)],normal_line))+config['margin']+.04
        lateral=(c-goals[0,:2])@normal_line
        blocked=np.flatnonzero(np.abs(lateral)<=radius)
        j=int(blocked[-1])+1 if len(blocked) else 0
        if j<len(future) and future[-1]-future[j]>=.5:release=float(future[j])
    states=np.empty((steps+1,3));states[0]=state;controls=np.zeros((steps,2));vectors=np.zeros((steps,2))
    velocity=previous[0]*np.array([np.cos(state[2]),np.sin(state[2])]);last=np.array(previous,copy=True)
    local_filter=config.get('progress_local',False) and mode in ('ccw','cw')
    aligned_hypotheses=[]
    if local_filter:
        packed=obstacle_batch(forecast,steps+1,config['gamma'])
        if config.get('align_reference_hypotheses',False) and config.get('progress_hypotheses',True):
            for i,original in enumerate(obstacles):
                if original.get('peer',False):continue
                motion=original.get('guidance_motion') or {}
                for model in motion.get('models',[]):
                    alternative=dict(original,guidance_motion=dict(stamp=motion['stamp'],weights=[1.],models=[model]))
                    aligned_hypotheses.append(dict(forecast[i],prediction_centers=progress_centers(alternative,times,dt)))
        hypothesis_batch=obstacle_batch(aligned_hypotheses,steps+1) if aligned_hypotheses else None
        v,w=np.meshgrid(np.linspace(*bounds,17),np.linspace(-config['omega_max'],config['omega_max'],17))
        choices=np.empty((3+v.size,2));choices[3:]=np.column_stack((v.ravel(),w.ravel()))
    vmax=max(abs(bounds[0]),abs(bounds[1]));omega_max=config['omega_max']
    circulation_active=False
    for j in range(steps):
        _check_preview_budget(config)
        p=states[j,:2];theta=states[j,2]
        desired=feeds[j]+1.2*(targets[j]-p)
        if mode=='yield' and times[j]<release:desired=np.zeros(2)
        if mode in ('ccw','cw'):
            c,s=np.cos(angles[j]),np.sin(angles[j]);rotation=np.array([[c,-s],[s,c]])
            local=(p-hypothesis_centers[:,j,:])@rotation;a,b=axes[j]
            h=b*(np.linalg.norm(local/[a,b],axis=1)-1)-config['margin']
            aim=targets[j]+2.*feeds[j]
            scale=1.+(config['margin']+.04)/b
            start=local/(scale*np.array([a,b]));end=(aim-hypothesis_centers[:,j,:])@rotation/(scale*np.array([a,b]))
            line=end-start;dot=np.sum(start*line,axis=1)
            fraction=np.clip(-dot/np.maximum(np.sum(line*line,axis=1),1e-12),0.,1.)
            blocked=(np.linalg.norm(start+fraction[:,None]*line,axis=1)<1.)&((fraction>1e-6)|(dot<0))
            if predictive:
                blocked=relative_obstruction(p,aim,look_centers[:,j:j+ahead+1],look_axes[j:j+ahead+1],
                    look_angles[j:j+ahead+1],ahead_times,config['margin'],vmax)
            if blocked.any():
                k=int(np.argmin(np.where(blocked,h,np.inf)))
                normal=rotation@(local[k]/np.array([a*a,b*b]));normal/=max(np.linalg.norm(normal),1e-12)
                tangent=(1 if mode=='ccw' else -1)*np.array([-normal[1],normal[0]])
                frame_velocity=((hypothesis_centers[k,j+1]-hypothesis_centers[k,j])/dt
                    if config.get('relative_avoidance_flow',False) else None)
                flow=bypass_velocity(tangent,normal,h[k],frame_velocity)
                level=_clip_scalar((.4-h[k])/.25,0.,1.);desired=(1-level)*desired+level*flow
                if j<config['horizon'] and level>1e-6:circulation_active=True
        desired*=min(1.,vmax/max(np.linalg.norm(desired),1e-12))
        velocity+=.25*(desired-velocity)
        if np.linalg.norm(velocity)<1e-8:angle=theta;sign=1.
        else:
            angle=np.arctan2(velocity[1],velocity[0]);delta=np.arctan2(np.sin(angle-theta),np.cos(angle-theta))
            sign=-1. if abs(delta)>np.pi/2 and bounds[0]<0 else 1.
            if sign<0:angle+=np.pi
        error=np.arctan2(np.sin(angle-theta),np.cos(angle-theta))
        omega=_clip_scalar(error/dt,-omega_max,omega_max)
        if config['delta_limit']:omega=_clip_scalar(omega,last[1]-config['delta_w'],last[1]+config['delta_w'])
        new_theta=theta+dt*omega;heading=np.array([np.cos(new_theta),np.sin(new_theta)])
        speed=_clip_scalar(np.dot(velocity,heading),bounds[0],bounds[1])
        if config['delta_limit']:speed=_clip_scalar(speed,last[0]-config['delta_v'],last[0]+config['delta_v'])
        if local_filter:
            # Receding-window escape only: filter the primitive against the
            # original next-node DCBF, without changing its gamma or envelope.
            choices[0]=[speed,omega];choices[1]=last;choices[2]=0.
            valid,_,angles_next=first_step_mask(states[j],choices,forecast,bounds,last,config,batch=packed,stage=j)
            if hypothesis_batch is not None:
                # Match the existing post-check h>=0 for each retained mean.
                # This does not add hypothetical-model DCBFs to the MPC.
                _,next_h,_,_=first_step_values(states[j],choices,hypothesis_batch,config['margin'],dt,stage=j)
                valid&=np.all(next_h>=-2e-6,axis=0)
            if valid.any():
                xy_velocity=choices[:,0,None]*np.column_stack((np.cos(angles_next),np.sin(angles_next)))
                score=(np.sum((xy_velocity-velocity)**2,axis=1)+.002*(choices[:,1]-last[1])**2
                    +.01*(1-np.cos(angles_next-(theta+dt*omega))))
                speed,omega=choices[int(np.argmin(np.where(valid,score,np.inf)))]
                new_theta=theta+dt*omega;heading=np.array([np.cos(new_theta),np.sin(new_theta)])
        controls[j]=[speed,omega];vectors[j]=speed*heading
        states[j+1,:2]=p+dt*vectors[j];states[j+1,2]=new_theta;last=controls[j]
    h=barrier_values(states,forecast,config['margin'])
    _check_preview_budget(config)
    hypothesis_h={}
    for i,original in enumerate(obstacles):
        motion=original.get('guidance_motion',{})
        if original.get('peer',False) or not config.get('progress_hypotheses',True):continue
        for k,model in enumerate(motion.get('models',[])):
            alternative=dict(original,guidance_motion=dict(stamp=motion['stamp'],weights=[1.],models=[model]))
            other=dict(forecast[i],prediction_centers=progress_centers(alternative,times,dt))
            hypothesis_h[original['name']+'/'+str(k)+'/'+model['mode']]=float(barrier_values(states,[other],config['margin']).min())
    n=config['horizon'];gamma=np.array([o.get('gamma',config['gamma']) for o in obstacles])[:,None]
    slack=h[:,1:n+1]-(1-gamma)*h[:,:n]
    feasible=bool(h.min(initial=np.inf)>=-2e-6 and slack.min(initial=np.inf)>=-2e-6)
    if config.get('progress_local',False) and mode not in ('ccw','cw'):feasible=False
    if hypothesis_h and min(hypothesis_h.values())<-2e-6:feasible=False
    if mode=='yield' and not np.isfinite(release):feasible=False
    error=states[:,:2]-targets;along=error[-1]@direction;lateral=error[-1]@normal_line
    headroom=max(max(abs(bounds[0]),abs(bounds[1]))-np.linalg.norm(feed),.02)
    recovery=max(0.,-along)/headroom+abs(lateral)/max(max(abs(bounds[0]),abs(bounds[1])),.02)
    cost=float(np.mean(np.sum(error**2,axis=1))+2*np.sum(error[-1]**2)+.03*recovery+
               .01*np.mean((controls[:,1]/config['omega_max'])**2))
    return dict(mode=mode,feasible=feasible,cost=cost,min_h=float(h.min(initial=np.inf)),
        min_slack=float(slack.min(initial=np.inf)),release_time=release if np.isfinite(release) else None,
        hypothesis_min_h=hypothesis_h,
        circulation_active=circulation_active,
        terminal_error=float(np.linalg.norm(error[-1])),estimated_recovery=float(recovery),
        states=states,controls=controls,vectors=vectors)


def progress_guidance(state,goals,obstacles,config,memory):
    """Compare waiting and both world-frame bypass sides using bounded previews."""
    memory.pop('progress_plan',None)
    original_obstacles=obstacles
    obstacles=[o for o in obstacles if o.get('name')!='inactive']
    candidates=[];feed=(goals[-1,:2]-goals[0,:2])/(config['horizon']*config['dt'])
    if np.linalg.norm(feed)<.02:return _legacy_continuous_guidance(state,goals,original_obstacles,config,memory)
    for i,obs in enumerate(obstacles):
        if obs.get('peer',False) or 'guidance_motion' not in obs:continue
        motion=obs['guidance_motion'];velocity=np.einsum('i,ij->j',motion['weights'],[m['velocity'] for m in motion['models']])
        speed=(max(np.linalg.norm(m['velocity']) for m in motion['models']) if config.get('progress_hypotheses',True)
               else np.linalg.norm(velocity))
        if speed<.02:continue
        h=barrier_values(np.asarray(state)[None,:],[obs],config['margin'])[0,0]
        if h<.65:candidates.append((h,obs['name'],i))
    if not candidates:
        memory.pop('progress_state',None)
        memory['progress_summary']=dict(mode='inactive',decision_id=memory.get('progress_counter',0))
        return _legacy_continuous_guidance(state,goals,original_obstacles,config,memory)
    _,name,index=min(candidates);obs=obstacles[index]
    now=float(obs['guidance_motion']['stamp']);previous=np.asarray(memory.get('progress_previous',[0.,0.]))
    bounds=tuple(memory.get('progress_bounds',(-.22,.22)))
    saved=memory.get('progress_state');reevaluate=(saved is None or saved['name']!=name or now-saved['decision_time']>=.5 or now<saved['decision_time'])
    if reevaluate:
        prior_decision=None if saved is None else dict(saved)
        begin=time.perf_counter();preview_config=config
        local_first=config.get('bounded_local_reference',False)
        search_complete=True
        if local_first:
            preview_config=dict(config,progress_horizon=config['horizon']*config['dt'],progress_local=True)
            incumbent=saved.get('selected') if saved is not None and saved['name']==name else None
            if config.get('active_side_lock',False) and (saved is None or not saved.get('side_committed',False)):
                incumbent=('ccw','cw')[memory.get('progress_counter',0)%2]
            order=[incumbent] if incumbent in ('ccw','cw') else []
            order+=[m for m in ('ccw','cw') if m not in order]
            previews={}
            for mode in order:
                try:previews[mode]=progress_preview(state,goals,obstacles,index,mode,preview_config,previous,bounds)
                except PreviewBudgetExceeded:search_complete=False;break
            if not previews:raise PreviewBudgetExceeded('No complete short reference within budget')
        else:
            previews={mode:progress_preview(state,goals,obstacles,index,mode,config,previous,bounds)
                for mode in ('direct','yield','ccw','cw')}
        valid=[p for p in previews.values() if p['feasible']]
        incumbent_side=(saved.get('selected') if saved is not None and saved['name']==name else None)
        needs_prefix=(not valid or incumbent_side in ('ccw','cw') and not previews.get(incumbent_side,{}).get('feasible',False))
        if needs_prefix and config.get('execution_backup',False) and not local_first:
            local_config=dict(config,progress_horizon=config['horizon']*config['dt'],progress_local=True)
            local_previews={mode:progress_preview(state,goals,obstacles,index,mode,local_config,previous,bounds)
                for mode in ('direct','yield','ccw','cw')}
            if not valid or incumbent_side in ('ccw','cw') and local_previews[incumbent_side]['feasible']:
                preview_config=local_config;previews=local_previews
                valid=[p for p in previews.values() if p['feasible']]
        selected=min(valid,key=lambda p:p['cost'])['mode'] if valid else 'legacy'
        switch_time=now
        if saved is not None and saved['name']==name:
            incumbent=previews.get(saved['selected'])
            switch_time=saved['switch_time']
            can_hold=not config.get('active_side_lock',False) or saved.get('side_committed',False)
            if can_hold and incumbent is not None and incumbent['feasible'] and selected!=saved['selected']:
                advantage=incumbent['cost']-previews[selected]['cost']
                if now-switch_time<.8 or advantage<max(.03,.15*incumbent['cost']):selected=saved['selected']
            # Keep a feasible circulation side while direct recovery is blocked.
            # A failed long extrapolation alone must not reverse a valid prefix.
            if (can_hold and config.get('execution_backup',False) and saved['selected'] in ('ccw','cw')
                    and incumbent is not None and incumbent['feasible'] and not previews.get('direct',{}).get('feasible',False)
                    and (preview_config.get('progress_local',False) or
                         incumbent['cost']<=2.*min(p['cost'] for p in valid)+.25)):
                selected=saved['selected']
            if selected!=saved['selected']:switch_time=now
        keep_commit=bool(saved is not None and saved['name']==name and saved.get('side_committed',False) and saved['selected']==selected)
        saved=dict(name=name,selected=selected,decision_time=now,switch_time=switch_time,local=preview_config.get('progress_local',False),
            decision_id=memory.get('progress_counter',0)+1,search_complete=search_complete,
            side_committed=bool(previews.get(selected,{}).get('circulation_active',False) and (search_complete or keep_commit)) if config.get('active_side_lock',False) else True)
        memory['progress_counter']=saved['decision_id'];memory['progress_state']=saved
        memory['progress_decision']=dict(saved,previous_decision=prior_decision,state=np.asarray(state).tolist(),goals=np.asarray(goals).tolist(),
            previous=previous.tolist(),bounds=list(bounds),selected_obstacle=index,
            config=dict({k:config[k] for k in ('dt','horizon','margin','gamma','omega_max','delta_limit','delta_v','delta_w')},
                        progress_horizon=preview_config.get('progress_horizon',4.),progress_wait_horizon=config.get('progress_wait_horizon',6.),
                        progress_local=preview_config.get('progress_local',False),
                        progress_side_lock=config.get('execution_backup',False),
                        progress_side_release=config.get('execution_backup',False),
                        bounded_local_reference=local_first,
                        align_reference_hypotheses=config.get('align_reference_hypotheses',False),
                        predictive_obstruction=config.get('predictive_obstruction',False),
                        obstruction_horizon=config.get('obstruction_horizon',2.),active_side_lock=config.get('active_side_lock',False),
                        relative_avoidance_flow=config.get('relative_avoidance_flow',False),
                        progress_hypotheses=config.get('progress_hypotheses',True),progress_abstain=config.get('progress_abstain',True)),
            obstacles=[dict(name=o['name'],center=np.asarray(o['center']).tolist(),a=o['a'],b=o['b'],theta=o['theta'],
                gamma=o.get('gamma',config['gamma']),prediction_centers=np.asarray(o.get('prediction_centers',np.tile(o['center'],(config['horizon']+1,1)))).tolist(),
                prediction_axes=np.array([obstacle_axes(o,j) for j in range(config['horizon']+1)]).tolist(),
                prediction_angles=obstacle_angles(o,config['horizon']+1).tolist(),peer=bool(o.get('peer',False)),guidance_motion=o.get('guidance_motion')) for o in obstacles],
            previews={mode:{k:v for k,v in p.items() if k not in ('states','controls','vectors')} for mode,p in previews.items()},
            decision_ms=1000*(time.perf_counter()-begin))
    mode=saved['selected']
    memory['progress_summary']=dict(mode=mode,decision_id=saved['decision_id'],obstacle=name,
        preview_scope='mpc_window' if saved.get('local',False) else 'long',search_complete=saved.get('search_complete',True))
    if mode=='legacy':
        if config.get('progress_abstain',True):
            memory['progress_summary']['mode']='tracking_only'
            return np.zeros((config['horizon'],2)),0.
        return _legacy_continuous_guidance(state,goals,original_obstacles,config,memory)
    preview=(previews[mode] if reevaluate else progress_preview(state,goals,obstacles,index,mode,
        dict(config,progress_horizon=config['horizon']*config['dt'],progress_local=saved.get('local',False)),previous,bounds))
    if not preview['feasible']:
        if config.get('progress_abstain',True):
            memory['progress_summary']['mode']='tracking_only_recheck'
            return np.zeros((config['horizon'],2)),0.
        memory['progress_summary']['mode']='legacy_recheck'
        return _legacy_continuous_guidance(state,goals,original_obstacles,config,memory)
    n=config['horizon']
    memory['progress_plan']=dict(controls=preview['controls'][:n].copy(),
        states=preview['states'][:n+1].copy(),plan_controls=preview['vectors'][:n].copy(),mode=mode)
    if config.get('execution_backup',False):
        return preview['vectors'][:n],1.,preview['states'][1:n+1,2].copy()
    return preview['vectors'][:n],1.


def continuous_guidance(state,goals,obstacles,config,memory):
    if config.get('progress_guidance',False):return progress_guidance(state,goals,obstacles,config,memory)
    return _legacy_continuous_guidance(state,goals,obstacles,config,memory)


def attach_progress_record(record,memory):
    record['progress']=dict(memory.get('progress_summary',{}))
    decision=memory.get('progress_decision')
    if decision is not None and decision['decision_id']!=memory.get('progress_logged_id'):
        record['progress_decision']=decision
        memory['progress_logged_id']=decision['decision_id']


class IterationDeadline(ca.Callback):
    """Clean IPOPT user-stop at iteration boundaries; not a preemptive timer."""
    def __init__(self, name, nx, ng, np_):
        ca.Callback.__init__(self)
        self.nx,self.ng,self.np=nx,ng,np_
        self.deadline=float('inf');self.iterations=0;self.stopped=False
        self.construct(name,{})

    def get_n_in(self):return len(ca.nlpsol_out())
    def get_n_out(self):return 1
    def get_name_in(self,index):return ca.nlpsol_out(index)
    def get_name_out(self,index):return 'stop'
    def get_sparsity_in(self,index):
        size={'x':self.nx,'f':1,'g':self.ng,'lam_x':self.nx,'lam_g':self.ng,'lam_p':self.np}
        return ca.Sparsity.dense(size[ca.nlpsol_out(index)],1)
    def get_sparsity_out(self,index):return ca.Sparsity.scalar()
    def eval(self,args):
        self.iterations+=1
        self.stopped=time.perf_counter()>=self.deadline
        return [float(self.stopped)]


class CachedMPCProblem:
    def __init__(self, config, obstacle_count, first_delta_bound):
        self.config = dict(config)
        self.n, self.dt = config['horizon'], config['dt']
        n = self.n
        self.opti = op = ca.Opti()
        self.x = x = op.variable(n+1, 3)
        self.u = u = op.variable(n, 2)
        self.r = r = op.variable(n+1, 2)
        self.q = q = op.variable(n, 2)
        self.initial = initial = op.parameter(3)
        self.goal = goal = op.parameter(n+1, 3)
        self.previous = previous = op.parameter(2)
        self.has_previous = has_previous = op.parameter()
        self.v_bounds = v_bounds = op.parameter(2)
        self.guide = guide = op.parameter(n, 2)
        self.guide_heading = op.parameter(n)
        self.guide_level = guide_level = op.parameter()
        # Optional stage-wise centers support odometry-predicted peer robots.
        # Static callers repeat the same center; their mathematical problem is unchanged.
        self.centers = op.parameter(n+1, 2*obstacle_count) if obstacle_count else None
        self.angles = op.parameter(n+1, obstacle_count) if obstacle_count else None
        self.axes = op.parameter(n+1, 2*obstacle_count) if obstacle_count else None
        self.barrier_gamma = op.parameter(obstacle_count) if obstacle_count else None
        op.subject_to(x[0, :] == initial.T)
        op.subject_to(r[0, :] == initial[:2].T)
        op.subject_to(op.bounded(v_bounds[0], u[:, 0], v_bounds[1]))
        op.subject_to(op.bounded(-config['omega_max'], u[:, 1], config['omega_max']))
        op.subject_to(op.bounded(-config['plan_v_max'], q, config['plan_v_max']))
        if config['delta_limit']:
            if first_delta_bound:
                op.subject_to(op.bounded(-config['delta_v'], u[0, 0]-previous[0], config['delta_v']))
                op.subject_to(op.bounded(-config['delta_w'], u[0, 1]-previous[1], config['delta_w']))
            for j in range(1, n):
                op.subject_to(op.bounded(-config['delta_v'], u[j, 0]-u[j-1, 0], config['delta_v']))
                op.subject_to(op.bounded(-config['delta_w'], u[j, 1]-u[j-1, 1], config['delta_w']))
        for j in range(n):
            theta = x[j, 2]+self.dt*u[j, 1]
            op.subject_to(x[j+1, 0] == x[j, 0]+self.dt*u[j, 0]*ca.cos(theta))
            op.subject_to(x[j+1, 1] == x[j, 1]+self.dt*u[j, 0]*ca.sin(theta))
            op.subject_to(x[j+1, 2] == theta)
            op.subject_to(r[j+1, :] == r[j, :]+self.dt*q[j, :])
        for k in range(obstacle_count):
            h = []
            for j in range(n+1):
                c, s = ca.cos(self.angles[j, k]), ca.sin(self.angles[j, k])
                dx, dy = x[j, 0]-self.centers[j, 2*k], x[j, 1]-self.centers[j, 2*k+1]
                a, b = self.axes[j, 2*k], self.axes[j, 2*k+1]
                h.append(b*(ca.sqrt(((c*dx+s*dy)/a)**2+((-s*dx+c*dy)/b)**2+1e-12)-1)-config['margin'])
            for j in range(n):
                op.subject_to(h[j+1] >= (1-self.barrier_gamma[k])*h[j])

        def quad(v, weights):
            return ca.mtimes([v, ca.diag(ca.DM(weights)), v.T])

        objective = 0
        for j in range(n):
            heading = ca.atan2(ca.sin(x[j, 2]-goal[j, 2]), ca.cos(x[j, 2]-goal[j, 2]))
            error = ca.horzcat(x[j, :2]-goal[j, :2], heading)
            objective += quad(error, config['q_state'])+quad(u[j, :], config['r_control'])
            objective += quad(r[j, :]-goal[j, :2], config['q_plan'])
            objective += quad(x[j, :2]-r[j, :], config['q_coupling'])
            objective += quad(q[j, :], config['r_plan'])
            velocity = ca.horzcat(u[j, 0]*ca.cos(x[j+1, 2]), u[j, 0]*ca.sin(x[j+1, 2]))
            objective += guide_level*config.get('guide_velocity_weight', 600.)*ca.sumsqr(velocity-guide[j, :])
            guide_angle = self.guide_heading[j]
            objective += guide_level*config.get('guide_heading_weight', 2.)*(1-ca.cos(x[j+1, 2]-guide_angle))
            smooth_scale = ca.horzcat(1+2*guide_level, 1+config.get('guide_delta_scale', 4.)*guide_level)
            if j == 0:
                objective += has_previous*quad(ca.sqrt(smooth_scale)*(u[j, :]-previous.T), config['r_delta'])
            else:
                objective += quad(ca.sqrt(smooth_scale)*(u[j, :]-u[j-1, :]), config['r_delta'])
                objective += quad(q[j, :]-q[j-1, :], config['r_delta_plan'])
        heading = ca.atan2(ca.sin(x[-1, 2]-goal[-1, 2]), ca.cos(x[-1, 2]-goal[-1, 2]))
        objective += quad(ca.horzcat(x[-1, :2]-goal[-1, :2], heading), config['q_terminal'])
        objective += quad(r[-1, :]-goal[-1, :2], config['q_plan'])
        objective += quad(x[-1, :2]-r[-1, :], config['q_coupling'])
        op.minimize(objective)
        self.cost_function=None
        if config.get('execution_backup',False):
            self.cost_function=ca.Function('cost_'+str(id(self)),
                [x,u,r,q,goal,previous,has_previous,guide,self.guide_heading,guide_level],[objective])
        options={'ipopt.max_iter': config['max_iter'], 'ipopt.print_level': 0,
                  'ipopt.acceptable_tol': 1e-3, 'ipopt.acceptable_obj_change_tol': 1e-3,
                  'print_time': 0}
        self.last_attempt={}
        self.deadline_callback=None
        self.dual_shift=np.arange(op.ng)
        offset=5
        def shift_block(start,count,width=1):
            for j in range(count):
                self.dual_shift[start+j*width:start+(j+1)*width]=np.arange(
                    start+min(j+1,count-1)*width,start+(min(j+1,count-1)+1)*width)
        for _ in range(4):shift_block(offset,n);offset+=n  # v,w,q_x,q_y bounds
        if config['delta_limit']:
            count=n-1+int(first_delta_bound);shift_block(offset,count,2);offset+=2*count
        shift_block(offset,n,5);offset+=5*n  # dynamics: x,y,theta,r_x,r_y
        for _ in range(obstacle_count):shift_block(offset,n);offset+=n
        if offset!=op.ng:raise RuntimeError('Unexpected dual layout; refusing an invalid warm-start mapping')
        if config.get('realtime',False):
            self.deadline_callback=IterationDeadline('deadline_'+str(id(self)),op.nx,op.ng,op.np)
            options.update({'ipopt.max_wall_time':config.get('nlp_wall_time',.065),
                'iteration_callback':self.deadline_callback,
                'iteration_callback_step':1 if config.get('execution_backup',False) else 3,
                'ipopt.warm_start_init_point':'yes','ipopt.warm_start_bound_push':1e-6,
                'ipopt.warm_start_mult_bound_push':1e-6,'ipopt.warm_start_slack_bound_push':1e-6})
        op.solver('ipopt', options)

    def solve(self, state, goals, obstacles, bounds, previous, guess, plan_guess, guidance=None,
              deadline=None, dual_guess=None):
        begin=time.perf_counter();cpu_begin=time.thread_time()
        self.last_attempt=dict(status='setup',iterations=-1,wall_ms=0.,cpu_ms=0.,
            setup_ms=0.,nlp_ms=0.,validation_ms=0.,deadline_stop=False,feasible=False,
            dual_warm=False,min_cbf_slack=None,max_dynamics_error=None)
        op, cfg = self.opti, self.config
        op.set_value(self.initial, state)
        op.set_value(self.goal, goals)
        op.set_value(self.previous, np.zeros(2) if previous is None else previous)
        op.set_value(self.has_previous, float(previous is not None))
        op.set_value(self.v_bounds, bounds)
        vectors, level = guidance[:2] if guidance is not None else (np.zeros((self.n, 2)), 0.)
        op.set_value(self.guide, vectors)
        # atan2(0,0) is harmless as a numerical angle, but its symbolic parameter
        # derivative is undefined even when multiplied by an inactive weight.
        headings=guidance[2] if guidance is not None and len(guidance)>2 else np.arctan2(vectors[:, 1], vectors[:, 0])
        op.set_value(self.guide_heading,headings)
        op.set_value(self.guide_level, level)
        if obstacles:
            op.set_value(self.barrier_gamma, np.array([o.get('gamma',cfg['gamma']) for o in obstacles]))
            op.set_value(self.centers, np.column_stack([
                np.asarray(o.get('prediction_centers', np.tile(o['center'], (self.n+1, 1))))
                for o in obstacles]))
            op.set_value(self.angles, np.column_stack([
                obstacle_angles(o, self.n+1) for o in obstacles]))
            op.set_value(self.axes, np.column_stack([
                np.array([obstacle_axes(o, j) for j in range(self.n+1)]) for o in obstacles]))
        op.set_initial(self.u, guess)
        op.set_initial(self.x, rollout(state, guess, self.dt))
        plan_states = np.vstack((state[:2], state[:2]+self.dt*np.cumsum(plan_guess, axis=0)))
        op.set_initial(self.r, plan_states)
        op.set_initial(self.q, plan_guess)
        warm=(cfg.get('realtime',False) and dual_guess is not None
              and np.shape(dual_guess)==(op.ng,) and np.isfinite(dual_guess).all())
        op.set_initial(op.lam_g, dual_guess if warm else 0)
        self.last_attempt['dual_warm']=bool(warm)
        if self.deadline_callback is not None:
            self.deadline_callback.deadline=deadline if deadline is not None else float('inf')
            self.deadline_callback.iterations=0;self.deadline_callback.stopped=False
        self.last_attempt['setup_ms']=1000*(time.perf_counter()-begin)
        solve_begin=time.perf_counter();solve_cpu_begin=time.thread_time();sol=None;error=None
        try:
            sol=op.solve()
        except RuntimeError as exc:
            error=exc
        self.last_attempt['nlp_ms']=1000*(time.perf_counter()-solve_begin)
        self.last_attempt['nlp_cpu_ms']=1000*(time.thread_time()-solve_cpu_begin)
        try:stats=op.stats()
        except RuntimeError:stats={}
        self.last_attempt.update(status=stats.get('return_status','exception'),
            iterations=int(stats.get('iter_count',-1)),
            deadline_stop=bool(self.deadline_callback is not None and self.deadline_callback.stopped))
        validate_begin=time.perf_counter()
        try:
            value=sol.value if sol is not None else op.debug.value
            controls=np.asarray(value(self.u)).reshape(self.n,2)
            plans=np.asarray(value(self.q)).reshape(self.n,2)
            states=np.asarray(value(self.x)).reshape(self.n+1,3)
            finite=all(np.isfinite(v).all() for v in (states,controls,plans))
            self.last_attempt['finite_candidate']=bool(finite)
            self.last_attempt['max_dynamics_error']=float(np.max(np.abs(rollout(state,controls,self.dt)-states)))
            if finite:
                self.last_attempt['max_control_bound_violation']=float(max(0.,bounds[0]-controls[:,0].min(),
                    controls[:,0].max()-bounds[1],np.abs(controls[:,1]).max()-cfg['omega_max']))
                self.last_attempt['max_plan_bound_violation']=float(max(0.,np.abs(plans).max()-cfg['plan_v_max']))
                h=barrier_values(states,obstacles,cfg['margin'])
                if h.size:
                    gammas=np.array([o.get('gamma',cfg['gamma']) for o in obstacles])[:,None]
                    slack=h[:,1:]-(1-gammas)*h[:,:-1]
                    k,j=np.unravel_index(int(np.argmin(slack)),slack.shape)
                    self.last_attempt.update(raw_min_cbf_slack=float(slack[k,j]),
                        worst_obstacle=obstacles[k].get('name',str(k)),worst_stage=int(j))
                    if sol is None:
                        self.last_attempt['constraint_residuals']=[dict(obstacle=o.get('name',str(i)),
                            initial_h=float(h[i,0]),min_slack=float(slack[i].min()),
                            stage=int(np.argmin(slack[i]))) for i,o in enumerate(obstacles)]
            candidate=dict(states=states,controls=controls,plan_controls=plans,
                cost=float(value(op.f)),status=self.last_attempt['status'],converged=sol is not None)
            valid=self.validate(candidate,state,obstacles,bounds,previous)
            self.last_attempt.update(feasible=bool(valid),min_cbf_slack=candidate.get('min_slack'))
            if valid and (sol is not None or cfg.get('realtime',False)):
                if sol is not None:candidate['lam_g']=np.asarray(sol.value(op.lam_g)).reshape(-1)
                candidate['validated_steps']=self.n
                return candidate
            if cfg.get('realtime',False) and np.isfinite(controls).all() and np.isfinite(plans).all():
                # A time-limited iterate is only a seed until its exact rollout
                # passes the SAME bounds and D-CBF numerical validation.
                projected=dict(controls=controls,plan_controls=plans,
                    states=rollout(state,controls,self.dt),cost=candidate['cost'],
                    status=candidate['status'],converged=False)
                if self.validate(projected,state,obstacles,bounds,previous):
                    projected['validated_steps']=self.n
                    self.last_attempt.update(feasible=True,min_cbf_slack=projected.get('min_slack'))
                    return projected
                self.last_attempt['iterate_control']=controls[0].tolist()
                self.last_attempt['min_cbf_slack']=projected.get('min_slack')
            if error is not None:raise error
            raise RuntimeError('Solver result failed numerical feasibility validation')
        finally:
            self.last_attempt.update(validation_ms=1000*(time.perf_counter()-validate_begin),
                wall_ms=1000*(time.perf_counter()-begin),cpu_ms=1000*(time.thread_time()-cpu_begin))

    def validate(self, candidate, state, obstacles, bounds, previous, one_step=False):
        cfg, tol = self.config, 2e-6
        controls, states = candidate['controls'], candidate['states']
        if not all(np.isfinite(candidate[k]).all() for k in ('states', 'controls', 'plan_controls')):
            return False
        if np.max(np.abs(rollout(state, controls, self.dt)-states)) > tol:
            return False
        if (np.min(controls[:, 0]) < bounds[0]-tol or np.max(controls[:, 0]) > bounds[1]+tol
                or np.max(np.abs(controls[:, 1])) > cfg['omega_max']+tol
                or np.max(np.abs(candidate['plan_controls'])) > cfg['plan_v_max']+tol):
            return False
        if cfg['delta_limit']:
            increments = np.diff(controls, axis=0)
            if previous is not None:
                increments = np.vstack((controls[0]-previous, increments))
            if np.any(np.abs(increments) > np.array([cfg['delta_v'], cfg['delta_w']])+tol):
                return False
        values = barrier_values(states, obstacles, cfg['margin'])
        if values.size:
            gammas = np.array([o.get('gamma',cfg['gamma']) for o in obstacles])[:,None]
            slack = values[:, 1:]-(1-gammas)*values[:, :-1]
            checked = slack[:, :1] if one_step else slack
            candidate['min_slack'] = float(checked.min())
            candidate['min_h'] = float((values[:, :2] if one_step else values).min())
            if checked.min() < -tol:
                return False
            if values[:, 0].min() >= -tol and candidate['min_h'] < -tol:
                return False
        else:
            candidate.update(min_slack=float('nan'), min_h=float('nan'))
        return True


class MPCSolverManager:
    def __init__(self, config, recovery=True):
        self.config = dict(config)
        self.recovery = bool(recovery)
        self.problems = {}
        self.histories = {}
        self.robot_memory = {}
        self.last_records = {}
        self.builds = 0

    def _screen_backup(self,problem,state,goals,obstacles,bounds,previous,plan,guess,guidance,deadline):
        """Prepare a checked FIRST command; repetition is not a full-window claim."""
        cfg=self.config;n=cfg['horizon'];dt=cfg['dt']
        choices=[guess[0],previous,np.zeros(2)]
        v,w=np.meshgrid(np.linspace(*bounds,33),np.linspace(-cfg['omega_max'],cfg['omega_max'],33))
        choices=np.vstack((choices,np.column_stack((v.ravel(),w.ravel()))))
        if time.perf_counter()>=deadline:return None,0,0
        valid,positions,angles=first_step_mask(state,choices,obstacles,bounds,previous,cfg)
        errors_xy=positions-goals[1,:2]
        heading=(angles-goals[1,2]+np.pi)%(2*np.pi)-np.pi
        velocities=choices[:,0,None]*np.column_stack((np.cos(angles),np.sin(angles)))
        guide_angle=(guidance[2][0] if len(guidance)>2 else np.arctan2(guidance[0][0,1],guidance[0][0,0]))
        scores=(np.sum(errors_xy**2*np.asarray(cfg['q_state'][:2]),axis=1)+cfg['q_state'][2]*heading**2
            +np.sum(choices**2*np.asarray(cfg['r_control']),axis=1)
            +np.sum((choices-previous)**2*np.asarray(cfg['r_delta']),axis=1)
            +guidance[1]*cfg.get('guide_velocity_weight',600.)*np.sum((velocities-guidance[0][0])**2,axis=1)
            +guidance[1]*cfg.get('guide_heading_weight',2.)*(1-np.cos(angles-guide_angle)))
        for index in np.argsort(np.where(valid,scores,np.inf)):
            if not valid[index] or time.perf_counter()>=deadline:break
            command=choices[index]
            short=dict(controls=command[None,:],states=rollout(state,command[None,:],dt),plan_controls=np.zeros((1,2)))
            if not problem.validate(short,state,obstacles,bounds,previous,one_step=True):continue
            u=np.tile(command,(n,1))
            return dict(controls=u,states=rollout(state,u,dt),plan_controls=plan.copy(),cost=float(scores[index]),
                status='validated_one_step',seed='prepared_control',converged=False,
                validated_steps=1,min_slack=short['min_slack'],min_h=short['min_h']),len(choices),int(valid.sum())
        if cfg.get('infeasible_escape',False) and not valid.any() and obstacles and time.perf_counter()<deadline:
            # This is explicitly NOT a feasible DCBF solution. Keep input bounds,
            # protect currently safe barriers that zero can keep safe, and require an
            # improvement over the zero-command worst residual before using it.
            packed=obstacle_batch(obstacles,2,cfg['gamma'])
            h0,h1,_,_=first_step_values(state,choices,packed,cfg['margin'],dt)
            _,zero_h,_,_=first_step_values(state,np.zeros((1,2)),packed,cfg['margin'],dt)
            input_valid,_,_=first_step_mask(state,choices,[],bounds,previous,cfg)
            protected=(h0>=-2e-6)&(zero_h[:,0]>=-2e-6)
            input_valid&=np.all(h1[protected]>=-2e-6,axis=0)
            residual=np.min(h1-(1-packed.gamma[:,None])*h0[:,None],axis=0)
            zero_residual=float(np.min(zero_h[:,0]-(1-packed.gamma)*h0))
            if input_valid.any():
                value=float(np.max(residual[input_valid]))
                eligible=input_valid&(residual>=value-1e-9)
                index=int(np.argmin(np.where(eligible,scores,np.inf)))
                if value>zero_residual+2e-6:
                    u=np.tile(choices[index],(n,1))
                    return dict(controls=u,states=rollout(state,u,dt),plan_controls=plan.copy(),cost=float(scores[index]),
                        status='best_effort_escape',seed='escape_control',converged=False,validated_steps=0,
                        min_slack=float(residual[index]),min_h=float(min(h0.min(),h1[:,index].min())),
                        escape_zero_min_slack=zero_residual,escape_next_h=h1[:,index].tolist(),
                        escape_initial_h=h0.tolist()),len(choices),0
        return None,len(choices),int(valid.sum())

    def _full_backup(self,problem,state,goals,obstacles,bounds,previous,controls,plan,guidance,source):
        """Use the original validator and symbolic MPC cost, not preview scores."""
        candidate=dict(controls=np.array(controls,copy=True),plan_controls=np.array(plan,copy=True),
            states=rollout(state,controls,self.config['dt']),status='validated_full_backup',
            seed=source,converged=False,validated_steps=self.config['horizon'])
        if not problem.validate(candidate,state,obstacles,bounds,previous):return None
        vectors,level=guidance[:2]
        headings=guidance[2] if len(guidance)>2 else np.arctan2(vectors[:,1],vectors[:,0])
        r=np.vstack((state[:2],state[:2]+self.config['dt']*np.cumsum(plan,axis=0)))
        candidate['cost']=float(problem.cost_function(candidate['states'],controls,r,plan,goals,previous,1.,vectors,headings,level))
        return candidate if np.isfinite(candidate['cost']) else None

    def solve(self, state, goals, obstacles, bounds, last_solution, label,
              deadline=None, applied_previous=None, seed_shift=1, cycle_token=None):
        if self.config.get('realtime',False):
            return self._solve_realtime(state,goals,obstacles,bounds,last_solution,label,
                deadline,applied_previous,seed_shift,cycle_token)
        return self._solve_legacy(state,goals,obstacles,bounds,last_solution,label,cycle_token)

    def _solve_realtime(self,state,goals,obstacles,bounds,last,label,deadline,previous,shift,token):
        begin=time.perf_counter();cfg=self.config;n=cfg['horizon'];dt=cfg['dt']
        command_deadline=deadline if deadline is not None else begin+cfg.get('manager_wall_time',.085)
        deadline=min(command_deadline,begin+cfg.get('manager_wall_time',.085))
        state=np.asarray(state,float);goals=np.asarray(goals,float)
        if previous is None:previous=np.zeros(2)
        previous=np.asarray(previous,float)
        key=(len(obstacles),bool(cfg['delta_limit']))
        if key not in self.problems:
            self.problems[key]=CachedMPCProblem(cfg,*key);self.builds+=1
        problem=self.problems[key]
        memory=self.robot_memory.setdefault(label,dict(positions=deque(maxlen=11),cooldown=0,preferred=1,hold=0))
        memory['positions'].append(state[:2].copy())
        memory['cooldown']=max(0,memory['cooldown']-1)
        if cfg.get('progress_guidance',False):memory.update(progress_previous=previous.copy(),progress_bounds=list(bounds))
        signature=tuple((o.get('name',str(i)),o.get('gamma',cfg['gamma'])) for i,o in enumerate(obstacles))
        usable=last is not None and np.shape(last.get('controls'))==(n,2)
        shift=min(n,max(1,int(shift)))
        if usable:
            controls=np.asarray(last['controls'])
            guess=np.vstack((controls[shift:],np.tile(controls[-1],(shift,1))))
            q=np.asarray(last.get('plan_controls',np.zeros((n,2))))
            plan=np.vstack((q[shift:],np.tile(q[-1],(shift,1))))
        else:
            delta=goals[1,:2]-state[:2]
            target=np.arctan2(delta[1],delta[0]) if np.linalg.norm(delta)>1e-8 else goals[0,2]
            angle=(target-state[2]+np.pi)%(2*np.pi)-np.pi
            guess=np.tile([cfg['fallback_kv']*np.linalg.norm(delta)*np.cos(angle),cfg['fallback_kw']*angle],(n,1))
            plan=np.diff(goals[:,:2],axis=0)/dt
        guess[:,0]=np.clip(guess[:,0],*bounds)
        guess[:,1]=np.clip(guess[:,1],-cfg['omega_max'],cfg['omega_max'])
        plan=np.clip(plan,-cfg['plan_v_max'],cfg['plan_v_max'])
        seed='shifted' if usable else 'tracking';side=None
        prepared=None;full_backup=None;prepare_ms=0.;checked=0;screen_valid=0
        execution_backup=cfg.get('execution_backup',False)
        if execution_backup:
            prepare_begin=time.perf_counter()
            # Do this before reference preview or NLP can spend the remaining budget.
            prepared,checked,screen_valid=self._screen_backup(problem,state,goals,obstacles,bounds,
                previous,plan,guess,(np.zeros((n,2)),0.),command_deadline)
            prepare_ms=1000*(time.perf_counter()-prepare_begin)
        preview_begin=time.perf_counter();preview_budget_hit=False
        preview_config=(dict(cfg,bounded_local_reference=cfg.get('bounded_local_reference',True),
            align_reference_hypotheses=cfg.get('align_reference_hypotheses',True),
            predictive_obstruction=cfg.get('predictive_obstruction',cfg.get('predictive_bypass',False)),
            active_side_lock=cfg.get('active_side_lock',cfg.get('predictive_bypass',False)),
            relative_avoidance_flow=cfg.get('relative_avoidance_flow',cfg.get('predictive_bypass',False)),_preview_deadline=min(deadline-.030,
            preview_begin+cfg.get('progress_wall_time',.025))) if execution_backup else cfg)
        try:guidance=continuous_guidance(state,goals,obstacles,preview_config,memory)
        except PreviewBudgetExceeded:
            memory.pop('progress_plan',None)
            memory['progress_summary']=dict(mode='budget_deferred',decision_id=memory.get('progress_counter',0))
            guidance=np.zeros((n,2)),0.;preview_budget_hit=True
        preview_ms=1000*(time.perf_counter()-preview_begin)
        if execution_backup and time.perf_counter()<deadline-.005:
            prepare_begin=time.perf_counter()
            proposals=[(seed,guess,plan)]
            preview=memory.get('progress_plan')
            if preview is not None:proposals.append(('progress_prefix',preview['controls'],preview['plan_controls']))
            for source,u,q in proposals:
                if time.perf_counter()>=deadline-.005:break
                candidate=self._full_backup(problem,state,goals,obstacles,bounds,previous,u,q,guidance,source)
                if candidate is not None and (full_backup is None or candidate['cost']<full_backup['cost']):full_backup=candidate
            prepare_ms+=1000*(time.perf_counter()-prepare_begin)
        pending=memory.setdefault('pending_turns',[])
        if self.recovery and pending and full_backup is None:
            side=pending.pop(0);guess=np.tile([min(.12,bounds[1]),side*min(1.2,cfg['omega_max'])],(n,1))
            seed='turn_left' if side>0 else 'turn_right'
        dual=None
        if (usable and side is None and last.get('signature')==signature and last.get('converged',False)
                and cfg.get('dual_warm_start',True) and np.shape(last.get('lam_g'))==(problem.opti.ng,)
                and np.linalg.norm(state[:2]-last['states'][0,:2])<.25):
            dual=np.asarray(last['lam_g']).copy()
            for _ in range(shift):dual=dual[problem.dual_shift]
        if full_backup is not None:
            guess=full_backup['controls'].copy();plan=full_backup['plan_controls'].copy()
            if full_backup['seed']!=seed:dual=None
            seed=full_backup['seed']
        trace=[];best=None;errors=[]
        # Exactly one NLP per period. Alternative seeds are carried to later
        # periods rather than serially consuming this period's control budget.
        reserve=max(cfg.get('fallback_reserve',.010),.020 if execution_backup else 0.)
        if time.perf_counter()<deadline-reserve-.005:
            try:
                best=problem.solve(state,goals,obstacles,bounds,previous,guess,plan,guidance,
                    deadline=deadline-reserve,dual_guess=dual)
                best['seed']=seed
            except RuntimeError as exc:errors.append(str(exc).splitlines()[-1])
            trace.append(dict(problem.last_attempt,seed=seed))
        else:errors.append('No remaining NLP budget')
        if full_backup is not None and (best is None or full_backup['cost']<best['cost']):best=full_backup
        if best is None and prepared is not None:best=prepared
        full_candidate=best is not None
        if execution_backup:full_candidate=best is not None and best['validated_steps']==n
        converged=bool(full_candidate and best.get('converged',False))
        current_h=barrier_values(np.tile(state,(n+1,1)),obstacles,cfg['margin'])
        near=bool(current_h.size and current_h[:,0].min()<.08)
        stopped=(len(memory['positions'])==11 and np.linalg.norm(state[:2]-memory['positions'][0])<.008)
        quiet=best is None or (abs(best['controls'][0,0])<.01 and abs(best['controls'][0,1])<.4)
        recovery=(self.recovery and (best is None or (near and stopped and quiet
                    and np.linalg.norm(goals[0,:2]-state[:2])>.15 and memory['cooldown']==0)))
        if recovery and not pending:
            if side is None:pending.extend([memory['preferred'],-memory['preferred']])
            else:pending.extend([-side])
            memory['cooldown']=max(1,int(1.5/dt))
        if best is not None and side is not None and not quiet:
            memory['preferred']=side;pending.clear()
        fallback_begin=time.perf_counter()
        if best is None and not execution_backup:
            choices=[guess[0],previous,np.zeros(2)]
            if trace and 'iterate_control' in trace[-1]:choices.insert(0,np.asarray(trace[-1]['iterate_control']))
            v,w=np.meshgrid(np.linspace(*bounds,33),np.linspace(-cfg['omega_max'],cfg['omega_max'],33))
            choices=np.vstack((choices,np.column_stack((v.ravel(),w.ravel()))))
            valid,positions,angles=first_step_mask(state,choices,obstacles,bounds,previous,cfg)
            screen_valid=int(valid.sum())
            errors_xy=positions-goals[1,:2]
            heading=(angles-goals[1,2]+np.pi)%(2*np.pi)-np.pi
            velocities=choices[:,0,None]*np.column_stack((np.cos(angles),np.sin(angles)))
            guide_angle=np.arctan2(guidance[0][0,1],guidance[0][0,0])
            scores=(np.sum(errors_xy**2*np.asarray(cfg['q_state'][:2]),axis=1)+cfg['q_state'][2]*heading**2
                +np.sum(choices**2*np.asarray(cfg['r_control']),axis=1)
                +np.sum((choices-previous)**2*np.asarray(cfg['r_delta']),axis=1)
                +guidance[1]*cfg.get('guide_velocity_weight',600.)*np.sum((velocities-guidance[0][0])**2,axis=1)
                +guidance[1]*cfg.get('guide_heading_weight',2.)*(1-np.cos(angles-guide_angle)))
            checked=len(choices)
            for index in np.argsort(np.where(valid,scores,np.inf)):
                if not valid[index] or time.perf_counter()>command_deadline:break
                command=choices[index]
                short=dict(controls=command[None,:],states=rollout(state,command[None,:],dt),
                           plan_controls=np.zeros((1,2)))
                if not problem.validate(short,state,obstacles,bounds,previous,one_step=True):continue
                u=np.tile(command,(n,1))
                best=dict(controls=u,states=rollout(state,u,dt),plan_controls=plan.copy(),cost=float(scores[index]),
                    status='validated_one_step',seed='screened_control',converged=False,
                    validated_steps=1,min_slack=short['min_slack'],min_h=short['min_h'])
                break
        fallback_ms=1000*(time.perf_counter()-fallback_begin)
        result_status=('no_validated_control' if best is None else 'best_effort_escape' if best.get('status')=='best_effort_escape' else
                       'converged' if converged else 'validated_full_backup' if best.get('status')=='validated_full_backup'
                       else 'feasible_iterate' if full_candidate else 'validated_one_step')
        record=dict(cycle_token=token,attempts=len(trace),failed_attempts=sum(not t.get('feasible',False) for t in trace),
            success=converged,accepted_candidate=best is not None,status=result_status,
            recovery_triggered=bool(recovery or side is not None),recovery_improved=False,
            recovery_candidate_found=bool(best is not None and side is not None),
            seed=seed,cost=best['cost'] if best is not None else None,
            min_cbf_slack=best.get('min_slack') if best is not None else None,
            wall_ms=1000*(time.perf_counter()-begin),guide_level=guidance[1],
            guide_obstacle=memory.get('guide_index'),errors=errors,attempt_trace=trace,
            fallback_ms=fallback_ms,fallback_checked=checked,fallback_screen_valid=screen_valid,dual_warm=dual is not None,
            budget_expired=time.perf_counter()>deadline,pending_turns=list(pending))
        if execution_backup:
            record.update(execution_backup=True,backup_prepare_ms=prepare_ms,
                preview_ms=preview_ms,preview_budget_hit=preview_budget_hit,
                preview_budget_ms=1000*cfg.get('progress_wall_time',.025),
                backup_prepared=prepared is not None and prepared.get('validated_steps',0)>0,
                escape_prepared=prepared is not None and prepared.get('status')=='best_effort_escape',
                full_backup_prepared=full_backup is not None,
                accepted_source=best.get('seed') if best is not None else None,
                accepted_validated_steps=best.get('validated_steps',0) if best is not None else 0)
            if best is not None and best.get('status')=='best_effort_escape':
                record.update(escape_zero_min_slack=best['escape_zero_min_slack'],
                    escape_initial_h=best['escape_initial_h'],escape_next_h=best['escape_next_h'])
        if current_h.size:record['min_model_h0']=float(current_h[:,0].min())
        if cfg.get('progress_guidance',False):attach_progress_record(record,memory)
        self.histories.setdefault(label,[]).append(record);self.last_records[label]=record
        if best is None:
            raise RuntimeError(label+': no validated control within the current computation budget')
        best.update(success=converged,signature=signature,obstacle_count=len(obstacles))
        return best

    def _solve_legacy(self, state, goals, obstacles, bounds, last_solution, label, cycle_token=None):
        begin = time.perf_counter()
        cfg, n = self.config, self.config['horizon']
        state, goals = np.asarray(state, float), np.asarray(goals, float)
        controls = None if last_solution is None else last_solution.get('controls')
        plans = None if last_solution is None else last_solution.get('plan_controls')
        valid_last = controls is not None and np.shape(controls) == (n, 2)
        previous = np.asarray(controls)[0].copy() if valid_last else None
        key = (len(obstacles), bool(cfg['delta_limit'] and previous is not None))
        if key not in self.problems:
            self.problems[key] = CachedMPCProblem(cfg, *key)
            self.builds += 1
        problem = self.problems[key]
        if valid_last:
            guess = np.vstack((controls[1:], controls[-1:])).copy()
        else:
            difference = goals[1, :2]-state[:2]
            angle = np.arctan2(difference[1], difference[0]) if np.linalg.norm(difference)>1e-8 else goals[0, 2]
            heading_error = (angle-state[2]+np.pi)%(2*np.pi)-np.pi
            command = [np.clip(cfg['fallback_kv']*np.linalg.norm(difference)*np.cos(heading_error), *bounds),
                       np.clip(cfg['fallback_kw']*heading_error, -cfg['omega_max'], cfg['omega_max'])]
            guess = np.tile(command, (n, 1))
        guess[:, 0] = np.clip(guess[:, 0], *bounds)
        guess[:, 1] = np.clip(guess[:, 1], -cfg['omega_max'], cfg['omega_max'])
        if cfg.get('feasible_seed_search', False) and obstacles:
            # A changing visible-contour envelope can invalidate the shifted
            # primal guess. Screen cheap rollouts before calling the NLP; this
            # changes initialization only, never the objective or constraints.
            def seed_score(seed):
                predicted = rollout(state, seed, cfg['dt'])
                h = barrier_values(predicted, obstacles, cfg['margin'])
                gammas = np.array([o.get('gamma',cfg['gamma']) for o in obstacles])[:,None]
                slack = h[:, 1:]-(1-gammas)*h[:, :-1]
                worst = float(slack.min())
                tracking = float(np.sum((predicted[:, :2]-goals[:, :2])**2))
                return worst, tracking
            initial_slack, initial_error = seed_score(guess)
            if initial_slack < -2e-4:
                candidates = [(guess, initial_slack, initial_error)]
                for speed in (-.2, -.1, 0., .1, .2):
                    for yaw_rate in (-1.2, 0., 1.2):
                        seed = np.tile([np.clip(speed,*bounds),
                            np.clip(yaw_rate,-cfg['omega_max'],cfg['omega_max'])],(n,1))
                        slack, tracking = seed_score(seed)
                        candidates.append((seed,slack,tracking))
                feasible = [item for item in candidates if item[1]>=-2e-6]
                choice = min(feasible,key=lambda item:item[2]) if feasible else max(candidates,key=lambda item:item[1])
                guess = choice[0].copy()
        if plans is not None and np.shape(plans) == (n, 2):
            plan_guess = np.vstack((plans[1:], plans[-1:]))
        else:
            plan_guess = np.diff(goals[:, :2], axis=0)/cfg['dt']
        plan_guess = np.clip(plan_guess, -cfg['plan_v_max'], cfg['plan_v_max'])
        memory = self.robot_memory.setdefault(label, {'positions': deque(maxlen=11),
                   'cooldown': 0, 'preferred': 1, 'hold': 0})
        if cfg.get('progress_guidance',False):memory.update(progress_previous=np.zeros(2) if previous is None else np.array(previous),progress_bounds=list(bounds))
        guidance = continuous_guidance(state, goals, obstacles, cfg, memory)
        attempts, errors, trace = 0, [], []

        def attempt(seed, name):
            nonlocal attempts
            attempts += 1
            try:
                result = problem.solve(state, goals, obstacles, bounds, previous, seed, plan_guess, guidance)
                result['seed'] = name
                trace.append(dict(problem.last_attempt,seed=name))
                return result
            except RuntimeError as error:
                errors.append(str(error).splitlines()[-1])
                trace.append(dict(problem.last_attempt,seed=name))
                return None

        best = attempt(guess, 'shifted' if valid_last else 'tracking')
        memory['positions'].append(state[:2].copy())
        memory['cooldown'] = max(0, memory['cooldown']-1)
        memory['hold'] = max(0, memory['hold']-1)
        error_norm = np.linalg.norm(goals[0, :2]-state[:2])
        current_h = barrier_values(np.tile(state, (n+1, 1)), obstacles, cfg['margin'])
        near = bool(current_h.size and current_h[:, 0].min() < .08)
        stopped = (len(memory['positions']) == memory['positions'].maxlen
                   and np.linalg.norm(state[:2]-memory['positions'][0]) < .008)
        quiet = best is None or (abs(best['controls'][0, 0]) < .01
                                 and abs(best['controls'][0, 1]) < .4)
        trigger = self.recovery and (best is None or (near and error_norm > .15 and quiet
                   and memory['cooldown'] == 0 and (stopped or memory['hold'] > 0)))
        improved = False
        if trigger:
            for side in (memory['preferred'], -memory['preferred']):
                seed = np.tile([min(.12, bounds[1]), side*min(1.2, cfg['omega_max'])], (n, 1))
                candidate = attempt(seed, 'turn_left' if side > 0 else 'turn_right')
                if candidate is not None and (best is None or candidate['cost'] < best['cost']-1e-5*max(1., abs(best['cost']))):
                    best = candidate
                    memory['preferred'] = side
                    improved = True
            memory['cooldown'] = max(1, int(1.5/cfg['dt']))
            if improved:
                memory['hold'] = max(1, int(3.0/cfg['dt']))
        success = best is not None
        if best is None:
            # Only validated fallback controls may reach the plant.
            for seed, name in ((guess, 'validated_shift'), (np.zeros((n, 2)), 'validated_stop')):
                candidate = {'states': rollout(state, seed, cfg['dt']), 'controls': seed.copy(),
                             'plan_controls': plan_guess.copy(), 'cost': float('nan'),
                             'status': name, 'seed': name}
                if problem.validate(candidate, state, obstacles, bounds, previous, one_step=True):
                    best = candidate
                    break
            if best is None:
                record=dict(cycle_token=cycle_token,attempts=attempts,failed_attempts=len(errors),success=False,
                    recovery_triggered=bool(trigger),recovery_improved=False,seed='none',status='no_validated_control',
                    cost=None,min_cbf_slack=None,wall_ms=1000*(time.perf_counter()-begin),
                    guide_level=guidance[1],guide_obstacle=memory.get('guide_index'),errors=errors,attempt_trace=trace)
                self.histories.setdefault(label,[]).append(record);self.last_records[label]=record
                raise RuntimeError(label+': no validated one-step fallback; refusing an unchecked control')
            values = barrier_values(best['states'], obstacles, cfg['margin'])
            best['min_h'] = float(values.min()) if values.size else float('nan')
        record = {'attempts': attempts, 'failed_attempts': len(errors), 'success': success,
                  'cycle_token':cycle_token,'attempt_trace':trace,
                  'recovery_triggered': bool(trigger), 'recovery_improved': bool(improved),
                  'seed': best['seed'], 'status': best['status'], 'cost': best['cost'],
                  'min_cbf_slack': best['min_slack'], 'wall_ms': 1000*(time.perf_counter()-begin),
                  'guide_level': guidance[1], 'guide_obstacle': memory.get('guide_index'),
                  'guide_side': memory.get('guide_side'),
                  'route_switches': memory.get('route_switches', 0),
                  'route_cost_ccw': memory.get('route_cost_ccw', float('nan')),
                  'route_cost_cw': memory.get('route_cost_cw', float('nan')),
                  'projected_follow': memory.get('projected_follow', False),
                  'errors': errors}
        if cfg.get('progress_guidance',False):attach_progress_record(record,memory)
        self.histories.setdefault(label, []).append(record);self.last_records[label]=record
        best.update(success=success, obstacle_count=len(obstacles))
        return best

    def summary(self):
        summary = {'cached_problem_builds': self.builds, 'recovery_enabled': self.recovery, 'robots': {}}
        for label, records in self.histories.items():
            times = [r['wall_ms'] for r in records]
            summary['robots'][label] = {'cycles': len(records),
                'solver_attempts': sum(r['attempts'] for r in records),
                'failed_attempts': sum(r['failed_attempts'] for r in records),
                'failed_cycles': sum(not r.get('accepted_candidate',r['status']!='no_validated_control') for r in records),
                'nonconverged_cycles':sum(not r['success'] for r in records),
                'validated_limited_cycles':sum(r.get('accepted_candidate',r['status']!='no_validated_control') and not r['success'] for r in records),
                'recovery_triggers': sum(r['recovery_triggered'] for r in records),
                'recovery_improvements': sum(r['recovery_improved'] for r in records),
                'recovery_candidates_found':sum(r.get('recovery_candidate_found',False) for r in records),
                'cycle_mean_ms': float(np.mean(times)), 'cycle_max_ms': float(np.max(times)),
                'cycle_p95_ms':float(np.percentile(times,95)),'cycle_p99_ms':float(np.percentile(times,99)),
                'ipopt_iterations':sum(max(0,t.get('iterations',0)) for r in records for t in r.get('attempt_trace',[])),
                'deadline_stops':sum(t.get('deadline_stop',False) or 'WallTime' in t.get('status','')
                    for r in records for t in r.get('attempt_trace',[]))}
        return summary
