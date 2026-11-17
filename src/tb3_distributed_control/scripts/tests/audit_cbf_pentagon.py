#!/usr/bin/env python3
"""Nominal formation/peer compatibility and conservative URDF collision bounds.

No controller is run. Primitive bounding boxes enclose the URDF collisions,
not visual meshes, real attachments or localization/tracking uncertainty.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from cbf_mpc_core import default_config,peer_obstacle
from mpc_dcbf_cached_solver import rollout,barrier_values,safety_residuals,first_step_mask


def transform(origin):
    xyz=np.zeros(3);rpy=np.zeros(3)
    if origin is not None:
        xyz=np.array([float(v) for v in origin.get('xyz','0 0 0').split()])
        rpy=np.array([float(v) for v in origin.get('rpy','0 0 0').split()])
    r,p,y=rpy;cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    result=np.eye(4);result[:3,:3]=rz@ry@rx;result[:3,3]=xyz
    return result


def body_bound():
    robot=ET.parse(PACKAGE/'urdf/turtlebot3_burger.urdf.xacro').getroot()
    matrices={'base_footprint':np.eye(4)};pending=list(robot.findall('joint'))
    while pending:
        done=[]
        for joint in pending:
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if parent in matrices:
                matrices[child]=matrices[parent]@transform(joint.find('origin'));done.append(joint)
        if not done:raise ValueError('Unresolved literal URDF joint chain')
        pending=[j for j in pending if j not in done]
    bounds=[]
    for link in robot.findall('link'):
        for collision in link.findall('collision'):
            geom=collision.find('geometry');box=geom.find('box');cylinder=geom.find('cylinder')
            if box is not None:half=np.array([float(x) for x in box.get('size').split()])/2
            elif cylinder is not None:
                r=float(cylinder.get('radius'));half=np.array([r,r,float(cylinder.get('length'))/2])
            else:raise ValueError('Unsupported collision primitive')
            corners=np.array(list(itertools.product((-1,1),repeat=3)))*half
            matrix=matrices[link.get('name')]@transform(collision.find('origin'))
            world=corners@matrix[:3,:3].T+matrix[:3,3]
            bounds.append(dict(link=link.get('name'),planar_outer_radius=float(np.linalg.norm(world[:,:2],axis=1).max())))
    return dict(parts=bounds,collision_outer_radius=max(b['planar_outer_radius'] for b in bounds),
        scope='Conservative primitive-box planar bound at nominal joints; not a measured physical robot radius')


def audit(radius=.4):
    cfg=default_config();points=radius*np.column_stack((np.cos(2*np.pi*np.arange(5)/5),np.sin(2*np.pi*np.arange(5)/5)))
    controls=np.tile([.15,0.],(cfg['horizon'],1));minimum_h=math.inf;minimum_residual=math.inf;all_valid=True
    for i in range(5):
        obs=[peer_obstacle(j,np.r_[points[j],0.],[.15,0.],.1,10,.2) for j in range(5) if j!=i]
        values=barrier_values(rollout(np.r_[points[i],0.],controls,.1),obs,cfg['margin'])
        minimum_h=min(minimum_h,float(values.min()));minimum_residual=min(minimum_residual,float(safety_residuals(values,obs,cfg).min()))
        all_valid &= bool(first_step_mask(np.r_[points[i],0.],controls[:1],obs,[-.22,.22],controls[0],cfg)[0][0])
    body=body_bound();spacing=float(2*radius*np.sin(np.pi/5))
    return dict(formation_radius=radius,positions=points.tolist(),adjacent_center_distance=spacing,
        physical_proxy_radius=.13,proxy_surface_gap=spacing-.26,peer_prior_radius=.2,ego_barrier_margin=cfg['margin'],
        circular_peer_center_boundary=.2+cfg['margin'],minimum_barrier=minimum_h,minimum_dcbf_residual=minimum_residual,
        same_speed_nominal_control_valid=all_valid,body_bound=body,
        proxy_contains_urdf_collision_bound=body['collision_outer_radius']<=.13,
        passed=all_valid and minimum_h>0 and minimum_residual>0 and body['collision_outer_radius']<=.13,
        limits='Nominal common-heading same-speed circle-prior compatibility only; no five-robot closed-loop or physical uncertainty guarantee')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    result=audit();out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
