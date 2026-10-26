#!/usr/bin/env python3
"""Independent Gazebo truth plots, forecast error checks, optional animation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Ellipse
from matplotlib.animation import FuncAnimation,PillowWriter


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--gif',action='store_true')
    args=parser.parse_args();p=Path(args.directory)
    rows=list(csv.DictReader((p/'gazebo_truth.csv').open()))
    if not rows:raise ValueError('No independent Gazebo samples')
    leader=list(csv.DictReader((p/'leader.csv').open()))
    mission_end=np.ceil(max(float(r['trajectory_time']) for r in leader))
    rows=[r for r in rows if float(r['time'])<=mission_end+.05]
    times=np.array([float(r['time']) for r in rows]);ros=np.array([float(r['ros_time']) for r in rows])
    if times.max()>10000:raise ValueError('Historical draft evaluator used wall time; do not plot as simulation time.')
    origin=ros[0]-times[0];meta=json.loads((p/'run_metadata.json').read_text())
    specs=json.loads((p/'dynamic_scenario.json').read_text())['specifications'];names=meta['robot_names']
    positions={name:np.array([[float(r[name+'_x']),float(r[name+'_y'])] for r in rows]) for name in names+[s['name'] for s in specs]}
    predictions=[json.loads(line) for line in (p/'obstacle_predictions.jsonl').open() if line.strip()]
    selected=[r for r in predictions if r['observer']==names[0]]
    comparison={};errors={0:[],10:[]};displacement_errors={0:[],10:[]}
    # Match only offline, using simultaneous truth. No association labels enter
    # the controller, and predictions outside the recorded time interval are skipped.
    for frame in selected:
        time=frame['stamp']-origin
        if not times[0]<=time<=times[-1]:continue
        for spec in specs:
            name=spec['name'];actual=np.array([np.interp(time,times,positions[name][:,a]) for a in (0,1)])
            candidates=frame['tracks']
            if not candidates:continue
            track=min(candidates,key=lambda tr:np.linalg.norm(np.asarray(tr['centers'][0])-actual))
            if np.linalg.norm(np.asarray(track['centers'][0])-actual)>.45:continue
            for j in errors:
                future=time+j*frame['dt']
                if j>=len(track['centers']) or future>min(times[-1],mission_end):continue
                actual_future=np.array([np.interp(future,times,positions[name][:,a]) for a in (0,1)])
                errors[j].append(float(np.linalg.norm(np.asarray(track['centers'][j])-actual_future)))
                displacement_errors[j].append(float(np.linalg.norm(
                    np.asarray(track['centers'][j])-track['centers'][0]-(actual_future-actual))))
    for j,values in errors.items():
        if values:comparison[str(j)]=dict(samples=len(values),rmse=float(np.sqrt(np.mean(np.square(values)))),
            mean=float(np.mean(values)),maximum=float(max(values)),
            displacement_rmse=float(np.sqrt(np.mean(np.square(displacement_errors[j])))))
    (p/'prediction_accuracy.json').write_text(json.dumps(dict(observer=names[0],errors_by_node=comparison,
        note='Offline nearest-truth association within .45 m; centres are ellipse estimates, not known object centres.'),indent=2))
    fig,axes=plt.subplots(2,2,figsize=(13,9));ax=axes[0,0]
    for name in names:ax.plot(*positions[name].T,lw=2,label=name)
    for spec in specs:
        q=positions[spec['name']];ax.plot(*q.T,'--',lw=1.5,label=spec['name'])
        for index in np.linspace(0,len(times)-1,4,dtype=int):ax.add_patch(Circle(q[index],spec['radius'],fill=False,color='darkorange',alpha=.5))
    ax.set_aspect('equal');ax.legend(fontsize=8);ax.set_title('Actual Gazebo trajectories');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    for i,name in enumerate(names):
        clear=np.min([np.linalg.norm(positions[name]-positions[s['name']],axis=1)-.13-s['radius'] for s in specs],axis=0)
        axes[0,1].plot(times,clear,lw=1.5,label=name)
        f=p/('tb3_%d.csv'%i);data=list(csv.DictReader(f.open()))
        axes[1,0].plot([float(r['time']) for r in data],[np.hypot(float(r['consensus_error_x']),float(r['consensus_error_y'])) for r in data],lw=1.5,label=name)
        axes[1,1].plot([float(r['time']) for r in data],[float(r['solve_ms']) for r in data],lw=.8,label=name)
    axes[0,1].axhline(0,color='red',ls='--');axes[0,1].set_title('Dynamic body clearance (circle proxy)');axes[0,1].set_ylabel('m')
    axes[1,0].set_title('Nominal reference position error');axes[1,0].set_ylabel('m')
    axes[1,1].axhline(100,color='red',ls='--');axes[1,1].set_title('Controller wall time');axes[1,1].set_ylabel('ms')
    for ax in axes.flat:ax.grid(alpha=.3)
    for ax in (axes[0,1],axes[1,0],axes[1,1]):ax.set_xlabel('Simulation trajectory time (s)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(p/'dynamic_diagnostics.png',dpi=300);fig.savefig(p/'dynamic_diagnostics.pdf');plt.close(fig)
    if not args.gif:return
    fig,ax=plt.subplots(figsize=(8,6))
    all_pos=np.vstack(list(positions.values()));lo=all_pos.min(axis=0)-.6;hi=all_pos.max(axis=0)+.6
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.grid(alpha=.25)
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    bodies=[];paths=[]
    for i,name in enumerate(names):
        color=plt.cm.tab10(i);patch=Circle(positions[name][0],.13,color=color);ax.add_patch(patch)
        line,=ax.plot([],[],color=color,lw=1.5,label=name);bodies.append(patch);paths.append(line)
    actors=[]
    for spec in specs:
        patch=Circle(positions[spec['name']][0],spec['radius'],color='darkorange',alpha=.65);ax.add_patch(patch);actors.append(patch)
        ax.plot(*positions[spec['name']].T,color='darkorange',ls=':',lw=.8,alpha=.5)
    predicted=[];envelopes=[]
    for _ in specs:
        line,=ax.plot([],[],'r--',lw=1.3);predicted.append(line)
        patch=Ellipse((0,0),0,0,fill=False,edgecolor='red',lw=1.);ax.add_patch(patch);envelopes.append(patch)
    frame_times=np.array([r['stamp']-origin for r in selected])
    indices=np.unique(np.linspace(0,len(times)-1,150,dtype=int))
    def update(index):
        for i,name in enumerate(names):
            bodies[i].center=positions[name][index];paths[i].set_data(*positions[name][:index+1].T)
        for patch,spec in zip(actors,specs):patch.center=positions[spec['name']][index]
        active=[]
        if len(frame_times):
            k=int(np.searchsorted(frame_times,times[index],side='right'))-1
            if k>=0 and times[index]-frame_times[k]<.4:active=[tr for tr in selected[k]['tracks'] if tr['dynamic']]
        for j,(line,patch) in enumerate(zip(predicted,envelopes)):
            if j>=len(active):line.set_data([],[]);patch.set_visible(False);continue
            tr=active[j];centers=np.asarray(tr['centers']);a,b=tr['axes'][0]
            line.set_data(*centers.T);patch.set_visible(True);patch.center=centers[0]
            scale=1.+.12/b;patch.width=2*a*scale;patch.height=2*b*scale;patch.angle=np.degrees(tr['theta'])
        ax.set_title('t=%.1f s | orange: actual obstacle; red: robot 0 prediction'%times[index])
        return paths+bodies+actors+predicted+envelopes
    animation=FuncAnimation(fig,update,frames=indices,interval=80,blit=False)
    animation.save(str(p/'dynamic_trajectory.gif'),writer=PillowWriter(fps=12),dpi=85);plt.close(fig)
    print('Dynamic diagnostics and GIF saved:',p)


if __name__=='__main__':main()
