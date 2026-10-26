#!/usr/bin/env python3
"""Scientific plots/GIF from recorded Gazebo truth; no ROS or control commands."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.patches import Circle,Rectangle
from matplotlib.transforms import Affine2D


def plot_peer_markers(directory):
    """Plot recorded ROS markers, not a screenshot of an RViz GUI session."""
    path=directory/'peer_marker_samples.json'
    if not path.exists():return
    samples=json.loads(path.read_text())
    if not samples:return
    observer='0' if '0' in samples else next(iter(samples))
    markers=samples[observer]
    styles={'peer_raw':('#df9b00',1.8,1.,'Laser-fit contour'),
            'peer_body':('#00995e',1.2,1.,'Effective body envelope'),
            'peer_safe':('#00995e',1.,.5,'CBF h=0 boundary'),
            'peer_future':('#307ec8',1.,.45,'Predicted h=0 boundary'),
            'peer_path':('#307ec8',1.,.65,'Predicted center path')}
    fig,ax=plt.subplots(figsize=(8,7));used=set()
    for marker in markers:
        if marker['namespace'] in styles and marker['points']:
            color,width,alpha,label=styles[marker['namespace']]
            xy=np.asarray(marker['points'])
            ax.plot(xy[:,0],xy[:,1],color=color,lw=width,alpha=alpha,
                    label=label if label not in used else None)
            used.add(label)
        if marker['namespace']=='peer_label':
            ax.text(*marker['position'],'tb3_'+str(marker['id']),fontsize=9,ha='center')
    ax.set_title('Recorded ROS peer markers: observer tb3_'+observer+' (not an RViz screenshot)')
    ax.set_xlabel('world x (m)');ax.set_ylabel('world y (m)')
    ax.set_aspect('equal');ax.grid(alpha=.25);ax.legend(loc='best',fontsize=8)
    fig.tight_layout();fig.savefig(directory/'peer_envelopes_ros_markers.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--gif',action='store_true')
    args=parser.parse_args();p=Path(args.directory)
    plot_peer_markers(p)
    result=json.loads((p/'evaluation.json').read_text())
    rows=list(csv.DictReader((p/'gazebo_truth.csv').open()))
    origin=result['leader_start_ros']
    rows=[r for r in rows if 0<=float(r['ros_time'])-origin<=result['duration']]
    t=np.array([float(r['ros_time'])-origin for r in rows])
    position=np.array([[[float(r['x'+str(i)]),float(r['y'+str(i)])] for r in rows] for i in range(5)])
    leader=np.column_stack((2.5*np.sin(t/15),.3+2.5*np.sin(t/30)))
    offsets=.4*np.column_stack((np.cos(np.arange(5)*2*np.pi/5),np.sin(np.arange(5)*2*np.pi/5)))
    errors=position-leader-offsets[:,None,:]
    metrics=[]
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    controls,angles=[],[]
    for i in range(5):
        for j in range(2):axes[j].plot(t,errors[i,:,j],label='Robot '+str(i),lw=1.)
        norm=np.linalg.norm(errors[i],axis=1);late=t>=145
        metrics.append(dict(robot=i,mean_error_m=float(norm.mean()),max_error_m=float(norm.max()),
            final_error_m=float(norm[-1]),late_x_rms_m=float(np.sqrt(np.mean(errors[i,late,0]**2))),
            late_y_rms_m=float(np.sqrt(np.mean(errors[i,late,1]**2)))))
    axes[0].set_ylabel('x consensus error (m)');axes[1].set_ylabel('y consensus error (m)')
    for ax in axes:ax.grid(alpha=.3);ax.axhline(0,color='black',lw=.5)
    axes[0].legend(ncol=5,fontsize=8);axes[1].set_xlabel('Leader trajectory time (s)')
    fig.tight_layout();fig.savefig(p/'gazebo_consensus_errors.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for i in range(5):
        data=list(csv.DictReader((p/('tb3_'+str(i)+'.csv')).open()))
        times=np.array([float(r['time']) for r in data]);v=np.array([float(r['v']) for r in data]);w=np.array([float(r['omega']) for r in data])
        theta=np.array([float(r['theta']) for r in data])
        leader_theta=np.arctan2((2.5/30)*np.cos(times/30),(2.5/15)*np.cos(times/15))
        difference=(theta-leader_theta+np.pi)%(2*np.pi)-np.pi
        axes[0].plot(times,v,label='Robot '+str(i),lw=.8)
        axes[1].plot(times,w,lw=.8);axes[2].plot(times,difference,lw=.8)
    axes[0].set_ylabel('v (m/s)');axes[1].set_ylabel('omega (rad/s)');axes[2].set_ylabel('heading error (rad)')
    axes[2].set_ylim(-4,4);axes[2].set_xlabel('Leader trajectory time (s)')
    for ax in axes:ax.grid(alpha=.25)
    axes[0].legend(ncol=5,fontsize=8);fig.tight_layout();fig.savefig(p/'gazebo_controls.png',dpi=160);plt.close(fig)
    (p/'truth_tracking_metrics.json').write_text(json.dumps(metrics,indent=2))
    if args.gif:
        fig,ax=plt.subplots(figsize=(7,6));ax.set_xlim(-3.8,3.7);ax.set_ylim(-3,3.6)
        for x,y,radius in [(1.7,0,.3),(-3,2,.4),(.5,2.6,.15)]:ax.add_patch(Circle((x,y),radius,color='gray',alpha=.5))
        rectangle=Rectangle((-.5,-.4),1,.8,color='gray',alpha=.5)
        rectangle.set_transform(Affine2D().rotate_deg(22.5).translate(1,-1.5)+ax.transData);ax.add_patch(rectangle)
        lines=[ax.plot([],[],lw=1.,label='Robot '+str(i))[0] for i in range(5)]
        bodies=[]
        for line in lines:
            patch=Circle((0,0),.13,color=line.get_color(),alpha=.8);ax.add_patch(patch);bodies.append(patch)
        leader_line,=ax.plot([],[],'k--',lw=.7,label='Leader');label=ax.text(.02,.97,'',transform=ax.transAxes,va='top')
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=7)
        def frame(k):
            for i,line in enumerate(lines):
                line.set_data(position[i,:k+1,0],position[i,:k+1,1]);bodies[i].center=position[i,k]
            leader_line.set_data(leader[:k+1,0],leader[:k+1,1]);label.set_text('t = %.1f s'%t[k])
            return lines+bodies+[leader_line,label]
        animation=FuncAnimation(fig,frame,frames=list(range(0,len(t),10))+[len(t)-1],interval=70,blit=False)
        animation.save(str(p/'gazebo_trajectory.gif'),writer=PillowWriter(fps=14),dpi=90)
        plt.close(fig)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
