#!/usr/bin/env python3
"""CBF experiment recording/export. World geometry is used only for plotting."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import pickle
import threading
import time

import numpy as np


ORIGINAL_CHARTS=(
    ('trajectory','Formation trajectory'),('linear_velocity','Linear velocity'),
    ('angular_velocity','Angular velocity'),('x_error','X consensus error'),
    ('y_error','Y consensus error'),('heading_error','Heading error'),
    ('time_analysis','Computation time'),('obstacle_distance','Obstacle distance'),
    ('clustering_analysis','Clustering analysis'),('avoidance_forces','Avoidance forces'),
    ('observer_estimation','Observer estimation'),('tracking_error_xy','XY tracking error'))


def aligned_centroid(robots):
    valid=[r for r in robots.values() if r is not None and r['t']]
    if len(valid)<3:return None
    start=max(min(r['t']) for r in valid);end=min(max(r['t']) for r in valid)
    times=np.unique([t for t in valid[0]['t'] if start<=t<=end])
    if not len(times):return None
    centers=[]
    for axis in ('xc','yc'):
        values=[]
        for r in valid:
            order=np.argsort(r['t'])
            t,index=np.unique(np.asarray(r['t'])[order],return_index=True)
            values.append(np.interp(times,t,np.asarray(r[axis])[order][index]))
        centers.append(np.mean(values,axis=0).tolist())
    return dict(t=times.tolist(),xc=centers[0],yc=centers[1])


def chart_manifest(stamp,robots,leader):
    valid=[r for r in robots.values() if r is not None]
    has=lambda field:any(np.isfinite(r.get(field,[])).any() for r in valid)
    manifest={name:dict(file=name+'_'+stamp+'.png',title=title,status='available')
              for name,title in ORIGINAL_CHARTS}
    manifest['avoidance_forces'].update(status='not_applicable',
        note='MPC-D-CBF has no CAPA avoidance force. The compatibility chart states N/A; no zero force is fabricated.')
    if not has('obstacle_distance'):
        manifest['obstacle_distance'].update(status='partial' if has('raw_min_distance') else 'unavailable',
            note='Historical reference-to-point distance was not recorded. Available laser-origin ranges are shown separately.')
    if leader is None:
        manifest['observer_estimation'].update(status='partial',note='Observer traces exist; historical leader trace unavailable.')
    if not has('thetae'):manifest['heading_error'].update(status='unavailable',note='Historical heading error was not recorded.')
    return manifest


def create_run_directory(package, requested=''):
    timestamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d-%H-%M-%S')
    if requested:
        path=Path(requested).expanduser().resolve()
        path.mkdir(parents=True,exist_ok=True)
        if any(path.glob('tb3_*.csv')):
            raise FileExistsError('Existing robot CSV files: choose a new output_dir: '+str(path))
        return path,timestamp
    root=Path(package)/'data_collect';root.mkdir(parents=True,exist_ok=True)
    for index in range(1000):
        suffix='' if index==0 else '-%02d'%index
        path=root/(timestamp+suffix+'-CBF-结果')
        try:
            path.mkdir()
            return path,timestamp+suffix
        except FileExistsError:
            continue
    raise RuntimeError('Could not create a unique experiment directory')


def update_geometry_ranges(stats, rows):
    for row in rows:
        key='%s/%s/%s'%(row['observer'],row['kind'],row['id'])
        item=stats.setdefault(key,dict(samples=0,laser_samples=0))
        item['samples']+=1;item['laser_samples']+=row['source']=='laser'
        for field in ('raw_a','raw_b','effective_a','effective_b'):
            value=row.get(field)
            if value is not None and np.isfinite(value):
                item[field+'_min']=min(item.get(field+'_min',value),value)
                item[field+'_max']=max(item.get(field+'_max',value),value)


def export_results(directory, robot_rows=None, leader_rows=None, geometry_stats=None):
    """Export the legacy-compatible pickle and CBF-specific plots from saved data."""
    path=Path(directory)
    metadata=json.loads((path/'run_metadata.json').read_text())
    stamp=metadata['timestamp']
    if robot_rows is None:
        robot_rows={i:list(csv.DictReader((path/('tb3_%d.csv'%i)).open()))
                    if (path/('tb3_%d.csv'%i)).exists() else [] for i in range(5)}
    if leader_rows is None:
        file=path/'leader.csv'
        leader_rows=list(csv.DictReader(file.open())) if file.exists() else []
    rows={i:[{k:float(v) if k not in ('robot','status','guide_obstacle','ipopt_status','solver_seed') and v not in ('',None)
                        else v for k,v in row.items()} for row in robot_rows.get(i,[])] for i in range(5)}
    leader=None
    if leader_rows:
        keys=('t','x0','y0','theta0','v0','w0','u0x','u0y')
        leader={k:[float(r['trajectory_time' if k=='t' else k]) for r in leader_rows] for k in keys}
    robots={};statistics={}
    for i,data in rows.items():
        if not data:
            robots[i]=None;continue
        robot={k:[] for k in ('t','xc','yc','thetac','vc','wc','xe','ye','thetae','xr','yr',
            'zxeo_x','zxeo_y','computation_time','full_cycle_time','raw_min_distance',
            'obstacle_distance','total_clusters_count','avoidance_force_magnitude',
            'static_clusters_count','peer_clusters_count')}
        for r in data:
            values=dict(t=r['time'],xc=r['x'],yc=r['y'],thetac=r['theta'],vc=r['v'],wc=r['omega'],
                xe=r['consensus_error_x'],ye=r['consensus_error_y'],thetae=r.get('heading_error',np.nan),
                xr=r.get('reference_x',r['x']-r['consensus_error_x']),
                yr=r.get('reference_y',r['y']-r['consensus_error_y']),
                zxeo_x=r['observer_x'],zxeo_y=r['observer_y'],
                computation_time=r['solve_ms']/1000.,full_cycle_time=r['solve_ms']/1000.,
                raw_min_distance=r.get('raw_min_range',np.nan),
                obstacle_distance=r.get('closest_obstacle_distance',np.nan),
                total_clusters_count=r.get('total_clusters_count',r['detected_ellipses']),
                static_clusters_count=r['detected_ellipses'],
                peer_clusters_count=r.get('detected_peer_ellipses',np.nan),avoidance_force_magnitude=np.nan)
            for k,value in values.items():robot[k].append(value)
        robots[i]=robot
        statuses={s:sum(r['status']==s for r in data) for s in sorted({r['status'] for r in data})}
        errors=np.hypot(robot['xe'],robot['ye'])
        milliseconds=np.array([r['solve_ms'] for r in data])
        period_ms=1000.*metadata.get('control_dt',.1)
        statistics[str(i)]=dict(rows=len(data),statuses=statuses,mean_position_error=float(errors.mean()),
            max_position_error=float(errors.max()),final_position_error=float(errors[-1]),
            mean_computation_ms=float(milliseconds.mean()),median_computation_ms=float(np.median(milliseconds)),
            std_computation_ms=float(milliseconds.std()),min_computation_ms=float(milliseconds.min()),
            max_computation_ms=float(milliseconds.max()),overruns=int(np.sum(milliseconds>period_ms)),
            overrun_percent=float(100.*np.mean(milliseconds>period_ms)))
    if geometry_stats is None:
        geometry_stats={};file=path/'geometry_estimates.csv'
        if file.exists():
            for r in csv.DictReader(file.open()):
                for k in ('raw_a','raw_b','effective_a','effective_b'):r[k]=float(r[k]) if r[k] else None
                update_geometry_ranges(geometry_stats,[r])
    categories=chart_manifest(stamp,robots,leader)
    dataset=dict(run_time=stamp,save_dir=str(path),leader=leader,robots=robots,formation_centroid=aligned_centroid(robots),
        metadata=metadata,cbf_diagnostics=rows,geometry_statistics=geometry_stats,
        chart_categories=categories,
        reference_note='Controller nominal reference; independent Gazebo truth evaluation is separate.')
    with (path/('consensus_data_'+stamp+'.pkl')).open('wb') as stream:pickle.dump(dataset,stream)
    (path/'experiment_statistics.json').write_text(json.dumps(statistics,indent=2,ensure_ascii=False))
    (path/'geometry_statistics.json').write_text(json.dumps(geometry_stats,indent=2,ensure_ascii=False))
    (path/'chart_categories.json').write_text(json.dumps(categories,indent=2,ensure_ascii=False))
    lines=['CBF experiment '+stamp,'Data: '+str(path),
        'Errors use the controller nominal reference; unavailable historical fields are NaN.',
        'CAPA avoidance force: not applicable. Controller duration is distinct from the scheduling interval.',
        'Control period: %.1f ms'%(1000.*metadata.get('control_dt',.1))]
    for i,s in statistics.items():
        lines.append('tb3_%s: rows=%d, mean/max/final error=%.5f/%.5f/%.5f m, compute=%.2f ms, status=%s'%
            (i,s['rows'],s['mean_position_error'],s['max_position_error'],s['final_position_error'],
             s['mean_computation_ms'],s['statuses']))
        lines.append('  timing median/std/min/max=%.2f/%.2f/%.2f/%.2f ms; overruns=%d (%.2f%%)'%
            (s['median_computation_ms'],s['std_computation_ms'],s['min_computation_ms'],
             s['max_computation_ms'],s['overruns'],s['overrun_percent']))
    (path/('computation_stats_'+stamp+'.txt')).write_text('\n'.join(lines)+'\n')
    plot_results(path,stamp,robots,leader,geometry_stats,metadata)
    return dataset


def plot_results(path,stamp,robots,leader,geometry_stats,metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Rectangle
    from matplotlib.transforms import Affine2D
    import xml.etree.ElementTree as ET
    valid={i:r for i,r in robots.items() if r is not None}
    if not valid:return
    fig,ax=plt.subplots(figsize=(8,7))
    for i,r in valid.items():
        line,=ax.plot(r['xc'],r['yc'],label='Robot '+str(i),lw=1.2)
        ax.plot(r['xr'],r['yr'],'--',color=line.get_color(),lw=.7,alpha=.5)
    centroid=aligned_centroid(robots)
    if centroid:ax.plot(centroid['xc'],centroid['yc'],'k:',lw=1.,label='Formation center (time aligned)')
    if leader:ax.plot(leader['x0'],leader['y0'],'k--',lw=1.,label='Leader history')
    world=Path(metadata.get('world_file',''))
    if world.is_file():
        for link in ET.parse(world).findall('.//model[@name="obstacles"]/link'):
            pose=[float(v) for v in link.findtext('pose').split()];g=link.find('collision/geometry')
            if g.find('cylinder') is not None:patch=Circle(pose[:2],float(g.findtext('cylinder/radius')),color='gray',alpha=.4)
            elif g.find('box') is not None:
                a,b,_=[float(v) for v in g.findtext('box/size').split()]
                patch=Rectangle((-a/2,-b/2),a,b,color='gray',alpha=.4)
                patch.set_transform(Affine2D().rotate(pose[5]).translate(*pose[:2])+ax.transData)
            else:continue
            ax.add_patch(patch)
    ax.set_aspect('equal');ax.grid(alpha=.25);ax.legend(fontsize=8)
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');fig.tight_layout()
    fig.savefig(path/('trajectory_'+stamp+'.png'),dpi=150);plt.close(fig)
    plots=[('x_error','xe','x consensus error (m)'),('y_error','ye','y consensus error (m)'),
           ('heading_error','thetae','heading error (rad)'),('linear_velocity','vc','v (m/s)'),
           ('angular_velocity','wc','omega (rad/s)'),('time_analysis','computation_time','control computation (ms)')]
    for name,key,ylabel in plots:
        fig,ax=plt.subplots(figsize=(9,3.6))
        for i,r in valid.items():
            values=np.asarray(r[key])*(1000. if key=='computation_time' else 1.)
            ax.plot(r['t'],values,label='Robot '+str(i),lw=.9)
        if key=='thetae':ax.set_ylim(-4,4)
        if key=='computation_time':
            period=1000.*metadata.get('control_dt',.1)
            ax.axhline(period,color='red',ls='--',lw=1.,label='Control period (%.0f ms)'%period)
            upper=max(period*2,max(max(r[key])*1000. for r in valid.values())*1.1)
            ax.axhspan(period,upper,color='red',alpha=.07);ax.set_ylim(0,upper)
        if not any(np.isfinite(r[key]).any() for r in valid.values()):
            ax.text(.5,.5,'Historical data unavailable',ha='center',transform=ax.transAxes)
        ax.set_xlabel('trajectory time (s)');ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend(ncol=5,fontsize=8)
        fig.tight_layout();fig.savefig(path/(name+'_'+stamp+'.png'),dpi=150);plt.close(fig)
    # Keep the original result categories, with honest handling of quantities
    # that do not exist in MPC-D-CBF or were absent from historical recordings.
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for ax,key,label in zip(axes,('obstacle_distance','raw_min_distance'),
            ('Nominal reference to measured obstacle point (m)','Laser-origin minimum range (m)')):
        for i,r in valid.items():ax.plot(r['t'],r[key],label='Robot '+str(i),lw=.9)
        if not any(np.isfinite(r[key]).any() for r in valid.values()):
            ax.text(.5,.5,'Not recorded in this historical run',ha='center',transform=ax.transAxes)
        ax.set_ylabel(label,fontsize=9);ax.grid(alpha=.25);ax.legend(ncol=5,fontsize=7)
    axes[-1].set_xlabel('trajectory time (s)');fig.tight_layout()
    fig.savefig(path/('obstacle_distance_'+stamp+'.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for i,r in valid.items():
        axes[0].step(r['t'],r['total_clusters_count'],where='post',label='Robot '+str(i),lw=1.)
        line,=axes[1].step(r['t'],r['static_clusters_count'],where='post',label='Static '+str(i),lw=.8)
        axes[1].step(r['t'],r['peer_clusters_count'],where='post',color=line.get_color(),ls='--',
                     label='Fresh peer '+str(i),lw=.8)
    axes[0].set_ylabel('Total observed/tracked groups');axes[1].set_ylabel('Static / fresh peer groups')
    if not any(np.isfinite(r['peer_clusters_count']).any() for r in valid.values()):
        axes[0].set_title('Historical counts contain static clusters only; peer counts unavailable',fontsize=10)
    for ax in axes:ax.grid(alpha=.25);ax.legend(ncol=5,fontsize=7);ax.set_ylim(bottom=0)
    axes[-1].set_xlabel('trajectory time (s)');fig.tight_layout()
    fig.savefig(path/('clustering_analysis_'+stamp+'.png'),dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));ax.set_axis_off()
    ax.set_title('Avoidance forces: not applicable to MPC-D-CBF')
    ax.text(.5,.55,'This controller does not compute a CAPA avoidance force.\n'
            'Avoidance is enforced through MPC-D-CBF constraints.\n'
            'No zero-force curve is substituted.',ha='center',va='center',transform=ax.transAxes,fontsize=12)
    fig.tight_layout();fig.savefig(path/('avoidance_forces_'+stamp+'.png'),dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6))
    if leader:
        ax.plot(leader['t'],leader['x0'],'k--',lw=2.,label='Leader x (reported)')
        ax.plot(leader['t'],leader['y0'],color='darkred',ls='--',lw=2.,label='Leader y (reported)')
    else:ax.text(.5,.96,'Leader history unavailable; recorded observer estimates shown',ha='center',transform=ax.transAxes)
    for i,r in valid.items():
        line,=ax.plot(r['t'],r['zxeo_x'],label='x_est_'+str(i),lw=1.)
        ax.plot(r['t'],r['zxeo_y'],color=line.get_color(),ls='-.',label='y_est_'+str(i),lw=1.)
    ax.set_xlabel('trajectory time (s)');ax.set_ylabel('position (m)');ax.grid(alpha=.25)
    ax.set_title('Observer estimation');ax.legend(ncol=3,fontsize=7);fig.tight_layout()
    fig.savefig(path/('observer_estimation_'+stamp+'.png'),dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for i,r in valid.items():
        ax.plot(r['t'],np.asarray(r['xr'])-r['xc'],color='red',lw=.9,alpha=.65,label='x_error_'+str(i))
        ax.plot(r['t'],np.asarray(r['yr'])-r['yc'],color='blue',lw=.9,alpha=.65,label='y_error_'+str(i))
    ax.axhline(0,color='black',ls=':',lw=.8);ax.set_xlabel('trajectory time (s)')
    ax.set_ylabel('tracking error (m)');ax.set_title('XY tracking error (xr-xc, yr-yc)')
    ax.legend(ncol=2,fontsize=7);ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(path/('tracking_error_xy_'+stamp+'.png'),dpi=150);plt.close(fig)
    if geometry_stats:
        keys=sorted(geometry_stats)
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        for j,field in enumerate(('a','b')):
            for offset,prefix,color,label in [(-.12,'raw_','#dfa500','Measured'),(.12,'effective_','#009966','Effective')]:
                lo=np.array([geometry_stats[k].get(prefix+field+'_min',np.nan) for k in keys])
                hi=np.array([geometry_stats[k].get(prefix+field+'_max',np.nan) for k in keys])
                axes[j].vlines(np.arange(len(keys))+offset,lo,hi,color=color,label=label)
                axes[j].scatter(np.arange(len(keys))+offset,lo,s=10,color=color)
            axes[j].set_ylabel('semi-axis '+field+' range (m)');axes[j].grid(alpha=.2);axes[j].legend(fontsize=8)
        axes[-1].set_xticks(range(len(keys)));axes[-1].set_xticklabels(keys,rotation=90,fontsize=6)
        fig.tight_layout();fig.savefig(path/('ellipse_axes_'+stamp+'.png'),dpi=150);plt.close(fig)


class ExperimentRecorder:
    GEOMETRY_FIELDS=['ros_time','observer','kind','id','source','raw_a','raw_b','effective_a',
                     'effective_b','point_count','measurement_age','size_floor_dominates']

    def __init__(self):
        import rospy,rospkg
        from std_msgs.msg import Bool,String
        from tb3_distributed_control.msg import LeaderHistoryData
        self.rospy=rospy;self.lock=threading.Lock();self.finish_lock=threading.RLock()
        self.requested=threading.Event();self.finished=False
        package=Path(rospkg.RosPack().get_path('tb3_distributed_control'))
        self.path,stamp=create_run_directory(package,rospy.get_param('/cbf/output_dir',''))
        self.run_id=rospy.get_param('/cbf/run_id')
        self.key='/cbf/results/'+self.run_id
        self.robots={i:[] for i in range(5)};self.leader=[];self.geometry_stats={}
        self.meta=dict(timestamp=stamp,run_id=self.run_id,output_dir=str(self.path),
            timezone='Asia/Shanghai',method='MPC-D-CBF',
            world_file=rospy.get_param('/cbf/world_file',''),
            formation_radius=rospy.get_param('/cbf/formation_radius',.4),
            control_dt=rospy.get_param('/cbf/mpc/dt',.1),
            realtime_solver=rospy.get_param('/cbf/realtime_solver',True),
            control_budget_ms=rospy.get_param('/cbf/control_budget_ms',100.),
            nlp_budget_ms=rospy.get_param('/cbf/nlp_budget_ms',65.),
            dual_warm_start=rospy.get_param('/cbf/dual_warm_start',True),
            stagger_callbacks=rospy.get_param('/cbf/stagger_callbacks',True),
            trajectory_type=rospy.get_param('/tb3_virtual_leader/leader_node/trajectory_type','unknown'))
        (self.path/'run_metadata.json').write_text(json.dumps(self.meta,indent=2,ensure_ascii=False))
        self.leader_stream=(self.path/'leader.csv').open('w',newline='')
        self.leader_writer=csv.DictWriter(self.leader_stream,fieldnames=
            ['trajectory_time','ros_time','x0','y0','theta0','v0','w0','u0x','u0y'])
        self.leader_writer.writeheader()
        self.geometry_stream=(self.path/'geometry_estimates.csv').open('w',newline='')
        self.geometry_writer=csv.DictWriter(self.geometry_stream,fieldnames=self.GEOMETRY_FIELDS)
        self.geometry_writer.writeheader();self.last_flush=time.monotonic()
        self.subscribers=[rospy.Subscriber('/leader/history',LeaderHistoryData,self.on_leader,queue_size=100),
            rospy.Subscriber('/system/shutdown',Bool,lambda m:self.requested.set() if m.data else None)]
        for i in range(5):
            rospy.set_param('/cbf/ready/tb3_'+str(i),False)
            self.subscribers.append(rospy.Subscriber('/tb3_%d/cbf_diagnostics'%i,String,self.on_robot,callback_args=i,queue_size=100))
            self.subscribers.append(rospy.Subscriber('/tb3_%d/cbf_geometry/estimates'%i,String,self.on_geometry,queue_size=100))
        rospy.on_shutdown(self.finish)
        # Publish only after files and subscriptions exist. A unique launch token
        # prevents the next experiment reusing stale readiness/output parameters.
        rospy.set_param(self.key+'/directory',str(self.path))
        rospy.loginfo('CBF experiment results: %s (CSV live; pickle/plots at completion or Ctrl+C)',self.path)

    def on_leader(self,msg):
        row={k:getattr(msg,k) for k in ('trajectory_time','x0','y0','theta0','v0','w0','u0x','u0y')}
        row['ros_time']=msg.ros_timestamp
        with self.lock:
            if not self.finished:self.leader.append(row);self.leader_writer.writerow(row)

    def on_robot(self,msg,i):
        row=json.loads(msg.data)
        with self.lock:
            if not self.finished:self.robots[i].append(row)

    def on_geometry(self,msg):
        rows=json.loads(msg.data)['rows']
        with self.lock:
            if self.finished:return
            self.geometry_writer.writerows(rows);update_geometry_ranges(self.geometry_stats,rows)
            if time.monotonic()-self.last_flush>1.:
                self.leader_stream.flush();self.geometry_stream.flush();self.last_flush=time.monotonic()

    def finish(self):
        with self.finish_lock:
            if self.finished:return
            with self.lock:
                self.finished=True
                self.leader_stream.flush();self.leader_stream.close()
                self.geometry_stream.flush();self.geometry_stream.close()
                robots={i:list(rows) for i,rows in self.robots.items()};leader=list(self.leader)
            try:
                export_results(self.path,robots,leader,self.geometry_stats)
                (self.path/'RESULTS_SAVED').write_text('CSV, pickle, plots and statistics exported.\n')
                try:self.rospy.set_param(self.key+'/saved',True)
                except Exception:pass  # The ROS master may already be closing on Ctrl+C.
                self.rospy.loginfo('CBF results saved: %s',self.path)
            except Exception as exc:
                self.rospy.logerr('CBF result export failed; raw CSV files retained in %s: %s',self.path,exc)
                (self.path/'export_error.txt').write_text(str(exc)+'\n')


if __name__=='__main__':
    import rospy
    rospy.init_node('cbf_experiment_results')
    recorder=ExperimentRecorder()
    while not rospy.is_shutdown() and not recorder.requested.wait(.1):pass
    if not rospy.is_shutdown():time.sleep(.5)
    recorder.finish()
