#!/usr/bin/env python3
"""CBF experiment recording/export. World geometry is used only for plotting."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import pickle
import threading
import time
import sys

import numpy as np

_scripts=Path(__file__).resolve().parent
for _directory in (_scripts,_scripts.parent/'纯仿真代码'):
    if _directory.is_dir() and str(_directory) not in sys.path:sys.path.insert(0,str(_directory))
from cbf_mpc_core import metric_definitions,METRICS_SCHEMA_VERSION


ORIGINAL_CHARTS=(
    ('trajectory','Formation trajectory'),('linear_velocity','Linear velocity'),
    ('angular_velocity','Angular velocity'),('x_error','X consensus error'),
    ('y_error','Y consensus error'),('heading_error','Heading error'),
    ('time_analysis','Computation time'),('obstacle_distance','Obstacle distance'),
    ('clustering_analysis','Clustering analysis'),('avoidance_forces','Avoidance forces'),
    ('observer_estimation','Observer estimation'),('tracking_error_xy','XY tracking error'))

# Match visualize_results.py: dense timing traces intentionally remain thinner.
PLOT_STYLE=dict(curve_width=2.,secondary_width=1.5,timing_width=.8,
    leader_width=2.5,snapshot_width=1.5,dpi=300,grid_alpha=.3)
ROBOT_COLORS=('#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd')


def _save_plot(fig,path,name,stamp):
    import matplotlib.pyplot as plt
    for extension in ('png','pdf'):
        fig.savefig(Path(path)/(name+'_'+stamp+'.'+extension),
            dpi=PLOT_STYLE['dpi'],bbox_inches='tight')
    plt.close(fig)


def _time_axis(ax,end,ylabel,title):
    from matplotlib.ticker import MaxNLocator
    end=max(float(end),1.)
    ticks=MaxNLocator(nbins=10).tick_values(0,end)
    ticks=ticks[(ticks>=0)&(ticks<=end)]
    # Include the actual end without nearly overlapping the preceding tick.
    if len(ticks) and not np.isclose(ticks[-1],end):
        if len(ticks)>1 and end-ticks[-1]<.35*(ticks[-1]-ticks[-2]):ticks=ticks[:-1]
        ticks=np.append(ticks,end)
    ax.set_xticks(ticks);ax.set_xlim(0,end)
    ax.set_xlabel('Trajectory Time (s)',fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12);ax.set_title(title,fontsize=14)
    ax.set_axisbelow(True);ax.grid(True,alpha=PLOT_STYLE['grid_alpha'])


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
    if leader is None and not (has('observer_aligned_x') and has('leader_reference_x')):
        manifest['observer_estimation'].update(status='partial',note='Observer traces exist; historical leader trace unavailable.')
    if not has('thetae'):manifest['heading_error'].update(status='unavailable',note='Historical heading error was not recorded.')
    if not has('plan_execution_error_x'):
        manifest['tracking_error_xy'].update(status='unavailable',note='Accepted auxiliary-plan execution reference was not recorded; nominal reference is not substituted.')
    return manifest


def finite_error_summary(values):
    values=np.asarray(values,float);valid=values[np.isfinite(values)]
    return dict(valid_samples=int(len(valid)),missing_samples=int(len(values)-len(valid)),
        mean=float(np.mean(valid)) if len(valid) else np.nan,
        mean_signed=float(np.mean(valid)) if len(valid) else np.nan,
        mae=float(np.mean(np.abs(valid))) if len(valid) else np.nan,
        rmse=float(np.sqrt(np.mean(valid**2))) if len(valid) else np.nan,
        max_abs=float(np.max(np.abs(valid))) if len(valid) else np.nan,
        final=float(values[-1]) if len(values) and np.isfinite(values[-1]) else np.nan,
        final_abs=float(abs(values[-1])) if len(values) and np.isfinite(values[-1]) else np.nan)


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
        robot={k:[] for k in ('t','control_t','xc','yc','thetac','vc','wc','xe','ye','thetae','xr','yr',
            'zxeo_x','zxeo_y','computation_time','full_cycle_time','raw_min_distance',
            'obstacle_distance','total_clusters_count','avoidance_force_magnitude',
            'static_clusters_count','dynamic_clusters_count','peer_clusters_count',
            'plan_reference_x','plan_reference_y','plan_execution_error_x','plan_execution_error_y',
            'observer_aligned_x','observer_aligned_y','observer_error_x','observer_error_y',
            'leader_reference_x','leader_reference_y','reference_quality','plan_reference_valid')}
        for r in data:
            values=dict(t=r.get('evaluation_time',r['time']),control_t=r['time'],xc=r['x'],yc=r['y'],thetac=r['theta'],vc=r['v'],wc=r['omega'],
                xe=r['consensus_error_x'],ye=r['consensus_error_y'],thetae=r.get('heading_error',np.nan),
                xr=r.get('reference_x',r['x']-r['consensus_error_x']),
                yr=r.get('reference_y',r['y']-r['consensus_error_y']),
                zxeo_x=r['observer_x'],zxeo_y=r['observer_y'],
                computation_time=r['solve_ms']/1000.,full_cycle_time=r['solve_ms']/1000.,
                raw_min_distance=r.get('raw_min_range',np.nan),
                obstacle_distance=r.get('closest_obstacle_distance',np.nan),
                total_clusters_count=r.get('total_clusters_count',r['detected_ellipses']),
                static_clusters_count=r.get('detected_static_ellipses',r['detected_ellipses']),
                dynamic_clusters_count=r.get('detected_dynamic_ellipses',np.nan),
                peer_clusters_count=r.get('detected_peer_ellipses',np.nan),avoidance_force_magnitude=np.nan)
            for key in ('plan_reference_x','plan_reference_y','plan_execution_error_x','plan_execution_error_y',
                    'observer_aligned_x','observer_aligned_y','observer_error_x','observer_error_y',
                    'leader_reference_x','leader_reference_y','reference_quality','plan_reference_valid'):
                values[key]=r.get(key,np.nan)
            for k,value in values.items():robot[k].append(value)
        robots[i]=robot
        statuses={s:sum(r['status']==s for r in data) for s in sorted({r['status'] for r in data})}
        errors=np.hypot(robot['xe'],robot['ye'])
        milliseconds=np.array([r['solve_ms'] for r in data])
        period_ms=1000.*metadata.get('control_dt',.1)
        eval_times=np.asarray(robot['t']);time_window=(eval_times>=0)&np.isfinite(eval_times)
        if metadata.get('total_time') is not None:time_window&=eval_times<=float(metadata['total_time'])+1e-9
        stamps=np.asarray([r.get('evaluation_ros_time',r.get('ros_time',r['time'])-r.get('odom_age',0.)) for r in data])
        fresh=np.asarray([abs(r.get('odom_age',0.))<=float(metadata.get('data_timeout',.6)) for r in data])
        last_stamp=-np.inf
        for j,t in enumerate(stamps):
            if not np.isfinite(t) or t<=last_stamp+1e-9:fresh[j]=False
            elif fresh[j]:last_stamp=t
        window=time_window&fresh
        position_summary=finite_error_summary(errors[window])
        quality=np.asarray(robot['reference_quality'])[window]
        statistics[str(i)]=dict(rows=len(data),statuses=statuses,mean_position_error=position_summary['mean'],
            max_position_error=position_summary['max_abs'],final_position_error=position_summary['final'],
            evaluated_rows=int(window.sum()),excluded_window_rows=int((~time_window).sum()),
            excluded_measurement_rows=int(np.sum(time_window&~fresh)),
            reference_quality_counts={label:int(np.sum(quality==code)) for code,label in
                ((0,'unavailable'),(1,'interpolated'),(2,'bounded_extrapolation'),(3,'exact'))},
            position_error_summary=position_summary,heading_error_summary=finite_error_summary(np.asarray(robot['thetae'])[window]),
            plan_execution_summary=finite_error_summary(np.hypot(robot['plan_execution_error_x'],robot['plan_execution_error_y'])[window]),
            observer_error_summary=finite_error_summary(np.hypot(robot['observer_error_x'],robot['observer_error_y'])[window]),
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
        reference_note='xr/yr are nominal compatibility fields. Internal plan execution uses dedicated plan_reference/error fields; independent Gazebo truth is separate.')
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


def formation_snapshots(robots, times=None, max_gap=.6):
    """Interpolate measured positions at common times; never bridge stale gaps.

    Robot IDs fix the edge order, preserving actual formation deformations.
    Missing robots/times are skipped rather than replaced by ideal vertices.
    """
    samples=[]
    for i in range(5):
        robot=robots.get(i)
        if robot is None:return []
        data=np.column_stack([robot[k] for k in ('t','xc','yc')])
        data=data[np.isfinite(data).all(axis=1)]
        if not len(data):return []
        data=data[np.argsort(data[:,0],kind='stable')]
        _,indices=np.unique(data[:,0],return_index=True)
        samples.append(data[indices])
    start=max(data[0,0] for data in samples)
    end=min(data[-1,0] for data in samples)
    if start>end:return []
    if times is None:
        # Match the original six snapshots from 30 s to the common final time.
        first=max(start,30.) if end>30. else start
        times=np.unique(np.linspace(first,end,6 if end>30. else 3))
    result=[]
    for stamp in times:
        if not np.isfinite(stamp) or not start<=stamp<=end:continue
        positions=[]
        for data in samples:
            index=int(np.searchsorted(data[:,0],stamp))
            if index<len(data) and abs(data[index,0]-stamp)<1e-9:
                positions.append(data[index,1:].tolist())
            elif index==0 or index==len(data) or data[index,0]-data[index-1,0]>max_gap:
                break
            else:
                positions.append([float(np.interp(stamp,data[:,0],data[:,axis])) for axis in (1,2)])
        if len(positions)==5:result.append(dict(time=float(stamp),positions=positions))
    return result


def plot_trajectory(path,stamp,robots,leader,metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Affine2D
    import xml.etree.ElementTree as ET
    valid={i:r for i,r in robots.items() if r is not None}
    if not valid:return
    path=Path(path)
    fig,ax=plt.subplots(figsize=(8,6))
    for i,r in sorted(valid.items()):
        line,=ax.plot(r['xc'],r['yc'],label='Robot '+str(i)+' actual',
            color=ROBOT_COLORS[i%5],lw=PLOT_STYLE['curve_width'],zorder=3)
        ax.plot(r['xr'],r['yr'],'--',color=line.get_color(),lw=PLOT_STYLE['curve_width'],alpha=.7,zorder=2)
    centroid=aligned_centroid(robots)
    if centroid:ax.plot(centroid['xc'],centroid['yc'],color='deepskyblue',lw=PLOT_STYLE['curve_width'],label='Actual formation center',zorder=4)
    if leader:ax.plot(leader['x0'],leader['y0'],color='purple',ls='--',lw=PLOT_STYLE['curve_width'],alpha=.7,label='Leader reference',zorder=3)
    dynamic_file=path/'dynamic_truth.csv'
    if dynamic_file.exists():
        records=list(csv.DictReader(dynamic_file.open()))
        for name in sorted({r['name'] for r in records}):
            rows=[r for r in records if r['name']==name]
            if not rows:continue
            ax.plot([float(r['x']) for r in rows],[float(r['y']) for r in rows],
                ls='-.',lw=1.5,label=name+' prescribed path')
            for t in np.linspace(0,max(float(r['time']) for r in rows),4):
                r=min(rows,key=lambda r:abs(float(r['time'])-t))
                ax.add_patch(Circle((float(r['x']),float(r['y'])),float(r['radius']),
                    fill=False,edgecolor='darkorange',alpha=.5,lw=1.))
    snapshots=formation_snapshots(robots)
    snapshot_colors=('#3454a1','#3079ad','#007f79','#977300','#b65b20','#ad303d')
    for index,snapshot in enumerate(snapshots):
        positions=np.asarray(snapshot['positions'])
        closed=np.vstack((positions,positions[0]))
        color=snapshot_colors[index%len(snapshot_colors)]
        ax.plot(closed[:,0],closed[:,1],color=color,lw=PLOT_STYLE['snapshot_width'],alpha=.85,zorder=5)
        ax.scatter(positions[:,0],positions[:,1],s=25,color=color,edgecolors='white',linewidths=.4,zorder=6)
        ax.annotate('%.1f s'%snapshot['time'],positions.mean(axis=0),xytext=(4,5),
            textcoords='offset points',fontsize=8,color=color,zorder=7,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=1.5))
    world=Path(metadata.get('world_file',''))
    if world.is_file():
        for link in ET.parse(world).findall('.//model[@name="obstacles"]/link'):
            pose=[float(v) for v in link.findtext('pose').split()];g=link.find('collision/geometry')
            if g.find('cylinder') is not None:patch=Circle(pose[:2],float(g.findtext('cylinder/radius')),color='gray',alpha=.7)
            elif g.find('box') is not None:
                a,b,_=[float(v) for v in g.findtext('box/size').split()]
                patch=Rectangle((-a/2,-b/2),a,b,color='gray',alpha=.7)
                patch.set_transform(Affine2D().rotate(pose[5]).translate(*pose[:2])+ax.transData)
            else:continue
            ax.add_patch(patch)
    ax.autoscale_view();ax.set_aspect('equal');ax.set_axisbelow(True);ax.grid(alpha=PLOT_STYLE['grid_alpha'])
    handles,labels=ax.get_legend_handles_labels()
    handles.append(Line2D([],[],color='gray',ls='--',lw=PLOT_STYLE['curve_width']));labels.append('Nominal reference (robot colors)')
    if snapshots:
        handles.append(Line2D([],[],color='slategray',marker='o',lw=PLOT_STYLE['snapshot_width'],markersize=4))
        labels.append('Measured formation snapshots')
    ax.legend(handles,labels,fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.1))
    radius=metadata.get('formation_radius')
    title='MPC-D-CBF formation trajectory'
    if radius is not None:title+=' (nominal radius %.2f m)'%float(radius)
    ax.set_title(title,fontsize=12)
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');fig.tight_layout()
    _save_plot(fig,path,'trajectory',stamp)
    (path/('trajectory_snapshots_'+stamp+'.json')).write_text(json.dumps(
        dict(source='Time-aligned recorded positions, connected in robot ID order 0-1-2-3-4-0',
             formation_radius=radius,snapshots=snapshots),indent=2))


def plot_results(path,stamp,robots,leader,geometry_stats,metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid={i:r for i,r in sorted(robots.items()) if r is not None and len(r['t'])}
    if not valid:return
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    end=float(np.ceil(max(leader['t'] if leader and len(leader['t']) else
        [max(r['t']) for r in valid.values()])))
    width=PLOT_STYLE['curve_width'];secondary=PLOT_STYLE['secondary_width']
    color=lambda i:ROBOT_COLORS[i%5]
    plot_trajectory(path,stamp,robots,leader,metadata)
    # Original figures 2--6: same sizes, line widths, color IDs and speed limits.
    specs=(('linear_velocity','vc','Linear Velocity (m/s)','Linear Velocities','v',(-.3,.3),'v0'),
        ('angular_velocity','wc','Angular Velocity (rad/s)','Angular Velocities','omega',(-3.,3.),'w0'),
        ('x_error','xe','X Direction Error (m)','X Position Error','xe',None,None),
        ('y_error','ye','Y Direction Error (m)','Y Position Error','ye',None,None),
        ('heading_error','thetae','Heading Angle Error (rad)','Heading Error','thetae',(-4.,4.),None))
    for name,key,ylabel,title,prefix,limits,leader_key in specs:
        fig,ax=plt.subplots(figsize=(8,4))
        for i,r in valid.items():
            times=r.get('control_t',r['t']) if key in ('vc','wc') else r['t']
            ax.plot(times,r[key],label=prefix+'_'+str(i),color=color(i),lw=width)
        if leader_key and leader and len(leader.get(leader_key,[])):
            ax.plot(leader['t'],leader[leader_key],'--',color='#8c564b',lw=width,label=prefix+'_Leader')
        if limits:ax.set_ylim(*limits)
        if not any(np.isfinite(r[key]).any() for r in valid.values()):
            ax.text(.5,.5,'Historical data unavailable',ha='center',transform=ax.transAxes)
        _time_axis(ax,end,ylabel,title+' - '+stamp)
        if key=='thetae' and int(metadata.get('metrics_schema_version',1))<2:
            ax.set_title('Historical heading error: sign as recorded - '+stamp,fontsize=12)
        ax.legend();fig.tight_layout();_save_plot(fig,path,name,stamp)
    # Figure 7: wall duration, not a fabricated separate pure-optimizer duration.
    fig,ax=plt.subplots(figsize=(16,10));stats=[];all_times=[]
    period=1000.*metadata.get('control_dt',.1);maximum=period*2
    for i,r in valid.items():
        values=np.asarray(r['computation_time'])*1000.
        control_times=r.get('control_t',r['t'])
        mask=np.isfinite(values)&np.isfinite(control_times)
        t=np.asarray(control_times)[mask];values=values[mask]
        if not len(values):continue
        all_times.extend(values.tolist());maximum=max(maximum,float(values.max())*1.12)
        timing_color=('blue','orange','green','red','purple')[i%5]
        ax.plot(t,values,lw=PLOT_STYLE['timing_width'],color=timing_color,alpha=.7,
            label='Robot %d - Controller Wall Time'%i)
        peak=int(np.argmax(values))
        if values[peak]>80:
            ax.scatter(t[peak],values[peak],s=50,color=timing_color,edgecolors='black',linewidths=1,zorder=10)
            ax.annotate('R%d: %.1f ms'%(i,values[peak]),(t[peak],values[peak]),xytext=(0,7),
                textcoords='offset points',fontsize=8,ha='center',color=timing_color,weight='bold')
        stats.append('Robot %d: mean %.1f, median %.1f, P95 %.1f, max %.1f ms; overrun %d/%d'%
            (i,values.mean(),np.median(values),np.percentile(values,95),values.max(),
             np.sum(values>period),len(values)))
    ax.axhline(period,color='red',ls='-.',lw=3,alpha=.95,label='Control Period (%.0f ms)'%period,zorder=10)
    ax.axhspan(period,maximum,color='red',alpha=.1,label='Overrun Zone');ax.set_ylim(0,maximum)
    if all_times:
        values=np.asarray(all_times)
        stats.append('Overall: mean %.1f, P99 %.1f, max %.1f ms; control period %.0f ms'%
            (values.mean(),np.percentile(values,99),values.max(),period))
    stats.append('Recorded controller duration includes solving/validation; it is not the interval between commands.')
    _time_axis(ax,end,'Time (ms)','Time Analysis: Controller Wall Time - '+stamp)
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=9)
    fig.text(.02,.02,'\n'.join(stats),fontsize=9,family='monospace',va='bottom',
        bbox=dict(boxstyle='round,pad=0.5',facecolor='lightgray',alpha=.95))
    fig.subplots_adjust(bottom=.25,right=.75)
    _save_plot(fig,path,'time_analysis',stamp)
    # Figure 8: restore the single-axis main chart; raw laser range is separate.
    # Neither the CAPA .30/.35 m thresholds nor a distance smoothing filter is
    # carried into this figure: neither represents the D-CBF safety boundary.
    for name,key,ylabel,title in (
            ('obstacle_distance','obstacle_distance','Distance to Closest Obstacle Point (m)',
             'Nominal Reference to Measured Obstacle Point'),
            ('laser_min_range','raw_min_distance','Minimum Range from Laser Origin (m)',
             'Measured Laser Minimum Range (supplement)')):
        fig,ax=plt.subplots(figsize=(10,6))
        for i,r in valid.items():
            values=np.asarray(r[key],float).copy();values[~np.isfinite(values)]=np.nan
            ax.plot(r['t'],values,color=color(i),lw=secondary,alpha=.8,label='Robot '+str(i))
        if not any(np.isfinite(r[key]).any() for r in valid.values()):
            ax.text(.5,.5,'Not recorded in this historical run',ha='center',transform=ax.transAxes)
        ax.set_ylim(bottom=0);_time_axis(ax,end,ylabel,title)
        ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8,framealpha=.9)
        fig.tight_layout();_save_plot(fig,path,name,stamp)
    # Figure 9: total counts in the original single panel; breakdown is an extra.
    fig,ax=plt.subplots(figsize=(12,6));largest=0.
    for i,r in valid.items():
        counts=np.asarray(r['total_clusters_count'],float)
        ax.step(r['t'],counts,where='post',color=color(i),lw=secondary,alpha=.8,
            label='Robot %d - Total Groups'%i)
        finite=counts[np.isfinite(counts)]
        if len(finite):largest=max(largest,float(finite.max()))
    ax.set_ylim(0,max(8,largest+1))
    _time_axis(ax,end,'Number of Groups','Clustering Analysis - Total Observed/Tracked Groups')
    ax.legend(loc='upper right',fontsize=10,framealpha=.9)
    fig.tight_layout();_save_plot(fig,path,'clustering_analysis',stamp)
    breakdown=[('static_clusters_count','Static / Non-moving Groups')]
    if metadata.get('dynamic_prediction',False):breakdown.append(('dynamic_clusters_count','Dynamic Groups'))
    breakdown.append(('peer_clusters_count','Fresh Peer Groups'))
    fig,axes=plt.subplots(len(breakdown),1,figsize=(12,3*len(breakdown)),sharex=True)
    for i,r in valid.items():
        for ax,(key,_) in zip(axes,breakdown):
            ax.step(r['t'],r.get(key,[np.nan]*len(r['t'])),where='post',color=color(i),lw=secondary,alpha=.8,label='Robot '+str(i))
    for ax,(_,title) in zip(axes,breakdown):
        _time_axis(ax,end,'Number of Groups',title+' (supplement)')
        ax.set_ylim(bottom=0);ax.legend(ncol=5,fontsize=8)
    if not any(np.isfinite(r['peer_clusters_count']).any() for r in valid.values()):
        axes[-1].text(.5,.5,'Not recorded in this historical run',ha='center',transform=axes[-1].transAxes)
    fig.tight_layout();_save_plot(fig,path,'clustering_breakdown',stamp)
    # Figure 10 remains explicitly N/A; this method has no CAPA force variable.
    fig,ax=plt.subplots(figsize=(12,6));ax.set_axis_off()
    ax.set_title('Avoidance Forces: Not Applicable to MPC-D-CBF',fontsize=14)
    ax.text(.5,.55,'This controller does not compute a CAPA avoidance force.\n'
            'Avoidance is enforced through MPC-D-CBF constraints.\n'
            'The original result category is retained for comparison.',
            ha='center',va='center',transform=ax.transAxes,fontsize=12)
    fig.tight_layout();_save_plot(fig,path,'avoidance_forces',stamp)
    # Figure 11: reference 2.5 pt; estimates 1.5 pt; solid x / dash-dot y.
    fig,ax=plt.subplots(figsize=(14,8))
    aligned=any(np.isfinite(r.get('observer_aligned_x',[])).any() and np.isfinite(r.get('leader_reference_x',[])).any() for r in valid.values())
    if aligned:
        r=next(r for r in valid.values() if np.isfinite(r.get('leader_reference_x',[])).any())
        ax.plot(r['t'],r['leader_reference_x'],'--',color='darkblue',lw=PLOT_STYLE['leader_width'],label='x_Leader')
        ax.plot(r['t'],r['leader_reference_y'],'--',color='darkred',lw=PLOT_STYLE['leader_width'],label='y_Leader')
    elif leader:
        for key,reference_color,label in (('x0','darkblue','x_Leader'),('y0','darkred','y_Leader')):
            ax.plot(leader['t'],leader[key],ls='--',color=reference_color,lw=PLOT_STYLE['leader_width'],
                label=label,alpha=.9,zorder=10)
    else:ax.text(.5,.96,'Leader history unavailable; recorded observer estimates shown',ha='center',transform=ax.transAxes)
    for i,r in valid.items():
        ax.plot(r['t'],r['observer_aligned_x'] if aligned else r['zxeo_x'],color=color(i),label='x_est_'+str(i),lw=secondary,alpha=.8)
        ax.plot(r['t'],r['observer_aligned_y'] if aligned else r['zxeo_y'],color=color(i),ls='-.',label='y_est_'+str(i),lw=secondary,alpha=.8)
    _time_axis(ax,end,'Position (m)',('Aligned Observer Reference' if aligned else 'Historical Raw Observer State')+' - '+stamp)
    ax.legend(loc='upper right',ncol=3,fontsize=7,framealpha=.9)
    fig.tight_layout();_save_plot(fig,path,'observer_estimation',stamp)
    # Dedicated prior-plan reference: the newly anchored r[0] would fake zero.
    fig,ax=plt.subplots(figsize=(8,6))
    for i,r in valid.items():
        ax.plot(r['t'],r.get('plan_execution_error_x',[np.nan]*len(r['t'])),color='red',lw=secondary,alpha=.8,label='x_error_'+str(i))
        ax.plot(r['t'],r.get('plan_execution_error_y',[np.nan]*len(r['t'])),color='blue',lw=secondary,alpha=.8,label='y_error_'+str(i))
    if not any(np.isfinite(r.get('plan_execution_error_x',[])).any() for r in valid.values()):
        ax.text(.5,.5,'Accepted plan reference not recorded; internal error unavailable',ha='center',transform=ax.transAxes)
    ax.axhline(0,color='black',ls=':',lw=secondary,alpha=.5,label='Zero Error')
    _time_axis(ax,end,'Plan - Measured Position (m)','Previous Accepted Plan / Execution Error - '+stamp)
    ax.legend(loc='upper right',ncol=2,fontsize=8,framealpha=.9)
    fig.tight_layout();_save_plot(fig,path,'tracking_error_xy',stamp)
    fig,ax=plt.subplots(figsize=(8,4))
    for i,r in valid.items():
        ax.plot(r['t'],r.get('observer_error_x',[np.nan]*len(r['t'])),color=color(i),lw=secondary,label='x_'+str(i))
        ax.plot(r['t'],r.get('observer_error_y',[np.nan]*len(r['t'])),color=color(i),ls='-.',lw=secondary,label='y_'+str(i))
    if not any(np.isfinite(r.get('observer_error_x',[])).any() for r in valid.values()):
        ax.text(.5,.5,'Aligned observer reference not recorded',ha='center',transform=ax.transAxes)
    _time_axis(ax,end,'Aligned Estimate - Leader (m)','Aligned Observer Reference Error')
    ax.legend(ncol=2,fontsize=7);fig.tight_layout();_save_plot(fig,path,'observer_error',stamp)
    if geometry_stats:
        keys=sorted(geometry_stats)
        fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
        for j,field in enumerate(('a','b')):
            for offset,prefix,range_color,label in [(-.12,'raw_','#b27a00','Measured'),(.12,'effective_','#009966','Effective')]:
                lo=np.array([geometry_stats[k].get(prefix+field+'_min',np.nan) for k in keys])
                hi=np.array([geometry_stats[k].get(prefix+field+'_max',np.nan) for k in keys])
                axes[j].vlines(np.arange(len(keys))+offset,lo,hi,color=range_color,label=label,lw=secondary)
                axes[j].scatter(np.arange(len(keys))+offset,lo,s=16,color=range_color)
            axes[j].set_ylabel('Semi-axis '+field+' Range (m)',fontsize=12)
            axes[j].grid(alpha=PLOT_STYLE['grid_alpha']);axes[j].legend(fontsize=8)
        axes[0].set_title('Measured / Effective Ellipse Semi-axis Ranges',fontsize=14)
        short_labels=[]
        previous_observer=None
        for index,key in enumerate(keys):
            observer,kind,identity=key.split('/',2)
            if previous_observer is not None and observer!=previous_observer:
                for ax in axes:ax.axvline(index-.5,color='gray',lw=.6,alpha=.5)
            previous_observer=observer
            short_labels.append(observer.replace('tb3_','R')+':'+dict(static='S',peer='P',dynamic='D').get(kind,'?')+identity)
        axes[-1].set_xticks(range(len(keys)));axes[-1].set_xticklabels(short_labels,rotation=90,fontsize=7)
        axes[-1].set_xlabel('Observer R / static S, dynamic D or peer P / object ID',fontsize=10)
        fig.tight_layout();_save_plot(fig,path,'ellipse_axes',stamp)
    (path/'plot_style_manifest.json').write_text(json.dumps(dict(
        reference='scripts/visualize_results.py',style=PLOT_STYLE,
        robot_colors=ROBOT_COLORS,time_end=end,
        supplements=['laser_min_range','clustering_breakdown','ellipse_axes'],
        notes=['Original 12 category filenames retained; CAPA force is N/A.',
               'X/Y error axes remain automatic as in the active original plotting code.',
               'Heading error retains the requested [-4, 4] rad range.',
               'Distances are unsmoothed measured series; CAPA thresholds are not D-CBF thresholds.',
               'Total groups include tracked static envelopes and freshly detected peers.']),indent=2))


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
            metrics_schema_version=METRICS_SCHEMA_VERSION,metric_definitions=metric_definitions(),
            total_time=rospy.get_param('/tb3_virtual_leader/leader_node/total_time',None),
            data_timeout=rospy.get_param('/cbf/data_timeout',.6),
            timezone='Asia/Shanghai',method='MPC-D-CBF',
            world_file=rospy.get_param('/cbf/world_file',''),
            formation_radius=rospy.get_param('/cbf/formation_radius',.5),
            control_dt=rospy.get_param('/cbf/mpc/dt',.1),
            realtime_solver=rospy.get_param('/cbf/realtime_solver',True),
            control_budget_ms=rospy.get_param('/cbf/control_budget_ms',100.),
            nlp_budget_ms=rospy.get_param('/cbf/nlp_budget_ms',65.),
            dual_warm_start=rospy.get_param('/cbf/dual_warm_start',True),
            stagger_callbacks=rospy.get_param('/cbf/stagger_callbacks',True),
            dynamic_prediction=rospy.get_param('/cbf/dynamic_prediction',False),
            gazebo_seed=rospy.get_param('/cbf/gazebo_seed',''),
            reliability_parameters=dict(dict(reliability_horizon=1.,score_tau=.5,weight_tau=.3,score_floor=.015,switch_ratio=.8,switch_hold=.6),
                **rospy.get_param('/cbf/dynamic/reliability',{})),
            prediction_mode=rospy.get_param('/cbf/dynamic/prediction_mode','cv'),
            progress_guidance=rospy.get_param('/cbf/progress_guidance',False),
            moving_frame_projection=rospy.get_param('/cbf/moving_frame_projection',False),
            singer_tau=rospy.get_param('/cbf/dynamic/singer_tau',1.),
            acceleration_buffer=rospy.get_param('/cbf/dynamic/acceleration_buffer',True),
            stage_velocity_guidance=rospy.get_param('/cbf/dynamic/stage_velocity_guidance',True),
            ca_jerk_density=rospy.get_param('/cbf/dynamic/ca_jerk_density',.01),
            ca_acceleration_variance=rospy.get_param('/cbf/dynamic/ca_acceleration_variance',.04),
            dynamic_reaction_time=rospy.get_param('/cbf/dynamic/reaction_time',.35),
            dynamic_extra_margin=rospy.get_param('/cbf/dynamic/extra_margin',.02),
            dynamic_scenario=rospy.get_param('/cbf/dynamic_scenario','none'),
            robot_names=rospy.get_param('/cbf/robot_names',['tb3_'+str(i) for i in range(5)]),
            trajectory_type=rospy.get_param('/tb3_virtual_leader/leader_node/trajectory_type','unknown'))
        (self.path/'run_metadata.json').write_text(json.dumps(self.meta,indent=2,ensure_ascii=False))
        self.leader_stream=(self.path/'leader.csv').open('w',newline='')
        self.leader_writer=csv.DictWriter(self.leader_stream,fieldnames=
            ['trajectory_time','ros_time','x0','y0','theta0','v0','w0','u0x','u0y'])
        self.leader_writer.writeheader()
        self.geometry_stream=(self.path/'geometry_estimates.csv').open('w',newline='')
        self.geometry_writer=csv.DictWriter(self.geometry_stream,fieldnames=self.GEOMETRY_FIELDS)
        self.geometry_writer.writeheader();self.last_flush=time.monotonic()
        self.prediction_stream=(self.path/'obstacle_predictions.jsonl').open('w')
        self.subscribers=[rospy.Subscriber('/leader/history',LeaderHistoryData,self.on_leader,queue_size=100),
            rospy.Subscriber('/system/shutdown',Bool,lambda m:self.requested.set() if m.data else None)]
        for i,name in enumerate(self.meta['robot_names']):
            rospy.set_param('/cbf/ready/tb3_'+str(i),False)
            self.subscribers.append(rospy.Subscriber('/tb3_%d/cbf_diagnostics'%i,String,self.on_robot,callback_args=i,queue_size=100))
            self.subscribers.append(rospy.Subscriber('/tb3_%d/cbf_geometry/estimates'%i,String,self.on_geometry,queue_size=100))
            self.subscribers.append(rospy.Subscriber('/'+name+'/cbf_geometry/predictions',String,self.on_prediction,queue_size=20))
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

    def on_prediction(self,msg):
        with self.lock:
            if not self.finished:
                self.prediction_stream.write(msg.data+'\n')
                self.prediction_stream.flush()

    def finish(self):
        with self.finish_lock:
            if self.finished:return
            with self.lock:
                self.finished=True
                self.leader_stream.flush();self.leader_stream.close()
                self.geometry_stream.flush();self.geometry_stream.close()
                self.prediction_stream.flush();self.prediction_stream.close()
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
