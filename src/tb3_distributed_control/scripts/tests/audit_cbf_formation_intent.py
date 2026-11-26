#!/usr/bin/env python3
"""Link small-state requests from producer trace through transport to reference.

Checks recorded events and original sampled hard constraints, not certified
future promises, continuous physical safety or arbitrary communication faults.
"""
import argparse,json,sys,collections
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'scripts'),str(P/'纯仿真代码')]
from mpc_dcbf_cached_solver import validate_formation_intent,CachedMPCProblem


def audit(directory):
    root=Path(directory);errors=[];sent={};received={};worker_records={};modes=collections.Counter()
    event_count=received_count=invalid_count=used=checked=0;stamps=[];sample_ages=[]
    def error(kind,**details):errors.append(dict(error=kind,**details))
    for file in root.glob('tb3_*_solver_trace.jsonl'):
        robot=int(file.name.split('_')[1])
        for event in map(json.loads,file.open()):
            local=event['manager'].get('local_worker',{});record=local.get('record',{})
            if record:worker_records[(robot,local['sequence'])]=record
    for file in root.glob('tb3_*_intent_messages.jsonl'):
        robot=int(file.name.split('_')[1]);last_sent=-1;last_time=None;last_received={}
        for event in map(json.loads,file.open()):
            p=event['packet'];packet_key=tuple(p)
            if event['direction']=='sent':
                now=event['send_stamp'];valid,reason=validate_formation_intent(p,now,last_sent)
                if not valid:error('Invalid published request',robot=robot,reason=reason)
                if last_time is not None and now-last_time<.19-1e-9:error('Publish rate exceeds declared bound',robot=robot)
                last_sent=p[1];last_time=now;stamps.append(now);event_count+=1;sent[(robot,packet_key)]=event
                record=worker_records.get((robot,event['source_sequence']))
                if record is None or record.get('intent_proposal')!=p or event['source_sequence']!=p[1]:
                    error('Published request has no matching recorded producer',robot=robot,sequence=p[1])
            else:
                peer=event['peer'];now=event['receive_stamp'];received_count+=1
                valid,reason=validate_formation_intent(p,now,last_received.get(peer,-1))
                if valid!=event['accepted'] or reason!=event['reason']:error('Reception validation mismatch',robot=robot,peer=peer)
                if event['accepted']:
                    last_received[peer]=p[1];received[(robot,peer,packet_key)]=now
                else:invalid_count+=1
    for (robot,peer,packet_key),stamp in received.items():
        if (peer,packet_key) not in sent:error('Accepted request has no matching sender log',robot=robot,peer=peer)
    for (robot,sequence),record in worker_records.items():
        summary=record.get('formation_intent',{});modes[summary.get('mode')]+=1
        if not summary.get('adopted_reference'):continue
        if record.get('spatial_timeout_carry',False):continue
        snapshot=record['input_snapshot'];cfg=snapshot['config'];now=summary['stamp']
        # Ignore a stale diagnostic left by a pre-preview deadline; do not
        # attribute a carried candidate to a new intent reference.
        current=max(o.get('coordination_stamp',0.) for o in snapshot['obstacles'] if o.get('peer',False))
        if abs(current-now)>1e-8:continue
        for peer in summary.get('peer_sources',[]):
            obs=next(o for o in snapshot['obstacles'] if o['name']=='peer_'+str(peer))
            p=obs['formation_intent'];valid,reason=validate_formation_intent(p,now)
            if not valid:error('Reference uses expired/invalid request',robot=robot,peer=peer,reason=reason)
            packet_key=tuple(p)
            if (robot,peer,packet_key) not in received:error('Reference request has no accepted reception',robot=robot,peer=peer)
            sample_ages.append(now-p[2]);used+=1
        controls=np.asarray(summary['reference_controls']);states=np.asarray(summary['reference_states'])
        vectors=controls[:,:1]*np.column_stack((np.cos(states[1:,2]),np.sin(states[1:,2])))
        candidate=dict(controls=controls,states=states,plan_controls=vectors)
        checker=CachedMPCProblem.__new__(CachedMPCProblem);checker.config=cfg;checker.dt=cfg['dt'];checker.constraint_mode='dcbf'
        if not checker.validate(candidate,np.asarray(snapshot['state']),snapshot['obstacles'],snapshot['bounds'],np.asarray(snapshot['previous'])):
            error('Adopted intent reference fails original hard prefix',robot=robot,sequence=sequence)
        guide=snapshot['guidance']
        if guide['level']!=1 or np.max(np.abs(np.asarray(guide['vectors'])-vectors))>1e-9:
            error('Recorded applied guide differs from intent reference',robot=robot,sequence=sequence)
        if summary['mean_reference_speed']<.09-1e-12:error('Low-motion reference labelled adopted',robot=robot,sequence=sequence)
        checked+=1
    meta=json.loads((root/'run_metadata.json').read_text());exchange=meta.get('intent_exchange',True)
    if exchange and not(event_count and used):error('Enabled exchange not exercised')
    if not exchange and(event_count or received_count or used):error('Local-only control received communication')
    duration=max(stamps)-min(stamps) if len(stamps)>1 else None
    return dict(passed=checked>0 and not errors,exchange_enabled=exchange,published_messages=event_count,
        received_messages=received_count,invalid_received=invalid_count,linked_accepted_receptions=len(received),
        used_peer_requests=used,checked_original_prefix_references=checked,unique_worker_modes=dict(modes),
        maximum_used_source_age=max(sample_ages) if sample_ages else None,
        payload_float64=12,payload_bytes_per_message=96,
        observed_publish_span=duration,sender_stream_payload_bytes=96*event_count,delivered_payload_bytes=96*received_count,
        delivered_payload_bytes_per_second=None if duration is None else 96*received_count/duration,
        bandwidth_excludes='ROS serialization headers, TCP/IP, retransmission and other topics',
        error_count=len(errors),first_error=errors[0] if errors else None,scope=__doc__)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(audit(a.directory),indent=2))
