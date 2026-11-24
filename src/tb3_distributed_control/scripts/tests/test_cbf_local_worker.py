"""Delayed IPC results must pass the original model at current inputs."""
from pathlib import Path
import sys,time,unittest
from unittest.mock import Mock
import numpy as np
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P/'scripts'),str(P/'纯仿真代码')]
from cbf_mpc_core import default_config
import mpc_dcbf_cached_solver as core


def config(**kw):
    cfg=default_config();cfg.update(realtime=True,execution_backup=True,
        nlp_wall_time=.065,manager_wall_time=.085,log_mpc_inputs=True,local_worker=True)
    cfg.update(kw);return cfg


class CurrentWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.problem=core.CachedMPCProblem(config(),1,False)

    def message(self,stamp=10.):
        u=np.tile([.15,.1],(10,1));u[:,0]=np.linspace(.10,.19,10)
        return dict(stamp=stamp,sequence=7,solution=dict(controls=u,plan_controls=u.copy()))

    def check(self,message,now=10.1,state=None,obs=None,problem=None,previous=None):
        return core.current_worker_candidate(problem or self.problem,message,now,
            np.zeros(3) if state is None else state,[] if obs is None else obs,[-.22,.22],
            np.zeros(2) if previous is None else previous,.3)

    def test_shift_and_reanchor_use_current_state_not_stale_rollout(self):
        m=self.message();m['solution']['states']=np.full((11,3),999.)
        state=np.array([.03,.02,.1]);candidate,reason=self.check(m,state=state)
        self.assertEqual(reason,'accepted');self.assertEqual(candidate['worker_shift'],1)
        np.testing.assert_array_equal(candidate['states'][0],state)
        np.testing.assert_allclose(candidate['controls'][0],m['solution']['controls'][1])
        self.assertFalse(candidate['success']);self.assertEqual(candidate['validated_steps'],10)

    def test_stale_future_failed_and_malformed_result_rejected(self):
        for msg,now,reason in [(None,10.,'pending'),(self.message(),10.301,'stale_or_future'),
                              (self.message(),9.99,'stale_or_future')]:
            candidate,got=self.check(msg,now);self.assertIsNone(candidate);self.assertEqual(got,reason)
        for key,bad in [('controls',np.zeros((1,2))),('plan_controls',np.full((10,2),np.nan))]:
            m=self.message();m['solution'][key]=bad;self.assertIsNone(self.check(m)[0])
        m=self.message();m['solution']=None;self.assertEqual(self.check(m)[1],'worker_no_solution')

    def test_new_geometry_and_original_speed_bounds_reject_old_solution(self):
        obs=[dict(name='new',center=np.array([.2,0]),a=.15,b=.15,theta=0.)]
        self.assertEqual(self.check(self.message(),obs=obs)[1],'current_validation_failed')
        m=self.message();m['solution']['controls'][2,0]=.23
        self.assertEqual(self.check(m)[1],'current_validation_failed')

    def test_actual_previous_control_used_for_increment_bound(self):
        p=core.CachedMPCProblem(config(delta_limit=True),1,True)
        m=self.message();m['solution']['controls']=np.tile([.15,0.],(10,1))
        self.assertIsNone(self.check(m,problem=p,previous=np.array([-.2,0]))[0])
        self.assertIsNotNone(self.check(m,problem=p,previous=np.array([.15,0]))[0])

    def test_worker_flag_does_not_change_nlp_variables_or_constraints(self):
        p=core.CachedMPCProblem(config(local_worker=False),1,False)
        self.assertEqual((p.opti.nx,p.opti.ng,p.opti.np),
            (self.problem.opti.nx,self.problem.opti.ng,self.problem.opti.np))

    def test_compensated_origin_keeps_unexecuted_first_command(self):
        msg=self.message();msg['application_stamp']=10.1
        c,reason=self.check(msg,now=10.1)
        self.assertEqual(reason,'accepted');self.assertEqual(c['worker_shift'],0)
        np.testing.assert_array_equal(c['controls'][0],msg['solution']['controls'][0])
        msg['submitted_wall']=time.perf_counter()-.31
        self.assertEqual(self.check(msg)[1],'stale_wall')

    def test_future_request_uses_actual_command_and_cv_forecast_without_mutation(self):
        cfg=config();state=np.array([.2,.3,.1]);command=np.array([.17,.3])
        centers=np.column_stack([1.+.01*np.arange(11),np.ones(11)])
        obstacle=dict(name='moving',center=centers[0],a=.2,b=.15,theta=0.,prediction_centers=centers,
            guidance_motion=dict(stamp=10.,weights=[1.],models=[dict(mode='cv',position=[1.,1.],velocity=[.1,0.],acceleration=[0.,0.],tau=1.)]))
        goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11)
        task=dict(state=state,stamp=10.,goals=goals,obstacles=[obstacle],previous=np.zeros(2))
        request=core.delay_compensated_request(task,command,cfg)
        np.testing.assert_allclose(request['state'],core.rollout(state,command[None,:],.1)[1])
        np.testing.assert_allclose(request['goals'][:,0],.015*np.arange(1,12))
        np.testing.assert_allclose(request['obstacles'][0]['prediction_centers'][:,0],1.+.01*np.arange(1,12))
        np.testing.assert_allclose(core.progress_centers(request['obstacles'][0],[0.,.5,2.],.1),[[1.01,1],[1.06,1],[1.21,1]])
        self.assertEqual(request['application_stamp'],10.1)
        np.testing.assert_array_equal(request['previous'],command)
        np.testing.assert_array_equal(task['state'],state);self.assertEqual(obstacle['guidance_motion']['models'][0]['position'],[1.,1.])

    def test_foreground_pending_and_invalid_worker_never_call_nlp(self):
        m=core.MPCSolverManager(config());m.problems[(1,False)]=self.problem
        original=self.problem.solve;self.problem.solve=Mock(side_effect=AssertionError('Foreground NLP'))
        goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11)
        obs=[dict(name='far',center=np.array([10.,0]),a=.15,b=.15,theta=0.)]
        try:
            for msg in (None,self.message(8.)):
                result=m.select_local(np.zeros(3),goals,obs,[-.22,.22],np.zeros(2),'r',10.,msg,
                    None,None,.3,time.perf_counter()+.08,1)
                self.assertEqual(result['validated_steps'],10)
                self.assertTrue(self.problem.validate(result,np.zeros(3),obs,[-.22,.22],np.zeros(2)))
                np.testing.assert_array_equal(m.last_records['r']['input_snapshot']['state'],np.zeros(3))
            self.problem.solve.assert_not_called();self.assertEqual(m.summary()['robots']['r']['cycles'],2)
        finally:self.problem.solve=original


class SpawnWorkerTests(unittest.TestCase):
    def test_separate_process_frozen_request_and_clean_exit(self):
        worker=core.LocalMPCWorker(config(),0)
        try:
            limit=time.monotonic()+20.
            while not worker.ready and time.monotonic()<limit:
                worker.poll()
                if worker.error:self.fail(worker.error)
                time.sleep(.01)
            self.assertTrue(worker.ready);self.assertNotEqual(worker.process.pid,__import__('os').getpid())
            state=np.zeros(3);goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11)
            task=dict(sequence=3,stamp=10.,submitted_wall=time.perf_counter(),state=state,goals=goals,
                obstacles=[],bounds=(-.22,.22),previous=np.zeros(2),last=None,shift=1,label='local',budget=.097)
            self.assertTrue(worker.submit(task));state[0]=999.;goals[:]=999.
            result=None;limit=time.monotonic()+10.
            while result is None and time.monotonic()<limit:
                result=worker.poll();time.sleep(.01)
            self.assertIsNotNone(result);self.assertEqual(result['sequence'],3)
            np.testing.assert_array_equal(result['record']['input_snapshot']['state'],np.zeros(3))
            self.assertIsNotNone(result['solution'])
        finally:worker.close()
        self.assertFalse(worker.process.is_alive());self.assertEqual(worker.process.exitcode,0)


if __name__=='__main__':unittest.main()
