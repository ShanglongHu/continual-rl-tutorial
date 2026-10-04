"""持续学习教学核的手算、有限差分、退化与真实曲线检查。"""
import ast
import importlib
import math
import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DIRECTORY=ROOT/"implementations"/"continual"


def module(name):
    return importlib.import_module("implementations.continual."+name)


class ContinualTests(unittest.TestCase):
    def test_all_runs_and_literal_metadata(self):
        modules={p.stem:module(p.stem) for p in DIRECTORY.glob("*.py")
                 if not p.name.startswith("_")}
        self.assertEqual(len(modules),22)
        for name,m in modules.items():
            with self.subTest(name=name):
                tree=ast.parse((DIRECTORY/(name+".py")).read_text())
                meta=next(n.value for n in tree.body if isinstance(n,ast.Assign)
                          and any(isinstance(t,ast.Name) and t.id=="META" for t in n.targets))
                self.assertEqual(ast.literal_eval(meta),m.META)
                emitted=[]
                rows=m.run(seed=4,steps=1000,emit=emitted.append)
                self.assertEqual(rows,emitted)
                self.assertEqual(rows,m.run(seed=4,steps=1000))
                self.assertEqual(rows[-1]["step"],1000)
                self.assertLessEqual(len(rows),100)
                self.assertTrue(all(math.isfinite(r["value"]) for r in rows))
                baseline=modules[m.META["baseline"]]
                for field in ("task","metric","unit","higher_better","budget"):
                    self.assertEqual(m.META[field],baseline.META[field])
                with self.assertRaises(ValueError):
                    m.run(steps=0)

    def test_rtrl_finite_difference_and_full_bptt(self):
        theta=[.4,.3,-.1]
        xs=[.2,-.5,.7,.1]
        target=.3
        grad,error=module("rtrl").gradient(theta,xs,target)
        full,_=module("tbptt").gradient(theta,xs,target,window=len(xs))
        for a,b in zip(grad,full):
            self.assertAlmostEqual(a,b,places=12)
        for j in range(3):
            plus,minus=theta[:],theta[:]
            plus[j]+=1e-6
            minus[j]-=1e-6
            ep=module("rtrl").gradient(plus,xs,target)[1]
            em=module("rtrl").gradient(minus,xs,target)[1]
            numerical=(.5*ep*ep-.5*em*em)/2e-6
            self.assertAlmostEqual(grad[j],numerical,places=8)

    def test_idbd_tidbd_gamma_zero_equivalence(self):
        w,beta,history,x,y=[.1,.2],[-3.,-3.],[.2,.1],[1.,.5],.8
        a=module("idbd").update(w,beta,history,x,y)
        b=module("tidbd").update(w,beta,history,[9.,4.],x,y,[1.,1.],gamma=0.)
        self.assertEqual(a[:3],b[:3])
        self.assertEqual(a[3],b[4])
        self.assertEqual(b[3],x)

    def test_ewc_anchor_penalty_and_zero_strength(self):
        m=module("ewc")
        unconstrained,_=m.update([1.,0.],[0.,0.],0.,[0.,0.],[2.,0.],strength=0.)
        self.assertEqual(unconstrained,[1.,0.])
        constrained,_=m.update([1.,0.],[0.,0.],0.,[0.,0.],[2.,0.],strength=.5,alpha=.1)
        self.assertEqual(constrained,[.9,0.])

    def test_reservoir_capacity_and_full_retention_before_capacity(self):
        import random
        buffer=[]
        for i in range(100):
            module("reservoir_replay").insert(buffer,i,i+1,random.Random(i),capacity=3)
        self.assertEqual(len(buffer),3)
        self.assertTrue(all(0<=x<100 for x in buffer))
        buffer=[]
        for i in range(3):
            module("reservoir_replay").insert(buffer,i,i+1,random.Random(i),capacity=3)
        self.assertEqual(buffer,[0,1,2])

    def test_differential_shared_old_error(self):
        q,rate,delta=module("differential_q").update([1.,2.],.5,0,1.,1,alpha=.1,eta=.2)
        self.assertEqual(delta,1.5)
        self.assertEqual(q,[1.15,2.])
        self.assertAlmostEqual(rate,.53)

    def test_gvf_td_and_gtd_lambda_zero(self):
        w=module("gvf_td").update([[0.,0.],[0.,0.]],0,1,[1.,0.],[.8,.5],alpha=.1)
        self.assertEqual(w,[[.1,0.],[0.,0.]])
        w,h,e=module("gvf_gtd_lambda").update([[0.,0.]],[[0.,0.]],[[0.,0.]],
            0,1,[1.],[.8],alpha=.1,beta=.2,lam=0.)
        self.assertEqual(w,[[.1,0.]])
        self.assertEqual(h,[[.2,0.]])
        self.assertEqual(e,[[1.,0.]])

    def test_potential_telescopes_with_true_terminal(self):
        m=module("potential_shaping")
        shaped=raw=0.
        for s in range(5):
            r,sp=module("chain_q").chain(s,1)
            raw+=.95**s*r
            shaped+=.95**s*m.shape(r,s,sp)
        self.assertAlmostEqual(shaped,raw-m.potential(0))
        self.assertEqual(m.potential(None),0.)

    def test_successor_features_synchronous_and_gpi(self):
        m=module("successor_features_gpi")
        psi=[[[0.,0.],[0.,0.]],[[0.,0.],[0.,0.]]]
        m.update(psi,0,alpha=1.)
        self.assertEqual(psi[0][0],[1.,0.])
        self.assertEqual(psi[1][0],[1.,0.])
        exact=[[[5.,0.],[4.,1.]],[[1.,4.],[0.,5.]]]
        self.assertEqual(m.select(exact,[.2,1.]),1)
        self.assertEqual(m.select(exact,[1.,.2]),0)

    def test_options_discount_by_primitive_duration(self):
        m=module("option_smdp_q")
        q={0:[0.,0.],2:[2.,4.]}
        target=m.update(q,0,1,[1.,2.],2,alpha=1.,gamma=.5)
        self.assertEqual(target,3.)
        self.assertEqual(q[0][1],3.)
        q={0:[0.,0.],2:[2.,4.]}
        self.assertEqual(m.update(q,0,1,[1.,2.],None,alpha=1.,gamma=.5),2.)
        self.assertEqual(m.run(steps=3)[-1]["step"],3)

    def test_option_evaluation_preserves_call_and_return(self):
        m=module("option_smdp_q")
        q={s:[0.,1.] for s in range(5)}
        q[1]=[1.,0.]
        expected=sum(-.01*.95**k for k in range(4))+.95**4
        self.assertAlmostEqual(m.evaluate(q),expected,places=14)
        self.assertAlmostEqual(m.chain_score(q),-.2,places=14)
        left={s:[1.,0.] for s in range(5)}
        self.assertAlmostEqual(m.evaluate(left),-.2,places=14)

    def test_exact_discounted_cycle_and_model_policy(self):
        m=module("chain_q")
        left={s:[1.,0.] for s in range(5)}
        self.assertAlmostEqual(m.chain_score(left),-.2,places=14)
        for name in ("learned_model_mpc","one_step_model"):
            planner=module(name)
            model={}
            for s in range(5):
                for a in range(2):
                    planner.update(model,s,a,-.01,None if s==4 and a==1 else s)
            self.assertAlmostEqual(planner.evaluate(model,1,1.),-.2,places=14)

    def test_empirical_model_learning_and_horizon(self):
        m=module("learned_model_mpc")
        model={}
        m.update(model,0,1,1.,1)
        m.update(model,0,1,3.,1)
        self.assertEqual(model[(0,1)][:2],(2,4.))
        self.assertEqual(m.action_values(model,0,horizon=1)[1],2.)
        m.update(model,1,1,10.,None)
        self.assertAlmostEqual(m.action_values(model,0,horizon=2)[1],11.5)
        # 精确确定性观测覆盖全部动作后，奖励模型误差应为0。
        model={}
        for s in range(5):
            for a in range(2):
                r,sp=m.chain(s,a)
                m.update(model,s,a,r,sp)
        self.assertEqual(m.model_error(model,1.),0.)

    def test_count_bonus_decay_and_raw_reward_evaluation(self):
        m=module("count_bonus")
        q={0:[0.,0.]}
        self.assertAlmostEqual(m.update(q,0,0,0.,None,1),.2)
        self.assertAlmostEqual(m.update(q,0,0,0.,None,4),.1)
        q={0:[0.,0.]}
        self.assertEqual(module("plain_q_exploration").update(q,0,0,0.,None,1),0.)


if __name__=="__main__":
    unittest.main()
