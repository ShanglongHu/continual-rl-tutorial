"""教学实现的数学/时序检查，不把冒烟结果作为性能证据。"""
import copy
import math
import random
import unittest
import importlib.util
if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("深度算法测试需要可选依赖 torch；标准库 CI 跳过，neural CI 实际执行。")
import torch
from implementations.deep import dqn, double_dqn, vpg, a2c, ppo, ddpg, td3, sac
from implementations.deep import c51, qr_dqn, cql, iql, offline_q_learning
from implementations.deep import trpo
from implementations.deep._common import Q, mlp, GaussianActor, advantages, polyak


class ConstantQ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
    def forward(self, x):
        return self.values.expand(len(x), -1)


class DeepChecks(unittest.TestCase):
    def test_trpo_conjugate_gradient_numeric_system(self):
        matrix=torch.tensor([[4.,1.],[1.,3.]],dtype=torch.float64)
        b=torch.tensor([1.,2.],dtype=torch.float64)
        solution=trpo.conjugate_gradient(lambda v:matrix@v,b,iterations=10,tolerance=1e-20)
        self.assertTrue(torch.allclose(solution,torch.linalg.solve(matrix,b),atol=1e-10))

    def test_trpo_hvp_finite_difference_and_symmetry(self):
        torch.manual_seed(31)
        actor=mlp(6,2).double(); x=torch.randn(12,6,dtype=torch.float64)
        old=torch.distributions.Categorical(logits=actor(x)).probs.detach()
        origin=trpo.flat_parameters(actor)
        v=torch.randn_like(origin); v/=v.norm()
        w=torch.randn_like(origin); w/=w.norm()
        hv=trpo.hessian_vector_product(actor,x,old,v,damping=0)
        hw=trpo.hessian_vector_product(actor,x,old,w,damping=0)
        self.assertAlmostEqual(float(w@hv),float(v@hw),places=9)
        def kl_gradient(flat):
            trpo.set_parameters(actor,flat)
            kl=torch.distributions.kl_divergence(
                torch.distributions.Categorical(probs=old),
                torch.distributions.Categorical(logits=actor(x))).mean()
            return trpo.flat_gradient(kl,list(actor.parameters())).detach()
        epsilon=1e-5
        difference=(kl_gradient(origin+epsilon*v)-kl_gradient(origin-epsilon*v))/(2*epsilon)
        trpo.set_parameters(actor,origin)
        self.assertTrue(torch.allclose(hv,difference,atol=1e-7,rtol=1e-5))

    def test_trpo_kl_acceptance_and_failed_search_rollback(self):
        torch.manual_seed(44)
        actor=mlp(6,2); x=torch.randn(32,6)
        with torch.no_grad():
            dist=torch.distributions.Categorical(logits=actor(x))
            actions=dist.sample(); old_logp=dist.log_prob(actions); old=dist.probs
        advantage=torch.where(actions==1,1.,-1.)
        origin=trpo.flat_parameters(actor)
        rejected=trpo.policy_update(actor,x,actions,old_logp,old,advantage,backtrack_iters=0)
        self.assertEqual(rejected["actor_updated"],0)
        self.assertTrue(torch.equal(origin,trpo.flat_parameters(actor)))
        accepted=trpo.policy_update(actor,x,actions,old_logp,old,advantage,delta=.01)
        self.assertEqual(accepted["actor_updated"],1)
        self.assertLessEqual(accepted["actual_kl"],.01)
        self.assertGreater(accepted["surrogate_improvement"],0)
    def test_dqn_double_selection_and_terminal_mask(self):
        # online 偏好动作 0，target 偏好动作 1：DDQN 的选择与评价必须分离。
        data = (torch.zeros(2, 6), torch.tensor([0, 0]),
                torch.tensor([0., 2.]), torch.zeros(2, 6), torch.tensor([0., 1.]))
        losses = []
        for mod in (dqn, double_dqn):
            online, target = ConstantQ([3., 1.]), ConstantQ([2., 9.])
            target.requires_grad_(False)
            opt = torch.optim.SGD(online.parameters(), lr=0.)
            loss = mod.update(online, target, opt, data)
            tail = 9. if mod is dqn else 2.
            y = torch.tensor([.99*tail, 2.])
            expected = torch.nn.functional.smooth_l1_loss(torch.tensor([3., 3.]), y)
            self.assertAlmostEqual(loss, float(expected), places=6)
            self.assertIsNone(target.values.grad)
            self.assertTrue(torch.isfinite(online.values.grad).all())
            losses.append(loss)
        self.assertNotEqual(*losses)

    def test_gae_terminal_vs_sampling_cutoff(self):
        # 采样截止允许 V(next)=5；真实终止会删除该尾价值。
        adv, returns = advantages([1.], [2.], [5.], [False], lam=.95)
        self.assertEqual(float(adv[0]), 4.)
        self.assertEqual(float(returns[0]), 6.)
        _, terminal_return = advantages([1.], [2.], [5.], [True])
        self.assertEqual(float(terminal_return[0]), 1.)
        # 终止使递推停止，不能把下一个 reset 回合的优势加过来。
        adv, _ = advantages([1., 7.], [0., 0.], [5., 0.], [True, True])
        self.assertEqual(adv.tolist(), [1., 7.])

    def test_polyak_hand_calculation(self):
        online, target = ConstantQ([10., 20.]), ConstantQ([2., 4.])
        polyak(online, target, .25)
        self.assertEqual(target.values.tolist(), [4., 8.])
        self.assertIsNone(target.values.grad)

    def test_td3_delays_actor_and_targets(self):
        torch.manual_seed(5)
        actor = mlp(2, 1)
        critics = [Q(), Q()]
        target_actor = copy.deepcopy(actor).requires_grad_(False)
        targets = [copy.deepcopy(q).requires_grad_(False) for q in critics]
        actor_opt = torch.optim.Adam(actor.parameters(), lr=.001)
        opts = [torch.optim.Adam(q.parameters(), lr=.002) for q in critics]
        data = (torch.randn(8, 2), torch.zeros(8, 1), -torch.ones(8),
                torch.randn(8, 2), torch.zeros(8))
        old_actor = [p.clone() for p in actor.parameters()]
        old_targets = [p.clone() for q in targets for p in q.parameters()]
        first = td3.update(actor, critics, target_actor, targets, actor_opt, opts, data, 1)
        self.assertEqual(first["actor_updated"], 0)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(old_actor, actor.parameters())))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(old_targets, [p for q in targets for p in q.parameters()])))
        second = td3.update(actor, critics, target_actor, targets, actor_opt, opts, data, 2)
        self.assertEqual(second["actor_updated"], 1)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(old_actor, actor.parameters())))
        self.assertTrue(all(p.grad is None for q in targets for p in q.parameters()))

    def test_sac_density_and_pathwise_gradient(self):
        torch.manual_seed(3)
        actor = GaussianActor()
        x = torch.randn(16, 2)
        action, logp = actor(x)
        self.assertEqual(tuple(logp.shape), (16,))
        self.assertTrue(torch.isfinite(logp).all())
        self.assertTrue((action.abs() <= 1).all())
        (action.square().mean()+.1*logp.mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_c51_probability_mass_and_exact_terminal_atom(self):
        support=torch.linspace(-1,1,21)
        p=torch.full((2,21),1/21)
        projected=c51.project(p,torch.tensor([0.,1.]),torch.ones(2),support)
        self.assertTrue(torch.allclose(projected.sum(-1),torch.ones(2)))
        self.assertAlmostEqual(float(projected[0,10]),1.,places=5)
        self.assertAlmostEqual(float(projected[1,-1]),1.,places=5)
        self.assertTrue((projected>=0).all())

    def test_quantile_huber_and_expectile_hand_values(self):
        # 一个正残差 1，tau=.25：Huber=.5，非对称权重=.25。
        prediction=torch.tensor([[0.]],requires_grad=True)
        loss=qr_dqn.quantile_loss(prediction,torch.tensor([[1.]]),torch.tensor([.25]))
        self.assertAlmostEqual(float(loss.detach()),.125)
        loss.backward()
        self.assertAlmostEqual(float(prediction.grad),-.25)
        # 正负 residual 各 2：.7*4 和 .3*4 的平均为 2。
        self.assertAlmostEqual(float(iql.expectile_loss(torch.tensor([2.,-2.]))),2.)

    def test_offline_dataset_independent_of_learner_seed(self):
        from implementations.deep._common import fixed_offline_data
        first=fixed_offline_data()
        torch.manual_seed(42); random.seed(77)
        second=fixed_offline_data()
        self.assertEqual(len(first),512)
        self.assertTrue(all(torch.equal(a[0],b[0]) and a[2]==b[2]
                            for a,b in zip(first,second)))
        self.assertEqual(cql.META["task"],iql.META["task"])
        for module in (cql,iql,offline_q_learning):
            self.assertEqual(module.META["budget"],"training_batches")
            rows=module.run(seed=0,steps=4)
            self.assertEqual(rows[-1]["training_batches"],4)
            self.assertEqual(rows[-1]["optimizer_steps"],12 if module is iql else 4)

    def test_tiny_rollouts_share_grid_and_emit(self):
        grids = []
        for mod in (dqn, double_dqn, vpg, a2c, ppo, ddpg, td3, sac, c51, qr_dqn, cql, iql, offline_q_learning, trpo):
            emitted = []
            rows = mod.run(seed=1, steps=48, emit=emitted.append)
            self.assertEqual(rows, emitted)
            self.assertLessEqual(len(rows), 100)
            self.assertTrue(all(math.isfinite(row["value"]) for row in rows))
            self.assertEqual(rows[0]["step"], 0)
            self.assertEqual(rows[-1]["step"], 48)
            self.assertEqual(mod.run(seed=1, steps=48), rows)
            grids.append([row["step"] for row in rows])
        self.assertTrue(all(g == grids[0] for g in grids))


if __name__ == "__main__":
    unittest.main()
