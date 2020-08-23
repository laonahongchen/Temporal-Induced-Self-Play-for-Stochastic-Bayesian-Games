import random


class Policy(object):
    def __init__(self, act_fn, prob_fn=None, strategy_fn=None, tpred_fn=None, vpred_fn=None):
        self.act_fn = act_fn
        self.prob_fn = prob_fn
        self.strategy_fn = strategy_fn
        self.tpred_fn = tpred_fn
        self.vpred_fn = vpred_fn

    def strategy(self, obs):
        return self.strategy_fn(obs)

    def act(self, obs):
        action = self.act_fn(obs)
        return action

    def act_with_prob(self, obs):
        action = self.act_fn(obs)
        prob = self.prob_fn(obs, action)
        return action, prob

    def act_clean(self, obs):
        action = self.act_fn(obs)
        if isinstance(action, tuple):
            return action[0]
        else:
            return action

    def prob(self, obs, ac):
        return self.prob_fn(obs, ac)

    def tpred(self, ob):
        return self.tpred_fn(ob)

    def vpred(self, ob):
        return self.vpred_fn(ob)


class MixedPolicy(Policy):
    def __init__(self, policies, probabilities):
        self.policies = policies
        self.probabilities = probabilities

        def act_fn(obs):
            p = random.choices(self.policies, weights=self.probabilities)[0]
            return p.act(obs)

        super().__init__(act_fn)

