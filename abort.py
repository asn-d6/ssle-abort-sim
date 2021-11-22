"""
Simulate RANDAO attacks against SSLE candidate and proposer selection.

Also simulate RANDAO attacks against the status quo of beacon chain proposal picking

Initial code is based on Gottfried Herold's prototype
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import math
import random

import scipy.stats

N_RUNS = 10000

FILTERED_SET_SIZE = 2**14 # 16384
SAMPLED_SET_SIZE = 2**13 # 8192

def popcount(x):
    assert(x >= 0)
    return bin(x).count("1")

class Attack(object):
    """Represents an attack against the system"""
    def __init__(self, net_gain, evil_candidates, cost, start_evil_candidates):
        """
        - `net_gain` is the adversarial gain (in proposals) over the naive strategy of not aborting
        - `evil_candidates` is the number of candidates the adversary got
        - `cost` is the number of aborts the adverary had to do to pull off the attack
        - `default_candidates` is the number of candidates would have gotten without an attack
        """
        self.evil_candidates = evil_candidates
        self.cost = cost
        self.start_evil_candidates = start_evil_candidates
        self.net_gain = net_gain

    def __gt__(self, other_attack):
        """Is this attack better than the other attack?"""
        return self.net_gain > other_attack.net_gain

    def __str__(self):
        return "Attack gain: %d extra candidates\n\ttotal candidates: %d\n\taborts: %d" % (self.gain, self.candidates, self.cost)


class Scenario(object):
    def __init__(self, scenario_str, filtered_set_size, sampled_set_size, adversary_strength, n_biasers):
        self.filtered_set_size = filtered_set_size
        self.sampled_set_size = sampled_set_size
        self.adversary_strength = adversary_strength
        # The number of evil last slot RANDAO revealers (if it's zero, we sample it using the adversary strength)
        self.n_biasers = n_biasers

        self.scenario_str = scenario_str
        if scenario_str == "candidate_selection":
            self.scenario_set_size = filtered_set_size
            self.attack = self.get_best_attack_in_candidate_selection
            self.get_net_gain = self._get_candidate_selection_net_gain
        elif scenario_str == "proposer_selection":
            self.scenario_set_size = sampled_set_size
            self.attack = self.get_best_attack_in_proposer_selection
            self.get_net_gain = self._get_proposer_selection_net_gain
        elif scenario_str == "status_quo":
            # In the status quo we filter 32 at each epoch and use them directly (so use proposer_selection net gain func)
            self.scenario_set_size = filtered_set_size
            self.attack = self.get_best_attack_in_status_quo
            self.get_net_gain = self._get_proposer_selection_net_gain

    def _get_n_biasers(self):
        """Return number of evil validators that can bias RANDAO during last epoch"""
        if self.n_biasers > 0:
            return self.n_biasers

        # RANDAO biasers need to sit sequentially on the last slots of the epoch. We model this using a negative
        # binomial distribution. A negative binomial RV is the number X of repeated trials to produce N successes.  We
        # flip this around and ask "What's the number of repeated evil candidates before we produce a single honest
        # proposer".
        biasers = scipy.stats.nbinom.rvs(n=1, p=1-self.adversary_strength)
        return min(biasers, 32) # SLOTS_PER_EPOCH=32

    def _filter(self):
        """
        Number of evil candidates for a 10% adversary in a filtered set of size 10, is the number of heads we get if we
        throw a coin 10 times, and the probability of landing head is 10%.
        """
        return scipy.stats.binom.rvs(self.filtered_set_size, self.adversary_strength)

    def _sample(self, filtered_evil_candidates):
        """
        Start with `filtered_evil_candidates` and return number of candidates that would get past proposer selection. For example,
        for filtered set size 2^14 and sampled set size 2^13, only half of the candidates get past proposer selection.
        """
        return scipy.stats.binom.rvs(n=filtered_evil_candidates, p=self.sampled_set_size/self.filtered_set_size)

    def _get_candidate_selection_net_gain(self, evil_candidates, start_evil_candidates, cost):
        # Candidate selection net gain is the number of extra sampled candidates minus the cost
        sampled_evil_candidates = self._sample(evil_candidates)
        sampled_start_candidates = self._sample(start_evil_candidates)
        return sampled_evil_candidates - sampled_start_candidates - cost

    def _get_proposer_selection_net_gain(self, evil_candidates, start_evil_candidates, cost):
        # Proposer selection net gain is the number of extra sampled candidates minus the cost
        return evil_candidates - cost - start_evil_candidates

    def _get_best_attack(self, shrink_func):
        """
        Return the best attack that this adversary can pull off depending on the shrinking function (either candidate selection or proposer selection)
        """
        # These are the candidates the attacker gets if they don't do a RANDAO abort attack
        start_evil_candidates = shrink_func()
        best_attack = Attack(net_gain=0, evil_candidates=start_evil_candidates, cost=0, start_evil_candidates=start_evil_candidates)

        n_biasers = self._get_n_biasers()
        for i in range(1, 2**n_biasers):
            evil_candidates = shrink_func() # this attack resulted in a fresh set of candidates
            cost = popcount(i) # number of aborts used for this attack
            net_gain = self.get_net_gain(evil_candidates, start_evil_candidates, cost)

            # Check if this is a better attack than what we previously had
            attack = Attack(net_gain, evil_candidates, cost, start_evil_candidates)
            if attack > best_attack:
                best_attack = attack

        return best_attack

    def get_best_attack_in_status_quo(self):
        """In the status quo, we filter from the entire set of validators to 32 validators"""
        assert(self.filtered_set_size == 32)
        return self.get_best_attack_in_candidate_selection()

    def get_best_attack_in_candidate_selection(self):
        """Return the best attack that this adversary can pull off during the filter event."""

        candidate_selection_func = self._filter

        return self._get_best_attack(candidate_selection_func)

    def get_best_attack_in_proposer_selection(self):
        """Return the best attack that this adversary can pull off during the proposer_selection event."""

        # In the proposer_selection shrinking function we first get the filtered set and then we sample out of it.  We do this as
        # a lambda and we capture the filtered set, so that we only filter once but we can still sample multiple times
        # out of it.
        filtered_evil_candidates = self._filter()
        proposer_selection_func = lambda: self._sample(filtered_evil_candidates)

        return self._get_best_attack(proposer_selection_func)

def run_scenario(scenario_str, filtered_set_size, sampled_set_size, adversary_strength, n_biasers, opportunities=1):
    """
    'opportunities' is the number of opportunities the attacker has to attack per run (used in "status quo" scenario)
    'n_biasers' is number of RANDAO biasers the adversary controls

    Returns:
    - Probability of aborting when given the opportunity
    - Average net gain (in proposals) per abort
    - Average total gain per simulation run
    - Average number of aborts per simulation run
    """
    scenario = Scenario(scenario_str, filtered_set_size, sampled_set_size, adversary_strength, n_biasers)

    total_gain = total_aborts = total_evil_candidates = total_start_evil_candidates = n_attacks = 0

    # Each simulation run represents a filter/sample event for "filter"/"sample" scenarios, whereas for the
    # "status_quo" scenario it represents an entire day of proposer selection
    for _ in range(N_RUNS):
        for _ in range(opportunities):
            attack = scenario.attack()

            total_gain += attack.net_gain
            total_aborts += attack.cost
            total_evil_candidates += attack.evil_candidates
            total_start_evil_candidates += attack.start_evil_candidates
            if attack.cost > 0:
                n_attacks += 1

    n_opportunities = N_RUNS*opportunities

    return n_attacks/n_opportunities, total_gain/n_attacks if n_attacks > 0 else 0, total_gain/N_RUNS, total_aborts/N_RUNS

STRENGTHS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.3]

def run():
    graph_candidate_selection_prob_abort = {}
    graph_candidate_selection_extra_proposals = {}
    graph_statusquo_prob_abort = {}
    graph_statusquo_extra_proposals = {}
    graph_candidate_selection_daily_gain = {}
    graph_candidate_selection_daily_aborts = {}
    graph_statusquo_daily_gain = {}
    graph_statusquo_daily_aborts = {}

    for strength in STRENGTHS:
        strength_str = "{:g}".format(strength*100) + "%"

        print("[!] Running candidate selection with n_biasers=1")

        prob_abort, extra_proposals, _, _ = run_scenario("candidate_selection", FILTERED_SET_SIZE, SAMPLED_SET_SIZE, strength, n_biasers=1)
        graph_candidate_selection_prob_abort[strength_str] = prob_abort
        graph_candidate_selection_extra_proposals[strength_str] = extra_proposals

        print("[!] Running status quo with n_biasers=1")

        prob_abort, extra_proposals, _, _ = run_scenario("status_quo", 32, 32, strength, n_biasers=1, opportunities=256)
        graph_statusquo_prob_abort[strength_str] = prob_abort
        graph_statusquo_extra_proposals[strength_str] = extra_proposals

        print("[!] Running candidate selection with n_biasers=0")

        _, _, daily_gain, daily_aborts = run_scenario("candidate_selection", FILTERED_SET_SIZE, SAMPLED_SET_SIZE, strength, n_biasers=0)
        graph_candidate_selection_daily_gain[strength_str] = daily_gain
        graph_candidate_selection_daily_aborts[strength_str] = daily_aborts

        print("[!] Running status quo with n_biasers=0")

        _, _, daily_gain, daily_aborts = run_scenario("status_quo", 32, 32, strength, n_biasers=0, opportunities=256)
        graph_statusquo_daily_gain[strength_str] = daily_gain
        graph_statusquo_daily_aborts[strength_str] = daily_aborts

    # Attack graph

    f = plt.figure()
    f.suptitle("Attack with a single malicious RANDAO revealer", fontsize=42)

    ax1 = f.add_subplot(1,2,1)
    ax1.set_xlabel("Adversary strength", fontsize=30)
    ax1.set_ylabel("Probability of aborting", fontsize=30)
    ax1.set_title("Probability of abort", fontsize=38)
    ax1.plot(graph_candidate_selection_prob_abort.keys(), graph_candidate_selection_prob_abort.values(), label="Whisk")
    ax1.plot(graph_statusquo_prob_abort.keys(), graph_statusquo_prob_abort.values(), label="status quo")
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    ax2 = f.add_subplot(1,2,2)
    ax2.set_xlabel("Adversary strength", fontsize=30)
    ax2.set_ylabel("Extra proposals gained", fontsize=30)
    ax2.set_title("Proposals gained per abort", fontsize=38)
    ax2.plot(graph_candidate_selection_extra_proposals.keys(), graph_candidate_selection_extra_proposals.values())
    ax2.plot(graph_statusquo_extra_proposals.keys(), graph_statusquo_extra_proposals.values())
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    f.legend(loc="lower left", fontsize=30)
    plt.show()


    # Daily graph

    plt.close(f)

    f = plt.figure()
    f.suptitle("Adversary revenue over entire day (256 epochs)", fontsize=42)

    ax1 = f.add_subplot(1,2,1)
    ax1.set_xlabel("Adversary strength", fontsize=30)
    ax1.set_ylabel("Extra proposals gained", fontsize=30)
    ax1.set_title("Net gain for entire day", fontsize=38)
    ax1.plot(graph_candidate_selection_daily_gain.keys(), graph_candidate_selection_daily_gain.values())
    ax1.plot(graph_statusquo_daily_gain.keys(), graph_statusquo_daily_gain.values())
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    ax2 = f.add_subplot(1,2,2)
    ax2.set_xlabel("Adversary strength", fontsize=30)
    ax2.set_ylabel("Aborts", fontsize=30)
    ax2.set_title("Number of aborts over entire day", fontsize=38)
    ax2.plot(graph_candidate_selection_daily_aborts.keys(), graph_candidate_selection_daily_aborts.values(), label="Whisk")
    ax2.plot(graph_statusquo_daily_aborts.keys(), graph_statusquo_daily_aborts.values(), label="status quo")
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    f.legend(loc="lower left", fontsize=30)
    plt.show()

if __name__ == '__main__':
    run()
