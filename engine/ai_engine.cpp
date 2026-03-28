#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

const double TWO_PI = 2.0 * M_PI;

inline double clamp(double v, double lo, double hi) {
    return max(lo, min(hi, v));
}

struct Point {
    double x, y;
    double t = 0;
};

inline double dist(Point a, Point b) {
    return hypot(a.x - b.x, a.y - b.y);
}

mt19937 gen(1337);
normal_distribution<double> d_randn(0.0, 1.0);
uniform_real_distribution<double> d_randu(0.0, 1.0);

inline double randn() {
    return d_randn(gen);
}

vector<Point> computeICBMTrajectory(double launchX, double targetX, double maxAlt) {
    double dx = targetX - launchX;
    int steps = 280;
    vector<Point> pts;
    pts.reserve(steps + 1);
    for (int i = 0; i <= steps; i++) {
        double t = (double)i / steps;
        pts.push_back({launchX + dx * t, maxAlt * 4.0 * t * (1.0 - t), t});
    }
    return pts;
}

struct PolicyNetwork {
    int inputDim, hiddenDim, outputDim;
    vector<double> w1, b1, w2, b2, logStd;

    PolicyNetwork(int iDim = 6, int hDim = 32, int oDim = 2) : inputDim(iDim), hiddenDim(hDim), outputDim(oDim) {
        double xH = sqrt(2.0 / (inputDim + hiddenDim));
        double xO = sqrt(2.0 / (hiddenDim + outputDim));
        w1.resize(inputDim * hiddenDim);
        for(auto& val : w1) val = randn() * xH;
        b1.assign(hiddenDim, 0.0);
        w2.resize(hiddenDim * outputDim);
        for(auto& val : w2) val = randn() * xO;
        b2.assign(outputDim, 0.0);
        logStd.assign(outputDim, -0.5);
    }

    struct Output {
        vector<double> mean;
        vector<double> hidden;
    };

    Output forward(const vector<double>& input) const {
        Output out;
        out.hidden.assign(hiddenDim, 0.0);
        for (int j = 0; j < hiddenDim; j++) {
            double s = b1[j];
            for (int i = 0; i < inputDim; i++) s += input[i] * w1[i * hiddenDim + j];
            out.hidden[j] = max(0.0, s); // ReLU
        }
        out.mean.assign(outputDim, 0.0);
        for (int j = 0; j < outputDim; j++) {
            double s = b2[j];
            for (int i = 0; i < hiddenDim; i++) s += out.hidden[i] * w2[i * outputDim + j];
            out.mean[j] = tanh(s);
        }
        return out;
    }

    struct SampleOut {
        vector<double> actions;
        vector<double> mean;
        vector<double> hidden;
    };

    SampleOut sample(const vector<double>& input) const {
        Output fwd = forward(input);
        SampleOut out;
        out.mean = fwd.mean;
        out.hidden = fwd.hidden;
        out.actions.resize(outputDim);
        for(int i=0; i<outputDim; i++) {
            out.actions[i] = clamp(out.mean[i] + randn() * exp(logStd[i]), -1.0, 1.0);
        }
        return out;
    }

    double logProb(const vector<double>& mean, const vector<double>& actions) const {
        double lp = 0;
        for (int i = 0; i < outputDim; i++) {
            double std = exp(logStd[i]);
            double diff = actions[i] - mean[i];
            lp += -0.5 * (diff * diff) / (std * std) - log(std) - 0.9189385332;
        }
        return lp;
    }

    vector<double> getParams() const {
        vector<double> p;
        p.reserve(w1.size() + b1.size() + w2.size() + b2.size() + logStd.size());
        p.insert(p.end(), w1.begin(), w1.end());
        p.insert(p.end(), b1.begin(), b1.end());
        p.insert(p.end(), w2.begin(), w2.end());
        p.insert(p.end(), b2.begin(), b2.end());
        p.insert(p.end(), logStd.begin(), logStd.end());
        return p;
    }

    void setParams(const vector<double>& p) {
        int k = 0;
        for (auto& v : w1) v = p[k++];
        for (auto& v : b1) v = p[k++];
        for (auto& v : w2) v = p[k++];
        for (auto& v : b2) v = p[k++];
        for (auto& v : logStd) v = p[k++];
    }

    int paramCount() const {
        return w1.size() + b1.size() + w2.size() + b2.size() + logStd.size();
    }
};

struct Experience {
    vector<double> state;
    vector<double> action;
    vector<double> mean;
    double logProb;
};

struct RolloutResult {
    vector<Experience> experience;
    double reward;
    bool hit;
    vector<Point> trail;
    vector<Point> traj;
    int startIdx;
    double minDist;
    vector<vector<double>> states;
    vector<vector<double>> actions;
    vector<vector<double>> means;
};

struct GRPOOptimizer {
    PolicyNetwork policy;
    PolicyNetwork refPolicy;
    double clipRatio;
    double lr;
    double klCoeff;
    int groupSize;
    int epochs;
    int refUpdateFreq;
    int updateCount;

    struct Stats {
        int episodes = 0;
        int hits = 0;
        double avgReward = 0;
        vector<double> rewardHistory;
        vector<double> hitRateHistory;
    } stats;

    GRPOOptimizer() : policy(6, 32, 2), refPolicy(6, 32, 2) {
        refPolicy.setParams(policy.getParams());
        clipRatio = 0.2;
        lr = 0.003;
        klCoeff = 0.04;
        groupSize = 8;
        epochs = 3;
        refUpdateFreq = 5;
        updateCount = 0;
    }

    RolloutResult rollout(double launchX, double targetX, double maxAlt, double radarRange, double interceptSpeed, int launchDelay) {
        RolloutResult res;
        res.traj = computeICBMTrajectory(launchX, targetX, maxAlt);
        Point tgt = {targetX, 0.0};
        
        int detectIdx = -1;
        for (int i = 0; i < (int)res.traj.size(); i++) {
            if (dist(res.traj[i], tgt) < radarRange) {
                detectIdx = i;
                break;
            }
        }
        if (detectIdx < 0) detectIdx = res.traj.size() * 0.35;
        res.startIdx = min(detectIdx + launchDelay, (int)res.traj.size() - 8);

        double ix = tgt.x, iy = 0.0, ivx = 0.0, ivy = 0.0;
        double maxA = interceptSpeed * 0.15;
        res.trail.push_back({ix, iy});
        res.hit = false;
        res.minDist = 1e9;

        for (int step = res.startIdx; step < (int)res.traj.size(); step++) {
            Point t = res.traj[step];
            int fi = min(step + 3, (int)res.traj.size() - 1);
            Point ft = res.traj[fi];
            
            vector<double> state = {
                (t.x - ix) / 400.0,
                (t.y - iy) / 400.0,
                ivx / interceptSpeed,
                ivy / interceptSpeed,
                (ft.x - ix) / 400.0,
                (ft.y - iy) / 400.0
            };

            auto sample = policy.sample(state);
            double lp = policy.logProb(sample.mean, sample.actions);

            ivx += sample.actions[0] * maxA;
            ivy += sample.actions[1] * maxA;
            double spd = hypot(ivx, ivy);
            if (spd > interceptSpeed) {
                ivx *= interceptSpeed / spd;
                ivy *= interceptSpeed / spd;
            }
            ix += ivx;
            iy += ivy;
            iy = max(0.0, iy);
            res.trail.push_back({ix, iy});
            res.experience.push_back({state, sample.actions, sample.mean, lp});

            double d = dist({ix, iy}, t);
            res.minDist = min(res.minDist, d);
            if (d < 18.0 && t.y > 5.0) {
                res.hit = true;
                res.traj.resize(step + 1);
                break;
            }
            if (t.y <= 5.0) break; // ICBM impacted target
            if (iy <= 0.0 && step > res.startIdx + 8) break;
        }

        res.reward = res.hit ? 200.0 : -50.0;
        res.reward -= res.minDist * 0.3;
        if (res.hit) {
            res.reward += (res.traj.size() - res.experience.size()) * 0.5;
        }
        return res;
    }

    json update(double launchX, double targetX, double maxAlt, double radarRange, double interceptSpeed, int launchDelay) {
        int G = groupSize;
        vector<RolloutResult> rollouts;
        for (int g = 0; g < G; g++) {
            rollouts.push_back(rollout(launchX, targetX, maxAlt, radarRange, interceptSpeed, launchDelay));
        }

        double sumR = 0;
        for (auto& r : rollouts) sumR += r.reward;
        double meanR = sumR / G;
        double stdR = 0;
        for (auto& r : rollouts) stdR += (r.reward - meanR) * (r.reward - meanR);
        stdR = sqrt(stdR / G + 1e-8);

        vector<double> advantages(G);
        for (int g = 0; g < G; g++) advantages[g] = (rollouts[g].reward - meanR) / stdR;

        int nP = policy.paramCount();
        vector<double> grad(nP, 0.0);
        double epsilon = 0.005;

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int g = 0; g < G; g++) {
                double adv = advantages[g];
                auto& expList = rollouts[g].experience;
                
                vector<Experience> sampleSteps;
                if (expList.size() > 15) {
                    int stepSize = ceil(expList.size() / 12.0);
                    for (int i = 0; i < (int)expList.size(); i += stepSize) sampleSteps.push_back(expList[i]);
                } else {
                    sampleSteps = expList;
                }

                if (sampleSteps.empty()) continue;

                for (auto& step : sampleSteps) {
                    vector<double> params = policy.getParams();
                    vector<double> delta(nP);
                    for(int i=0; i<nP; i++) delta[i] = (d_randu(gen) > 0.5) ? 1.0 : -1.0;

                    auto evalLoss = [&](const vector<double>& pArr) {
                        policy.setParams(pArr);
                        auto fwd = policy.forward(step.state);
                        double clp = policy.logProb(fwd.mean, step.action);
                        double ratio = exp(clamp(clp - step.logProb, -10.0, 10.0));
                        double clipR = clamp(ratio, 1.0 - clipRatio, 1.0 + clipRatio);
                        auto rm = refPolicy.forward(step.state).mean;
                        double rlp = refPolicy.logProb(rm, step.action);
                        return min(ratio * adv, clipR * adv) - klCoeff * (clp - rlp);
                    };

                    vector<double> pPlus(nP), pMinus(nP);
                    for(int i=0; i<nP; i++) {
                        pPlus[i] = params[i] + epsilon * delta[i];
                        pMinus[i] = params[i] - epsilon * delta[i];
                    }

                    double lPlus = evalLoss(pPlus);
                    double lMinus = evalLoss(pMinus);
                    policy.setParams(params);

                    double gEst = (lPlus - lMinus) / (2.0 * epsilon);
                    double scale = 1.0 / (G * sampleSteps.size() * epochs);
                    for (int i = 0; i < nP; i++) grad[i] += gEst * delta[i] * scale;
                }
            }
        }

        vector<double> params = policy.getParams();
        for (int i = 0; i < nP; i++) params[i] += lr * grad[i];
        policy.setParams(params);

        updateCount++;
        if (updateCount % refUpdateFreq == 0) {
            refPolicy.setParams(policy.getParams());
        }

        int hits = 0;
        for (auto& r : rollouts) {
            if (r.hit) hits++;
        }

        stats.episodes += G;
        stats.hits += hits;
        stats.avgReward = meanR;
        stats.rewardHistory.push_back(meanR);
        stats.hitRateHistory.push_back((double)hits / G);

        json out;
        out["type"] = "update";
        out["stats"] = {
            {"episodes", stats.episodes},
            {"hits", stats.hits},
            {"avgReward", stats.avgReward},
            {"rewardHistory", stats.rewardHistory},
            {"hitRateHistory", stats.hitRateHistory}
        };
        return out;
    }
};

struct ESOptimizer {
    PolicyNetwork policy;
    double lr;
    double sigma;
    int popSize;

    struct Stats {
        int episodes = 0;
        int hits = 0;
        double avgReward = 0;
        vector<double> rewardHistory;
        vector<double> hitRateHistory;
    } stats;

    ESOptimizer() : policy(6, 32, 2) {
        lr = 0.05;
        sigma = 0.1;
        popSize = 32;
    }

    RolloutResult rollout(const PolicyNetwork& net, double launchX, double targetX, double maxAlt, double radarRange, double interceptSpeed, int launchDelay) {
        RolloutResult res;
        res.traj = computeICBMTrajectory(launchX, targetX, maxAlt);
        Point tgt = {targetX, 0.0};
        
        int detectIdx = -1;
        for (int i = 0; i < (int)res.traj.size(); i++) {
            if (dist(res.traj[i], tgt) < radarRange) { detectIdx = i; break; }
        }
        if (detectIdx < 0) detectIdx = res.traj.size() * 0.35;
        res.startIdx = min(detectIdx + launchDelay, (int)res.traj.size() - 8);

        double ix = tgt.x, iy = 0.0, ivx = 0.0, ivy = 0.0;
        double maxA = interceptSpeed * 0.15;
        res.trail.push_back({ix, iy});
        res.hit = false;
        res.minDist = 1e9;

        for (int step = res.startIdx; step < (int)res.traj.size(); step++) {
            Point t = res.traj[step];
            int fi = min(step + 3, (int)res.traj.size() - 1);
            Point ft = res.traj[fi];
            vector<double> state = { (t.x - ix)/400.0, (t.y - iy)/400.0, ivx/interceptSpeed, ivy/interceptSpeed, (ft.x - ix)/400.0, (ft.y - iy)/400.0 };

            auto fwd = net.forward(state);
            vector<double> actions = fwd.mean; 

            ivx += actions[0] * maxA; ivy += actions[1] * maxA;
            double spd = hypot(ivx, ivy);
            if (spd > interceptSpeed) { ivx *= interceptSpeed / spd; ivy *= interceptSpeed / spd; }
            ix += ivx; iy += ivy; iy = max(0.0, iy);
            res.trail.push_back({ix, iy});
            
            double d = dist({ix, iy}, t);
            res.minDist = min(res.minDist, d);
            if (d < 18.0 && t.y > 5.0) { res.hit = true; res.traj.resize(step + 1); break; }
            if (t.y <= 5.0) break; // ICBM impacted target
            if (iy <= 0.0 && step > res.startIdx + 8) break;
        }
        res.reward = res.hit ? 200.0 : -50.0;
        res.reward -= res.minDist * 0.3;
        if (res.hit) res.reward += (res.traj.size() - res.trail.size()) * 0.5;
        return res;
    }

    json update(double launchX, double targetX, double maxAlt, double radarRange, double interceptSpeed, int launchDelay, bool randomize) {
        int nP = policy.paramCount();
        vector<double> params = policy.getParams();
        vector<vector<double>> epsilon(popSize, vector<double>(nP));
        vector<double> rewards(popSize);
        int hits = 0;
        double sumR = 0;

        for (int i = 0; i < popSize; i++) {
            vector<double> jitteredParams = params;
            for (int j = 0; j < nP; j++) {
                epsilon[i][j] = randn();
                jitteredParams[j] += sigma * epsilon[i][j];
            }
            PolicyNetwork evalNet = policy;
            evalNet.setParams(jitteredParams);
            
            double lX = launchX + (randomize ? (d_randu(gen) - 0.5) * 80.0 : 0);
            double tX = targetX + (randomize ? (d_randu(gen) - 0.5) * 80.0 : 0);
            double mA = maxAlt + (randomize ? (d_randu(gen) - 0.5) * 60.0 : 0);
            
            RolloutResult r = rollout(evalNet, lX, tX, mA, radarRange, interceptSpeed, launchDelay);
            rewards[i] = r.reward;
            if (r.hit) hits++;
            sumR += r.reward;
        }

        double meanR = sumR / popSize;
        double stdR = 0;
        for(double r : rewards) stdR += (r - meanR) * (r - meanR);
        stdR = sqrt(stdR / popSize + 1e-8);
        vector<double> A(popSize);
        for(int i=0; i<popSize; i++) A[i] = (rewards[i] - meanR) / stdR;

        vector<double> grad(nP, 0.0);
        for (int j = 0; j < nP; j++) {
            for (int i = 0; i < popSize; i++) {
                grad[j] += A[i] * epsilon[i][j];
            }
            grad[j] /= (popSize * sigma);
            params[j] += lr * grad[j];
        }
        policy.setParams(params);

        stats.episodes += popSize;
        stats.hits += hits;
        stats.avgReward = meanR;
        stats.rewardHistory.push_back(meanR);
        stats.hitRateHistory.push_back((double)hits / popSize);

        json out;
        out["type"] = "update";
        out["stats"] = {
            {"episodes", stats.episodes},
            {"hits", stats.hits},
            {"avgReward", stats.avgReward},
            {"rewardHistory", stats.rewardHistory},
            {"hitRateHistory", stats.hitRateHistory}
        };
        return out;
    }
};

struct PPOOptimizer {
    int inD = 6, hidD = 32, actD = 2;
    // Actor
    vector<vector<double>> W1_a, W2_a;
    vector<double> b1_a, b2_a;
    // Critic
    vector<vector<double>> W1_c, W2_c;
    vector<double> b1_c, b2_c;

    double lr = 0.001;
    double gamma = 0.99;
    double clipRatio = 0.2;
    double sigma = 0.2; // Fixed std for simplicity
    int ppoEpochs = 4;

    struct Stats {
        int episodes = 0;
        int hits = 0;
        double avgReward = 0;
        vector<double> rewardHistory;
        vector<double> hitRateHistory;
    } stats;

    PPOOptimizer() {
        auto initW = [](int in, int out) {
            vector<vector<double>> W(out, vector<double>(in));
            double bound = sqrt(6.0 / (in + out));
            for(int i=0; i<out; i++) for(int j=0; j<in; j++) W[i][j] = (d_randu(gen)*2.0 - 1.0) * bound;
            return W;
        };
        W1_a = initW(inD, hidD); W2_a = initW(hidD, actD);
        b1_a.assign(hidD, 0.0); b2_a.assign(actD, 0.0);
        
        W1_c = initW(inD, hidD); W2_c = initW(hidD, 1);
        b1_c.assign(hidD, 0.0); b2_c.assign(1, 0.0);
    }

    void forwardActor(const vector<double>& x, vector<double>& a1, vector<double>& mean) {
        a1.assign(hidD, 0.0); mean.assign(actD, 0.0);
        for(int i=0; i<hidD; i++) {
            double sum = b1_a[i];
            for(int j=0; j<inD; j++) sum += W1_a[i][j] * x[j];
            a1[i] = max(0.0, sum); // ReLU
        }
        for(int i=0; i<actD; i++) {
            double sum = b2_a[i];
            for(int j=0; j<hidD; j++) sum += W2_a[i][j] * a1[j];
            mean[i] = tanh(sum);
        }
    }

    double forwardCritic(const vector<double>& x) {
        vector<double> a1(hidD, 0.0);
        for(int i=0; i<hidD; i++) {
            double sum = b1_c[i];
            for(int j=0; j<inD; j++) sum += W1_c[i][j] * x[j];
            a1[i] = max(0.0, sum);
        }
        double v = b2_c[0];
        for(int j=0; j<hidD; j++) v += W2_c[0][j] * a1[j];
        return v;
    }

    RolloutResult rollout(double launchX, double targetX, double maxAlt, double radarRange, double interceptSpeed, int launchDelay) {
        RolloutResult res;
        res.traj = computeICBMTrajectory(launchX, targetX, maxAlt);
        Point tgt = {targetX, 0.0};
        
        int detectIdx = -1;
        for (int i = 0; i < (int)res.traj.size(); i++) {
            if (dist(res.traj[i], tgt) < radarRange) { detectIdx = i; break; }
        }
        if (detectIdx < 0) detectIdx = res.traj.size() * 0.35;
        res.startIdx = min(detectIdx + launchDelay, (int)res.traj.size() - 8);

        double ix = tgt.x, iy = 0.0, ivx = 0.0, ivy = 0.0;
        double maxA = interceptSpeed * 0.15;
        res.trail.push_back({ix, iy});
        res.hit = false;
        res.minDist = 1e9;

        for (int step = res.startIdx; step < (int)res.traj.size(); step++) {
            Point t = res.traj[step];
            int fi = min(step + 3, (int)res.traj.size() - 1);
            Point ft = res.traj[fi];
            vector<double> state = { (t.x - ix)/400.0, (t.y - iy)/400.0, ivx/interceptSpeed, ivy/interceptSpeed, (ft.x - ix)/400.0, (ft.y - iy)/400.0 };

            vector<double> a1, mean;
            forwardActor(state, a1, mean);
            
            vector<double> action = { mean[0] + randn() * sigma, mean[1] + randn() * sigma };

            ivx += action[0] * maxA; ivy += action[1] * maxA;
            double spd = hypot(ivx, ivy);
            if (spd > interceptSpeed) { ivx *= interceptSpeed / spd; ivy *= interceptSpeed / spd; }
            ix += ivx; iy += ivy; iy = max(0.0, iy);
            res.trail.push_back({ix, iy});
            
            res.states.push_back(state);
            res.actions.push_back(action);
            res.means.push_back(mean);
            
            double d = dist({ix, iy}, t);
            res.minDist = min(res.minDist, d);
            if (d < 18.0 && t.y > 5.0) { res.hit = true; res.traj.resize(step + 1); break; }
            if (t.y <= 5.0) break; // ICBM impacted target
            if (iy <= 0.0 && step > res.startIdx + 8) break;
        }
        res.reward = res.hit ? 200.0 : -50.0;
        res.reward -= res.minDist * 0.3;
        if (res.hit) res.reward += (res.traj.size() - res.trail.size()) * 0.5;
        return res;
    }

    json update(double launchX, double targetX, double maxAlt, double radarRange, double interceptSpeed, int launchDelay, bool randomize, int batchSize) {
        vector<RolloutResult> batch;
        int hits = 0;
        double sumR = 0;
        
        for(int b=0; b<batchSize; b++) {
            double lX = launchX + (randomize ? (d_randu(gen) - 0.5) * 80.0 : 0);
            double tX = targetX + (randomize ? (d_randu(gen) - 0.5) * 80.0 : 0);
            double mA = maxAlt + (randomize ? (d_randu(gen) - 0.5) * 60.0 : 0);
            RolloutResult r = rollout(lX, tX, mA, radarRange, interceptSpeed, launchDelay);
            if(r.hit) hits++;
            sumR += r.reward;
            batch.push_back(r);
        }

        // Compute Advantages
        vector<vector<double>> allStates;
        vector<vector<double>> allActions;
        vector<vector<double>> allOldMeans;
        vector<double> allAdvs, allReturns;

        for (auto& r : batch) {
            int T = r.states.size();
            vector<double> values(T+1, 0.0);
            for(int t=0; t<T; t++) values[t] = forwardCritic(r.states[t]);
            
            double ret = r.reward; 
            double lastV = 0.0;
            double lastAdv = 0.0;
            double lam = 0.95;
            for(int t=T-1; t>=0; t--) {
                double r_t = (t == T-1) ? ret : 0.0;
                double delta = r_t + gamma * lastV - values[t];
                double adv = delta + gamma * lam * lastAdv;
                
                allStates.push_back(r.states[t]);
                allActions.push_back(r.actions[t]);
                allOldMeans.push_back(r.means[t]);
                allAdvs.push_back(adv);
                allReturns.push_back(adv + values[t]);
                
                lastV = values[t];
                lastAdv = adv;
            }
        }

        double meanAdv = 0; for(double a : allAdvs) meanAdv += a;
        meanAdv /= allAdvs.size();
        double stdAdv = 0; for(double a : allAdvs) stdAdv += (a - meanAdv)*(a - meanAdv);
        stdAdv = sqrt(stdAdv / allAdvs.size() + 1e-8);
        for(double& a : allAdvs) a = (a - meanAdv) / stdAdv;

        int N = allStates.size();
        for(int epoch=0; epoch<ppoEpochs; epoch++) {
            // Batch Gradients
            vector<vector<double>> dW1_a(hidD, vector<double>(inD, 0.0));
            vector<vector<double>> dW2_a(actD, vector<double>(hidD, 0.0));
            vector<double> db1_a(hidD, 0.0), db2_a(actD, 0.0);
            
            vector<vector<double>> dW1_c(hidD, vector<double>(inD, 0.0));
            vector<vector<double>> dW2_c(1, vector<double>(hidD, 0.0));
            vector<double> db1_c(hidD, 0.0), db2_c(1, 0.0);

            for(int i=0; i<N; i++) {
                vector<double> state = allStates[i];
                vector<double> a1, mean;
                forwardActor(state, a1, mean);
                
                vector<double> oldMean = allOldMeans[i];
                vector<double> action = allActions[i];
                double adv = allAdvs[i];
                
                double logp = 0, old_logp = 0;
                for(int d=0; d<actD; d++) {
                    logp += -0.5 * pow((action[d] - mean[d])/sigma, 2);
                    old_logp += -0.5 * pow((action[d] - oldMean[d])/sigma, 2);
                }
                double ratio = exp(logp - old_logp);
                
                double clippedRatio = max(1.0 - clipRatio, min(1.0 + clipRatio, ratio));
                double loss_actor = min(ratio * adv, clippedRatio * adv);
                
                vector<double> dL_dmean(actD, 0.0);
                if (ratio * adv <= clippedRatio * adv) {
                    for(int d=0; d<actD; d++) {
                        dL_dmean[d] = adv * ratio * (action[d] - mean[d]) / (sigma * sigma);
                    }
                }
                
                vector<double> dZ2(actD);
                for(int d=0; d<actD; d++) dZ2[d] = dL_dmean[d] * (1.0 - mean[d]*mean[d]);
                
                vector<double> dZ1(hidD, 0.0);
                for(int d=0; d<actD; d++) {
                    db2_a[d] += dZ2[d];
                    for(int h=0; h<hidD; h++) {
                        dZ1[h] += W2_a[d][h] * dZ2[d]; // backward pass FIRST
                        dW2_a[d][h] += dZ2[d] * a1[h]; // then compute grad
                    }
                }
                for(int h=0; h<hidD; h++) {
                    dZ1[h] = (a1[h] > 0) ? dZ1[h] : 0.0; 
                    db1_a[h] += dZ1[h];
                    for(int f=0; f<inD; f++) {
                        dW1_a[h][f] += dZ1[h] * state[f];
                    }
                }

                vector<double> c_a1(hidD, 0.0);
                for(int h=0; h<hidD; h++) {
                    double sum = b1_c[h];
                    for(int f=0; f<inD; f++) sum += W1_c[h][f] * state[f];
                    c_a1[h] = max(0.0, sum);
                }
                double v = b2_c[0];
                for(int h=0; h<hidD; h++) v += W2_c[0][h] * c_a1[h];
                
                double td_err = allReturns[i] - v; 
                double c_dZ2 = td_err;
                vector<double> c_dZ1(hidD, 0.0);
                db2_c[0] += c_dZ2;
                for(int h=0; h<hidD; h++) {
                    c_dZ1[h] += W2_c[0][h] * c_dZ2;
                    dW2_c[0][h] += c_dZ2 * c_a1[h];
                }
                for(int h=0; h<hidD; h++) {
                    c_dZ1[h] = (c_a1[h] > 0) ? c_dZ1[h] : 0.0;
                    db1_c[h] += c_dZ1[h];
                    for(int f=0; f<inD; f++) {
                        dW1_c[h][f] += c_dZ1[h] * state[f];
                    }
                }
            }

            double scale = lr / N;
            for(int d=0; d<actD; d++) {
                b2_a[d] += scale * db2_a[d];
                for(int h=0; h<hidD; h++) W2_a[d][h] += scale * dW2_a[d][h];
            }
            for(int h=0; h<hidD; h++) {
                b1_a[h] += scale * db1_a[h];
                for(int f=0; f<inD; f++) W1_a[h][f] += scale * dW1_a[h][f];
            }
            b2_c[0] += scale * db2_c[0];
            for(int h=0; h<hidD; h++) W2_c[0][h] += scale * dW2_c[0][h];
            for(int h=0; h<hidD; h++) {
                b1_c[h] += scale * db1_c[h];
                for(int f=0; f<inD; f++) W1_c[h][f] += scale * dW1_c[h][f];
            }
        }

        double meanR = sumR / batchSize;
        stats.episodes += batchSize;
        stats.hits += hits;
        stats.avgReward = meanR;
        stats.rewardHistory.push_back(meanR);
        stats.hitRateHistory.push_back((double)hits / batchSize);

        json out;
        out["type"] = "update";
        out["stats"] = {
            {"episodes", stats.episodes},
            {"hits", stats.hits},
            {"avgReward", stats.avgReward},
            {"rewardHistory", stats.rewardHistory},
            {"hitRateHistory", stats.hitRateHistory}
        };
        return out;
    }
};

int main() {
    GRPOOptimizer optGRPO;
    ESOptimizer optES;
    PPOOptimizer optPPO;
    
    string line;
    while (getline(cin, line)) {
        if (line.empty()) continue;
        try {
            json msg = json::parse(line);
            string cmd = msg.value("cmd", "");
            
            if (cmd == "update") {
                double launchX = msg["icbmP"]["launchX"];
                double targetX = msg["icbmP"]["targetX"];
                double maxAlt = msg["icbmP"]["maxAlt"];
                double radarRange = msg["defP"]["radarRange"];
                double interceptSpeed = msg["defP"]["interceptSpeed"];
                int launchDelay = msg["defP"]["launchDelay"];
                
                string algo = msg["opt"].value("algo", "GRPO");
                bool randomize = msg["opt"].value("randomize", false);
                
                if (algo == "PPO") {
                    optPPO.lr = msg["opt"].value("lr", optPPO.lr);
                    optPPO.clipRatio = msg["opt"].value("clipRatio", optPPO.clipRatio);
                    int bSize = msg["opt"].value("groupSize", 32);
                    json res = optPPO.update(launchX, targetX, maxAlt, radarRange, interceptSpeed, launchDelay, randomize, bSize);
                    cout << res.dump() << endl;
                } else if (algo == "ES") {
                    optES.popSize = msg["opt"].value("groupSize", optES.popSize);
                    optES.lr = msg["opt"].value("lr", optES.lr);
                    optES.sigma = msg["opt"].value("sigma", optES.sigma);
                    json res = optES.update(launchX, targetX, maxAlt, radarRange, interceptSpeed, launchDelay, randomize);
                    cout << res.dump() << endl;
                } else {
                    optGRPO.groupSize = msg["opt"].value("groupSize", optGRPO.groupSize);
                    optGRPO.clipRatio = msg["opt"].value("clipRatio", optGRPO.clipRatio);
                    optGRPO.klCoeff = msg["opt"].value("klCoeff", optGRPO.klCoeff);
                    optGRPO.lr = msg["opt"].value("lr", optGRPO.lr);
                    
                    double lX = launchX + (randomize ? (d_randu(gen) - 0.5) * 80.0 : 0);
                    double tX = targetX + (randomize ? (d_randu(gen) - 0.5) * 80.0 : 0);
                    double mA = maxAlt + (randomize ? (d_randu(gen) - 0.5) * 60.0 : 0);
                    
                    json res = optGRPO.update(lX, tX, mA, radarRange, interceptSpeed, launchDelay);
                    cout << res.dump() << endl;
                }
            } 
            else if (cmd == "rollout") {
                double launchX = msg["icbmP"]["launchX"];
                double targetX = msg["icbmP"]["targetX"];
                double maxAlt = msg["icbmP"]["maxAlt"];
                double radarRange = msg["defP"]["radarRange"];
                double interceptSpeed = msg["defP"]["interceptSpeed"];
                int launchDelay = msg["defP"]["launchDelay"];
                string algo = msg["opt"].value("algo", "GRPO");
                
                RolloutResult r;
                if (algo == "PPO") {
                    r = optPPO.rollout(launchX, targetX, maxAlt, radarRange, interceptSpeed, launchDelay);
                } else if (algo == "ES") {
                    r = optES.rollout(optES.policy, launchX, targetX, maxAlt, radarRange, interceptSpeed, launchDelay);
                } else {
                    vector<double> oldStd = optGRPO.policy.logStd;
                    optGRPO.policy.logStd.assign(optGRPO.policy.outputDim, -4.0);
                    r = optGRPO.rollout(launchX, targetX, maxAlt, radarRange, interceptSpeed, launchDelay);
                    optGRPO.policy.logStd = oldStd;
                }
                
                json res;
                res["type"] = "rollout";
                res["hit"] = r.hit;
                res["reward"] = r.reward;
                res["startIdx"] = r.startIdx;
                res["minDist"] = r.minDist;
                
                res["traj"] = json::array();
                for (auto& p : r.traj) res["traj"].push_back({{"x", p.x}, {"y", p.y}});
                
                res["trail"] = json::array();
                for (auto& p : r.trail) res["trail"].push_back({{"x", p.x}, {"y", p.y}});
                
                cout << res.dump() << endl;
            }
            else if (cmd == "reset") {
                optGRPO = GRPOOptimizer();
                optES = ESOptimizer();
                optPPO = PPOOptimizer();
                json res;
                res["type"] = "reset";
                cout << res.dump() << endl;
            }
        } catch(const exception& e) {
            json err;
            err["type"] = "error";
            err["message"] = e.what();
            cout << err.dump() << endl;
        }
    }
    return 0;
}
