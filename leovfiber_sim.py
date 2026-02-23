import simpy
import random
import numpy as np
import matplotlib.pyplot as plot

#Configuration
packet_count = 1000
mean_queue_delay = 2.0

Speeds = {
    "Fiber": 200000,
    "Light": 300000
}

Jitters = {
    "Min": 0,
    "Max": 10
}

SatelliteConfig = {
    "NumHops":10,
    "SatAlt": 550,
    "InterSatDistance":1000,
    "SatProcMin": 0.5,
    "SatProcMax": 2.0
}

GroundSatTimeConfig = {
    "UpMin":7,
    "UpMax": 20,
    "DownMin":4,
    "DownMax":12
}

#Distance Profiles (used to simulate different locations)
DistanceProfiles = {
    "Milan": {"ID": 11500, "IT": 210},
    "Singapore": {"ID": 900, "IT": 10200},
    "Midpoint": {"ID": 5750, "IT": 5750},
    "Bahrain": {"ID": 7700, "IT": 4200}
}


def jitter():
    return random.uniform(Jitters['Min'], Jitters['Max'])

def queue_delay():
    return random.expovariate(1 / mean_queue_delay)

def fiber_delay(distance):
    propagation = (distance / Speeds['Fiber']) * 1000
    processing = random.uniform(0.1, 0.5)
    return propagation + processing + queue_delay()

def leo_delay(distance, direction):
    delay = 0

    # Ground <-> satellite propagation
    distance_to_ground = np.sqrt(distance**2 + SatelliteConfig['SatAlt']**2)
    delay += (distance_to_ground / Speeds['Light']) * 1000

    # Ground processing
    if direction == "up":
        delay += random.uniform(GroundSatTimeConfig["UpMin"], GroundSatTimeConfig['UpMax'])
    else:
        delay += random.uniform(GroundSatTimeConfig['DownMin'], GroundSatTimeConfig['DownMax'])

    # Inter-satellite hops
    for _ in range(SatelliteConfig['NumHops'] - 1):
        delay += (SatelliteConfig['InterSatDistance'] / Speeds['Light']) * 1000
        delay += random.uniform(SatelliteConfig['SatProcMin'], SatelliteConfig['SatProcMax'])

    # Downlink
    delay += (distance_to_ground / Speeds['Light']) * 1000
    delay += queue_delay()

    return delay

def hybrid_delay(distance_km, direction):
    return (
        leo_delay(distance_km * 0.1, direction)
        + fiber_delay(distance_km * 0.9)
    )

def packet(env, up, down, rtt_log):
    start = env.now
    if (yield env.process(up.transmit())):
        if (yield env.process(down.transmit())):
            rtt_log.append((env.now - start) * 1000)

def summarize(name, RTTs): #prints results on cmd
    arr = np.array(RTTs)
    print(f"{name}")
    print(f"  Received: {len(arr)}/{packet_count}")
    print(f"  Loss: {100*(1-len(arr)/packet_count):.2f}%")
    if len(arr) > 1:
        jitter = np.mean(np.abs(np.diff(arr)))
        print(f"  Avg RTT: {np.mean(arr):.2f} ms")
        print(f"  Median RTT: {np.median(arr):.2f} ms")
        print(f"  95th pct: {np.percentile(arr,95):.2f} ms")
        print(f"  Jitter: {jitter:.2f} ms")
    print("")
    
class MarkovLossModel:
    def __init__(self, good_loss, bad_loss, p_to_bad, p_to_good):
        self.good_loss = good_loss
        self.bad_loss = bad_loss
        self.p_to_bad = p_to_bad
        self.p_to_good = p_to_good
        self.state = "good"

    def lost(self):
        if self.state == "good":
            if random.random() < self.p_to_bad:
                self.state = "bad"
            return random.random() < self.good_loss
        else:
            if random.random() < self.p_to_good:
                self.state = "good"
            return random.random() < self.bad_loss
        
class NetworkPath:
    def __init__(self, env, delay_fn, loss_model):
        self.env = env
        self.delay_fn = delay_fn
        self.loss_model = loss_model

    def transmit(self):
        if self.loss_model.lost():
            return False

        delay = self.delay_fn() + jitter()
        yield self.env.timeout(delay / 1000)
        return True

def hybrid_loss(fiber_loss, leo_loss): #calculate MarkovLossModel for the hybrid approach
    good_loss_hybrid = 1 - (1 - fiber_loss.good_loss) * (1 - leo_loss.good_loss)
    bad_loss_hybrid  = 1 - (1 - fiber_loss.bad_loss)  * (1 - leo_loss.bad_loss)
    p_to_bad_hybrid  = (fiber_loss.p_to_bad + leo_loss.p_to_bad) / 2
    p_to_good_hybrid = (fiber_loss.p_to_good + leo_loss.p_to_good) / 2

    return MarkovLossModel(good_loss_hybrid, bad_loss_hybrid, p_to_bad_hybrid, p_to_good_hybrid)


RESULTS = {}

#RUNNING SIMULATIONS
def run_scenario(label, ID_delay, IT_delay ):
    env = simpy.Environment()

    fiber_loss = MarkovLossModel(0.002, 0.02, 0.01, 0.2)
    leo_loss = MarkovLossModel(0.01, 0.08, 0.05, 0.15)
    
    if "Fiber" in label:
        loss = fiber_loss
    elif "LEO" in label:
        loss = leo_loss
    else:
        loss = hybrid_loss(fiber_loss,leo_loss) 


    rtt_ID, rtt_IT = [], []

    ID_up = NetworkPath(env, lambda: ID_delay("up"), loss)
    ID_down = NetworkPath(env, lambda: ID_delay("down"), loss)

    IT_up = NetworkPath(env, lambda: IT_delay("up"), loss)
    IT_down = NetworkPath(env, lambda: IT_delay("down"), loss)

    for _ in range(packet_count):
        env.process(packet(env, ID_up, ID_down, rtt_ID))
        env.process(packet(env, IT_up, IT_down, rtt_IT))

    env.run()

    RESULTS[(label, "Indonesia")] = rtt_ID
    RESULTS[(label, "Italy")] = rtt_IT

    summarize(f"{label} – Indonesia", rtt_ID)
    summarize(f"{label} – Italy", rtt_IT)

for server, d in DistanceProfiles.items():
    run_scenario(
        f"Fiber – {server}",
        lambda _: fiber_delay(d["ID"]),
        lambda _: fiber_delay(d["IT"])
    )

    run_scenario(
        f"LEO – {server}",
        lambda dir: leo_delay(d["ID"], dir),
        lambda dir: leo_delay(d["IT"], dir)
    )

    run_scenario(
        f"Hybrid – {server}",
        lambda dir: hybrid_delay(d["ID"], dir),
        lambda dir: hybrid_delay(d["IT"], dir)
    )
    
    
#GRAPHING    

labels, data = [], []

for (scenario, user), rtts in RESULTS.items():
    if rtts:
        labels.append(f"{scenario}\n{user}")
        data.append(rtts)

# Boxplot graph
plot.figure(figsize=(18,6))
plot.boxplot(data, showfliers=False)
plot.xticks(range(1, len(labels)+1), labels, rotation=45, ha="right")
plot.ylabel("RTT (ms)")
plot.title("RTT Distribution Across Scenarios")
plot.grid(axis="y", linestyle="--", alpha=0.6)
plot.tight_layout()
plot.show()


# CDF graph
plot.figure(figsize=(12,6))
for (scenario, user), rtts in RESULTS.items():
    if rtts:
        sorted_rtts = np.sort(rtts)
        cdf = np.arange(1, len(sorted_rtts)+1) / len(sorted_rtts)
        plot.plot(sorted_rtts, cdf, label=f"{scenario} – {user}")

plot.xlabel("RTT (ms)")
plot.ylabel("CDF")
plot.title("CDF of RTT Across Scenarios")
plot.grid(True, linestyle="--", alpha=0.5)
plot.legend(fontsize=8)
plot.tight_layout()
plot.show()

# Packet Loss graph
loss_percentages = []
labels_loss = []
for (scenario, user), rtts in RESULTS.items():
    loss = 100*(1 - len(rtts)/packet_count)
    loss_percentages.append(loss)
    labels_loss.append(f"{scenario}\n{user}")

plot.figure(figsize=(18,6))
plot.bar(labels_loss, loss_percentages, color='skyblue')
plot.xticks(rotation=45, ha="right")
plot.ylabel("Packet Loss (%)")
plot.title("Packet Loss Across Scenarios")
plot.grid(axis="y", linestyle="--", alpha=0.5)
plot.tight_layout()
plot.show()

# Jitter graph
avg_jitters = []
labels_jitter = []
for (scenario, user), rtts in RESULTS.items():
    if len(rtts) > 1:
        jitter = np.mean(np.abs(np.diff(rtts)))
    else:
        jitter = 0
    avg_jitters.append(jitter)
    labels_jitter.append(f"{scenario}\n{user}")

plot.figure(figsize=(18,6))
plot.bar(labels_jitter, avg_jitters, color='lightcoral')
plot.xticks(rotation=45, ha="right")
plot.ylabel("Average Jitter (ms)")
plot.title("Average Jitter Across Scenarios")
plot.grid(axis="y", linestyle="--", alpha=0.5)
plot.tight_layout()
plot.show()

    