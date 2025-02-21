import dimod

# Financial Data (example values for 2023)
revenue = 500  # in millions
assets = 1200  # in millions
liabilities = 800  # in millions

# Weights for valuation components
alpha = 1.0  # Revenue weight
beta = 0.8   # Assets weight
gamma = 0.6  # Liabilities weight

# Threshold: minimum defensible valuation
valuation_threshold = 200  # in millions

# Convert financial data to binary variables (scaled)
scaled_revenue = revenue // 50  
scaled_assets = assets // 50
scaled_liabilities = liabilities // 50

# Number of binary variables
n = 6  
x_r = [f"r_{i}" for i in range(n)]  # Revenue variables
x_a = [f"a_{i}" for i in range(n)]  # Asset variables
x_l = [f"l_{i}" for i in range(n)]  # Liability variables

# Define QUBO model
qubo = {}

# **Fix 1: Reward Selection of Revenue and Assets**
for i in range(n):
    qubo[(x_r[i], x_r[i])] =  alpha * (2**i) * scaled_revenue  # Higher revenue is encouraged
    qubo[(x_a[i], x_a[i])] =  beta * (2**i) * scaled_assets  # Higher assets is encouraged
    qubo[(x_l[i], x_l[i])] = -gamma * (2**i) * scaled_liabilities  # Higher liabilities are discouraged

# **Fix 2: Force at Least One Revenue or Asset Variable to be 1**
large_penalty = 100000  # Big enough to make zero impossible
for var in x_r + x_a:
    qubo[(var, var)] -= large_penalty  # Forces at least one to be selected

# **Fix 3: Enforce Minimum Defensible Valuation**
P = 500  
for i in range(n):
    qubo[(x_r[i], x_r[i])] += P * max(0, (valuation_threshold - scaled_revenue))
    qubo[(x_a[i], x_a[i])] += P * max(0, (valuation_threshold - scaled_assets))
    qubo[(x_l[i], x_l[i])] += P * max(0, (valuation_threshold - scaled_liabilities))

# **DEBUGGING: Print the QUBO Dictionary**
print("🔍 QUBO Matrix Before Solving:")
for key, value in qubo.items():
    print(f"{key}: {value}")

# Solve using simulated annealing
sampler = dimod.SimulatedAnnealingSampler()
response = sampler.sample_qubo(qubo, num_reads=10)

# Interpret results
best_solution = response.first.sample

# **Check if at least one revenue/asset variable was selected**
nonzero_count = sum(best_solution[var] for var in x_r + x_a)
if nonzero_count == 0:
    print("🚨 ERROR: The solver still returned all zeros. This should be impossible!")

best_valuation = sum(
    alpha * (2**i) * scaled_revenue * best_solution[x_r[i]] +
    beta * (2**i) * scaled_assets * best_solution[x_a[i]] -
    gamma * (2**i) * scaled_liabilities * best_solution[x_l[i]]
    for i in range(n)
)

print(f"\n✅ Lowest Defensible Valuation: ${best_valuation:.2f} million")
print("Binary variable states:", best_solution)
