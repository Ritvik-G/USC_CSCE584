# Multi-Agent System (MAS) - Smart Appliance Scheduler

A multi-agent system for optimizing household appliance scheduling to minimize electricity costs. The system uses machine learning and optimization algorithms to predict prices, power consumption, and schedule appliances intelligently.

## System Overview

The MAS consists of four specialized agents that work together:

### **Agent 1: Price Forecasting**
- **Purpose**: Predicts electricity prices ($/kWh) for each hour of the day
- **Model**: Feed-forward neural network (MLPRegressor)
- **Input**: Day number and hour
- **Output**: Forecasted price for that hour
- **Location**: `agent1/agent1_price_forecasting.py`

### **Agent 2: Power Forecasting**
- **Purpose**: Predicts household baseline power consumption (kW) for each hour
- **Model**: Feed-forward neural network (MLPRegressor)
- **Input**: Day number and hour
- **Output**: Forecasted power consumption without scheduled appliances
- **Location**: `agent2/agent2_power_forecasting.py`

### **Agent 3: Appliance Profiling**
- **Purpose**: Analyzes and profiles appliance energy usage patterns
- **Method**: Statistical analysis of historical device consumption data
- **Output**: Energy profiles (duration, power, peak hours) for each appliance type
- **Categorizes appliances**: Essential (Fridge, Heater), Necessary (Oven, TV), Expendable (Washer, Dishwasher)
- **Location**: `agent3/agent3_appliance_profiling.py`

### **Agent 4: Scheduling Optimizer**
- **Purpose**: Optimizes when to schedule appliances to minimize cost
- **Methods**:
  - **Linear Programming (LP)**: Guaranteed optimal solution
  - **Reinforcement Learning (RL) Greedy**: Heuristic-based greedy approach
- **Constraints**: Power limit, appliance duration, one-time execution per day
- **Output**: Optimized schedule with cost savings vs. baseline
- **Location**: `agent4/agent4_scheduler.py`

## Quick Start

### Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn pulp scipy
```

### Run Individual Agents

**Agent 1 - Price Forecasting:**
```bash
cd agent1
python agent1_price_forecasting.py
```

**Agent 2 - Power Forecasting:**
```bash
cd agent2
python agent2_power_forecasting.py
```

**Agent 3 - Appliance Profiling:**
```bash
cd agent3
python agent3_appliance_profiling.py
```

**Agent 4 - Scheduling Optimizer:**
```bash
cd agent4
python agent4_scheduler.py
```

### Run Integration Test

Tests all agents working together in a unified system with an example:

```bash
python integration_test.py
```

**What it does:**
- Loads trained models from all 4 agents
- Tests each agent's predictions with sample inputs
- Runs the scheduler on test appliances (Washing Machine, Dishwasher)
- Compares LP vs RL optimization methods
- Displays results and savings

### Run Robust Evaluation

Comprehensive evaluation across diverse scenarios with statistical analysis:

```bash
cd robust_evals
python robust_evaluation.py
```

Or from parent directory:
```bash
python robust_evals/robust_evaluation.py
```

**What it does:**
- Tests across multiple days and appliance combinations
- Evaluates both LP and RL scheduling methods
- Generates statistics on cost savings
- Produces visualizations of schedules and price trends
- Compares performance metrics across scenarios
- Creates detailed reports and plots


## Key Metrics

- **Cost Savings**: Percentage reduction in electricity cost vs. baseline
- **Schedulability**: Which appliances can be shifted without user impact
- **Constraint Satisfaction**: Power limit and timing constraints met
- **Optimality Gap**: LP vs RL performance difference

## Notes

- All agents use trained models saved as pickle/JSON files
- Prices and power are normalized to 0-24 hour format
- Appliances are scheduled for 2-hour duration slots
- LP method guarantees optimal solution but slower
- RL method is fast heuristic suitable for real-time systems
