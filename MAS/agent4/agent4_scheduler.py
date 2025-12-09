"""
Agent 4: Appliance Scheduling Optimizer
Optimizes appliance schedules using Linear Programming and Reinforcement Learning
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
import pickle

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_palette("husl")


class SchedulingOptimizerAgent:
    
    def __init__(self, agent1, agent2, agent3):
        """Initialize with trained agents 1, 2, 3"""
        self.agent1 = agent1  # Price forecasting
        self.agent2 = agent2  # Power forecasting
        self.agent3 = agent3  # Appliance profiling
        self.results = {}
        
    def get_baseline_schedule(self, appliances, profiles):
        """Get original schedule based on peak usage hours"""
        baseline = {}
        for app in appliances:
            peak_hour = profiles[app]['peak_hours'][0]
            baseline[app] = {
                'start_hour': peak_hour,
                'energy_kwh': profiles[app]['avg_energy_kwh'],
                'cost': 0
            }
        return baseline
        
    def optimize_lp(self, day, appliances, max_power=10.0):
        """Linear Programming - guaranteed optimal solution"""
        # Get forecasts
        prices = self.agent1.predict_day(day)
        power = self.agent2.predict_day(day)
        profiles = {app: self.agent3.get_profile(app) for app in appliances}
        
        # Get baseline
        baseline_schedule = self.get_baseline_schedule(appliances, profiles)
        baseline_cost = sum([profiles[app]['avg_energy_kwh'] * prices[baseline_schedule[app]['start_hour']]
                            for app in appliances])
        
        # LP formulation
        prob = LpProblem("Schedule", LpMinimize)
        
        # Decision variables
        x = {(app, h): LpVariable(f"x_{app}_{h}", cat='Binary') 
             for app in appliances for h in range(1, 25)}
        
        # Objective: minimize cost
        prob += lpSum([x[app, h] * profiles[app]['avg_energy_kwh'] * prices[h]
                      for app in appliances for h in range(1, 25)])
        
        # Constraint: each appliance runs once
        for app in appliances:
            prob += lpSum([x[app, h] for h in range(1, 25)]) == 1
        
        # Constraint: power limit
        for h in range(1, 25):
            appliance_power = lpSum([
                x[app, start] * profiles[app]['estimated_power_kw']
                for app in appliances
                for start in range(1, 25)
                if start <= h < start + profiles[app]['estimated_duration_hours']
            ])
            prob += power[h] + appliance_power <= max_power
        
        # Solve
        prob.solve()
        
        # Extract schedule
        schedule = {}
        for app in appliances:
            for h in range(1, 25):
                if x[app, h].varValue == 1:
                    schedule[app] = {
                        'start_hour': h,
                        'energy_kwh': profiles[app]['avg_energy_kwh'],
                        'cost': profiles[app]['avg_energy_kwh'] * prices[h]
                    }
        
        total_cost = value(prob.objective)
        
        return {
            'method': 'LP',
            'schedule': schedule,
            'baseline_schedule': baseline_schedule,
            'total_cost': total_cost,
            'baseline_cost': baseline_cost,
            'savings': baseline_cost - total_cost
        }
    
    def optimize_rl(self, day, appliances, max_power=10.0):
        """RL Greedy - schedules at cheapest valid hours"""
        prices = self.agent1.predict_day(day)
        power = self.agent2.predict_day(day)
        profiles = {app: self.agent3.get_profile(app) for app in appliances}
        
        baseline_schedule = self.get_baseline_schedule(appliances, profiles)
        baseline_cost = sum([profiles[app]['avg_energy_kwh'] * prices[baseline_schedule[app]['start_hour']]
                            for app in appliances])
        
        # Sort hours by price
        sorted_hours = sorted(range(1, 25), key=lambda h: prices[h])
        
        schedule = {}
        scheduled_hours = set()
        
        # Greedy assignment: cheapest valid hour for each appliance
        for app in appliances:
            profile = profiles[app]
            duration = int(profile['estimated_duration_hours'])
            
            best_hour = None
            best_cost = float('inf')
            
            for hour in sorted_hours:
                if hour in scheduled_hours:
                    continue
                if hour + duration > 25:
                    continue
                if power[hour] + profile['estimated_power_kw'] > max_power:
                    continue
                
                cost = profile['avg_energy_kwh'] * prices[hour]
                if cost < best_cost:
                    best_cost = cost
                    best_hour = hour
            
            if best_hour:
                schedule[app] = {
                    'start_hour': best_hour,
                    'energy_kwh': profile['avg_energy_kwh'],
                    'cost': best_cost
                }
                for h in range(best_hour, min(best_hour + duration, 25)):
                    scheduled_hours.add(h)
        
        total_cost = sum([schedule[app]['cost'] for app in schedule])
        
        return {
            'method': 'RL (Greedy)',
            'schedule': schedule,
            'baseline_schedule': baseline_schedule,
            'total_cost': total_cost,
            'baseline_cost': baseline_cost,
            'savings': baseline_cost - total_cost
        }
    
    def compare(self, day, appliances, max_power=10.0):
        """Run both methods and compare"""
        lp_result = self.optimize_lp(day, appliances, max_power)
        rl_result = self.optimize_rl(day, appliances, max_power)
        
        self.results = {
            'day': day,
            'appliances': appliances,
            'lp': lp_result,
            'rl': rl_result,
            'prices': self.agent1.predict_day(day)
        }
        
        return self.results
    
    def plot_results(self, save_path='agent4_results.png'):
        """Visualize original vs optimized schedules"""
        if not self.results:
            print("No results. Run compare() first.")
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        lp = self.results['lp']
        rl = self.results['rl']
        prices = self.results['prices']
        baseline = lp['baseline_schedule']
        
        colors = {'Washing Machine': '#2E86AB', 'Dishwasher': '#A23B72', 
                 'Oven': '#F18F01', 'Microwave': '#C73E1D'}
        
        # Original schedule
        plt.subplot(3, 2, 1)
        y = 0
        for app, details in baseline.items():
            color = colors.get(app, '#888888')
            plt.barh(y, 2, left=details['start_hour']-1, height=0.8, 
                    color=color, alpha=0.6, edgecolor='black', linewidth=2)
            plt.text(details['start_hour'], y, app, ha='left', va='center', fontweight='bold', fontsize=9)
            y += 1
        plt.xlim(0, 24)
        plt.xlabel('Hour', fontweight='bold')
        plt.title('Original Schedule (Peak Hours)', fontweight='bold')
        plt.yticks([])
        plt.grid(True, alpha=0.3, axis='x')
        plt.xticks(range(0, 25, 4))
        
        # LP schedule
        plt.subplot(3, 2, 2)
        y = 0
        for app, details in lp['schedule'].items():
            color = colors.get(app, '#888888')
            plt.barh(y, 2, left=details['start_hour']-1, height=0.8, 
                    color=color, alpha=0.9, edgecolor='black', linewidth=2)
            plt.text(details['start_hour'], y, app, ha='left', va='center', fontweight='bold', fontsize=9)
            y += 1
        plt.xlim(0, 24)
        plt.xlabel('Hour', fontweight='bold')
        plt.title('LP Optimized Schedule', fontweight='bold')
        plt.yticks([])
        plt.grid(True, alpha=0.3, axis='x')
        plt.xticks(range(0, 25, 4))
        
        # RL schedule
        plt.subplot(3, 2, 3)
        y = 0
        for app, details in rl['schedule'].items():
            color = colors.get(app, '#888888')
            plt.barh(y, 2, left=details['start_hour']-1, height=0.8, 
                    color=color, alpha=0.9, edgecolor='black', linewidth=2)
            plt.text(details['start_hour'], y, app, ha='left', va='center', fontweight='bold', fontsize=9)
            y += 1
        plt.xlim(0, 24)
        plt.xlabel('Hour', fontweight='bold')
        plt.title('RL Greedy Schedule', fontweight='bold')
        plt.yticks([])
        plt.grid(True, alpha=0.3, axis='x')
        plt.xticks(range(0, 25, 4))
        
        # Cost comparison
        plt.subplot(3, 2, 4)
        methods = ['Original', 'LP\nOptimized', 'RL\nGreedy']
        costs = [lp['baseline_cost'], lp['total_cost'], rl['total_cost']]
        colors_bar = ['#888888', '#2E86AB', '#C73E1D']
        bars = plt.bar(methods, costs, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
        
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${cost:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylabel('Cost ($)', fontweight='bold')
        plt.title('Cost Comparison', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Prices with schedules
        plt.subplot(3, 2, 5)
        hours = list(range(1, 25))
        price_values = [prices[h] for h in hours]
        plt.plot(hours, price_values, linewidth=3, marker='o', color='#F18F01', label='Price', markersize=6)
        
        for app, details in baseline.items():
            plt.axvline(details['start_hour'], color='gray', linestyle=':', linewidth=2, alpha=0.7, 
                       label='Original' if app == list(baseline.keys())[0] else '')
        
        for app, details in lp['schedule'].items():
            plt.axvline(details['start_hour'], color='blue', linestyle='--', linewidth=2, alpha=0.7, 
                       label='LP' if app == list(lp['schedule'].keys())[0] else '')
        
        plt.xlabel('Hour', fontweight='bold')
        plt.ylabel('Price ($/kWh)', fontweight='bold')
        plt.title('Prices & Scheduling Times', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.xticks(range(0, 25, 4))
        
        # Savings summary
        plt.subplot(3, 2, 6)
        plt.axis('off')
        
        savings_data = [
            ['Metric', 'LP', 'RL'],
            ['Cost', f"${lp['total_cost']:.4f}", f"${rl['total_cost']:.4f}"],
            ['Savings', f"${lp['savings']:.4f}", f"${rl['savings']:.4f}"],
            ['Savings %', f"{lp['savings']/lp['baseline_cost']*100:.1f}%", 
             f"{rl['savings']/rl['baseline_cost']*100:.1f}%"],
            ['Baseline', f"${lp['baseline_cost']:.4f}", f"${rl['baseline_cost']:.4f}"]
        ]
        
        table = plt.table(cellText=savings_data, cellLoc='center', loc='center', colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        for i in range(3):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Savings Summary', fontweight='bold', fontsize=12, pad=20)
        
        plt.suptitle(f'Agent 4: Scheduling Results (Day {self.results["day"]})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def generate_report(self, save_path='agent4_report.txt'):
        """Generate text report"""
        if not self.results:
            print("No results. Run compare() first.")
            return
        
        lp = self.results['lp']
        rl = self.results['rl']
        baseline = lp['baseline_schedule']
        
        report = f"""
{'='*70}
AGENT 4: SCHEDULING OPTIMIZER
{'='*70}

Day: {self.results['day']}
Appliances: {', '.join(self.results['appliances'])}

ORIGINAL SCHEDULE:
"""
        for app, details in baseline.items():
            report += f"  {app}: Hour {details['start_hour']}\n"
        
        report += f"\nBaseline Cost: ${lp['baseline_cost']:.4f}\n"
        report += f"\n{'─'*70}\n"
        report += f"LINEAR PROGRAMMING:\n"
        report += f"  Cost: ${lp['total_cost']:.4f}\n"
        report += f"  Savings: ${lp['savings']:.4f} ({lp['savings']/lp['baseline_cost']*100:.1f}%)\n\n"
        
        for app, details in lp['schedule'].items():
            report += f"  {app}: Hour {details['start_hour']}, Cost ${details['cost']:.4f}\n"
        
        report += f"\n{'─'*70}\n"
        report += f"RL GREEDY:\n"
        report += f"  Cost: ${rl['total_cost']:.4f}\n"
        report += f"  Savings: ${rl['savings']:.4f} ({rl['savings']/rl['baseline_cost']*100:.1f}%)\n\n"
        
        for app, details in rl['schedule'].items():
            report += f"  {app}: Hour {details['start_hour']}, Cost ${details['cost']:.4f}\n"
        
        diff = abs(rl['total_cost'] - lp['total_cost'])
        report += f"\n{'─'*70}\n"
        report += f"LP vs RL Gap: ${diff:.4f}\n"
        report += f"Annual Savings (300 uses): ${lp['savings'] * 300:.2f}\n"
        report += f"\n{'='*70}\n"
        
        with open(save_path, 'w') as f:
            f.write(report)
        print(report)
    
    def save(self, prefix='agent4'):
        """Save results"""
        if not self.results:
            print("No results to save.")
            return
            
        with open(f'{prefix}_results.json', 'w') as f:
            results_clean = {
                'day': self.results['day'],
                'appliances': self.results['appliances'],
                'baseline_cost': self.results['lp']['baseline_cost'],
                'lp_cost': self.results['lp']['total_cost'],
                'rl_cost': self.results['rl']['total_cost'],
                'lp_savings': self.results['lp']['savings'],
                'rl_savings': self.results['rl']['savings']
            }
            json.dump(results_clean, f, indent=2)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from agent1.agent1_price_forecasting import PriceForecastingAgent
    from agent2.agent2_power_forecasting import PowerForecastingAgent
    from agent3.agent3_appliance_profiling import ApplianceProfilingAgent
    
    # Load agents
    agent1 = PriceForecastingAgent()
    agent1.load('../agent1/agent1')
    
    agent2 = PowerForecastingAgent()
    agent2.load('../agent2/agent2')
    
    agent3 = ApplianceProfilingAgent()
    agent3.load('../agent3/agent3')

    # Create scheduler
    scheduler = SchedulingOptimizerAgent(agent1, agent2, agent3)
    
    # Run optimization
    results = scheduler.compare(day=45, appliances=['Washing Machine', 'Dishwasher'], max_power=10.0) # Add more devices here from the list if needed
    
    # Generate outputs
    scheduler.plot_results('agent4_results.png')
    scheduler.generate_report('agent4_report.txt')
    scheduler.save('agent4')