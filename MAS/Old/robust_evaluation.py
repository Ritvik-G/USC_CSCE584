"""
Robust Evaluation System for Multi-Agent Scheduler
Comprehensive testing across diverse scenarios with statistical analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MAS.agent1.agent1_price_forecasting import PriceForecastingAgent
from MAS.agent2.agent2_power_forecasting import PowerForecastingAgent
from MAS.agent3.agent3_appliance_profiling import ApplianceProfilingAgent
from MAS.agent4.agent4_scheduler import SchedulingOptimizerAgent
import json
import time
from scipy import stats


class RobustEvaluator:
    
    def __init__(self, agent1, agent2, agent3):
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent3 = agent3
        self.scheduler = SchedulingOptimizerAgent(agent1, agent2, agent3)
        self.results = []
        
    def analyze_price_volatility(self, day):
        """Calculate price volatility for a day"""
        prices = list(self.agent1.predict_day(day).values())
        return {
            'mean': np.mean(prices),
            'std': np.std(prices),
            'cv': np.std(prices) / np.mean(prices)
        }
    
    def analyze_power_demand(self, day):
        """Calculate power demand characteristics for a day"""
        power = self.agent2.predict_day(day)
        power_values = list(power.values())
        return {
            'mean': np.mean(power_values),
            'peak': np.max(power_values)
        }
    
    def select_test_scenarios(self):
        """Select diverse test scenarios from data"""
        
        print("\nSelecting test scenarios...")
        scenarios = []
        available_days = range(1, 184)
        
        # Analyze all days
        price_volatilities = [(day, self.analyze_price_volatility(day)['cv']) 
                             for day in available_days]
        power_demands = [(day, self.analyze_power_demand(day)['mean']) 
                        for day in available_days]
        
        price_volatilities.sort(key=lambda x: x[1])
        power_demands.sort(key=lambda x: x[1])
        
        # Select representative scenarios
        test_configs = [
            ([0, 5, 10], 'Low Price Volatility', price_volatilities),
            ([-1, -5, -10], 'High Price Volatility', price_volatilities),
            ([0, 5, 10], 'Low Power Demand', power_demands),
            ([-1, -5, -10], 'High Power Demand', power_demands),
        ]
        
        for indices, scenario_type, data_list in test_configs:
            for idx in indices:
                scenarios.append({
                    'day': data_list[idx][0],
                    'type': scenario_type,
                    'appliances': ['Washing Machine', 'Dishwasher']
                })
        
        # Additional scenarios
        for day in [30, 60, 90]:
            scenarios.append({
                'day': day,
                'type': '3 Appliances',
                'appliances': ['Washing Machine', 'Dishwasher', 'Oven']
            })
        
        for day in [7, 14, 21, 28]:
            scenarios.append({
                'day': day,
                'type': 'Weekend Pattern',
                'appliances': ['Washing Machine', 'Dishwasher']
            })
        
        for day in [45, 90, 135, 180]:
            scenarios.append({
                'day': day,
                'type': 'Mid-Range',
                'appliances': ['Washing Machine', 'Dishwasher']
            })
        
        print(f"✓ Selected {len(scenarios)} diverse scenarios")
        return scenarios
    
    def run_evaluation(self, max_power=10.0):
        """Run comprehensive evaluation"""
        
        print("\n" + "="*70)
        print("RUNNING EVALUATION")
        print("="*70)
        
        scenarios = self.select_test_scenarios()
        
        for i, scenario in enumerate(scenarios):
            print(f"[{i+1}/{len(scenarios)}] Day {scenario['day']}, {scenario['type']}")
            
            try:
                start_time = time.time()
                results = self.scheduler.compare(
                    day=scenario['day'],
                    appliances=scenario['appliances'],
                    max_power=max_power
                )
                exec_time = time.time() - start_time
                
                lp = results['lp']
                rl = results['rl']
                price_info = self.analyze_price_volatility(scenario['day'])
                power_info = self.analyze_power_demand(scenario['day'])
                
                self.results.append({
                    'scenario_id': i + 1,
                    'day': scenario['day'],
                    'scenario_type': scenario['type'],
                    'num_appliances': len(scenario['appliances']),
                    'appliances': scenario['appliances'],
                    'baseline_cost': lp['baseline_cost'],
                    'lp_cost': lp['total_cost'],
                    'rl_cost': rl['total_cost'],
                    'lp_savings': lp['savings'],
                    'rl_savings': rl['savings'],
                    'lp_savings_percent': (lp['savings'] / lp['baseline_cost'] * 100),
                    'rl_savings_percent': (rl['savings'] / rl['baseline_cost'] * 100),
                    'lp_rl_gap': abs(lp['total_cost'] - rl['total_cost']),
                    'lp_rl_gap_percent': (abs(lp['total_cost'] - rl['total_cost']) / lp['total_cost'] * 100),
                    'execution_time': exec_time,
                    'price_volatility': price_info['cv'],
                    'peak_demand': power_info['peak']
                })
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        print(f"\n✓ Completed {len(self.results)}/{len(scenarios)} scenarios")
        return self.results
    
    def calculate_statistics(self):
        """Calculate summary statistics"""
        df = pd.DataFrame(self.results)
        
        return {
            'total_scenarios': len(df),
            'lp_mean_savings': df['lp_savings'].mean(),
            'lp_std_savings': df['lp_savings'].std(),
            'lp_mean_savings_percent': df['lp_savings_percent'].mean(),
            'lp_std_savings_percent': df['lp_savings_percent'].std(),
            'rl_mean_savings': df['rl_savings'].mean(),
            'rl_std_savings': df['rl_savings'].std(),
            'rl_mean_savings_percent': df['rl_savings_percent'].mean(),
            'rl_std_savings_percent': df['rl_savings_percent'].std(),
            'mean_gap': df['lp_rl_gap'].mean(),
            'mean_gap_percent': df['lp_rl_gap_percent'].mean(),
            'mean_exec_time': df['execution_time'].mean(),
            'lp_rl_ttest': stats.ttest_rel(df['lp_savings'], df['rl_savings']),
            'savings_normality': stats.shapiro(df['lp_savings'])
        }
    
    def plot_comprehensive_results(self, save_path='robust_evaluation_results.png'):
        """Generate comprehensive visualization"""
        df = pd.DataFrame(self.results)
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # 1. Savings distribution
        axes[0, 0].hist(df['lp_savings_percent'], bins=20, alpha=0.7, label='LP')
        axes[0, 0].hist(df['rl_savings_percent'], bins=20, alpha=0.7, label='RL')
        axes[0, 0].set_xlabel('Savings (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Savings Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Savings by scenario type
        scenario_types = df['scenario_type'].unique()
        lp_means = [df[df['scenario_type']==t]['lp_savings_percent'].mean() for t in scenario_types]
        rl_means = [df[df['scenario_type']==t]['rl_savings_percent'].mean() for t in scenario_types]
        x = np.arange(len(scenario_types))
        width = 0.35
        axes[0, 1].bar(x - width/2, lp_means, width, label='LP', alpha=0.8)
        axes[0, 1].bar(x + width/2, rl_means, width, label='RL', alpha=0.8)
        axes[0, 1].set_xlabel('Scenario Type')
        axes[0, 1].set_ylabel('Mean Savings (%)')
        axes[0, 1].set_title('Performance by Scenario')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(scenario_types, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. LP vs RL scatter
        axes[0, 2].scatter(df['lp_savings_percent'], df['rl_savings_percent'], alpha=0.6)
        lim = max(df['lp_savings_percent'].max(), df['rl_savings_percent'].max())
        axes[0, 2].plot([0, lim], [0, lim], 'k--', lw=2)
        axes[0, 2].set_xlabel('LP Savings (%)')
        axes[0, 2].set_ylabel('RL Savings (%)')
        axes[0, 2].set_title('LP vs RL Comparison')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Optimality gap
        axes[1, 0].hist(df['lp_rl_gap_percent'], bins=20, color='coral', alpha=0.7)
        axes[1, 0].set_xlabel('Optimality Gap (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('RL Optimality Gap')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Savings vs price volatility
        axes[1, 1].scatter(df['price_volatility'], df['lp_savings_percent'], alpha=0.6)
        axes[1, 1].set_xlabel('Price Volatility (CV)')
        axes[1, 1].set_ylabel('LP Savings (%)')
        axes[1, 1].set_title('Savings vs Volatility')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Savings vs peak demand
        axes[1, 2].scatter(df['peak_demand'], df['lp_savings_percent'], alpha=0.6, color='green')
        axes[1, 2].set_xlabel('Peak Demand (kW)')
        axes[1, 2].set_ylabel('LP Savings (%)')
        axes[1, 2].set_title('Savings vs Peak Demand')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Box plot by number of appliances
        df.boxplot(column='lp_savings_percent', by='num_appliances', ax=axes[2, 0])
        axes[2, 0].set_xlabel('Number of Appliances')
        axes[2, 0].set_ylabel('LP Savings (%)')
        axes[2, 0].set_title('Savings by Appliance Count')
        axes[2, 0].get_figure().suptitle('')
        
        # 8. Execution time
        axes[2, 1].hist(df['execution_time'], bins=20, color='purple', alpha=0.7)
        axes[2, 1].set_xlabel('Execution Time (s)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title('Computational Performance')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Summary statistics
        stats_data = self.calculate_statistics()
        axes[2, 2].axis('off')
        summary_text = f"""
SUMMARY STATISTICS

Total Scenarios: {stats_data['total_scenarios']}

LP Performance:
  Mean: {stats_data['lp_mean_savings_percent']:.2f}%
  Std: {stats_data['lp_std_savings_percent']:.2f}%

RL Performance:
  Mean: {stats_data['rl_mean_savings_percent']:.2f}%
  Gap: {stats_data['mean_gap_percent']:.2f}%

Annual Savings:
  ${stats_data['lp_mean_savings'] * 300:.2f}/year
        """
        axes[2, 2].text(0.1, 0.5, summary_text, fontsize=10, 
                       verticalalignment='center', family='monospace')
        
        plt.suptitle('Robust Evaluation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✓ Plots saved: {save_path}")
    
    def generate_report(self, save_path='robust_evaluation_report.txt'):
        """Generate text report"""
        df = pd.DataFrame(self.results)
        stats_data = self.calculate_statistics()
        
        report = f"""
{'='*70}
ROBUST EVALUATION REPORT
{'='*70}

Total Scenarios Tested: {stats_data['total_scenarios']}

{'='*70}
LINEAR PROGRAMMING RESULTS:
{'='*70}

Savings Performance:
  - Mean Savings: ${stats_data['lp_mean_savings']:.4f} ({stats_data['lp_mean_savings_percent']:.2f}%)
  - Std Deviation: ${stats_data['lp_std_savings']:.4f} ({stats_data['lp_std_savings_percent']:.2f}%)
  - Annual Savings: ${stats_data['lp_mean_savings'] * 300:.2f}

{'='*70}
REINFORCEMENT LEARNING RESULTS:
{'='*70}

Savings Performance:
  - Mean Savings: ${stats_data['rl_mean_savings']:.4f} ({stats_data['rl_mean_savings_percent']:.2f}%)
  - Optimality Gap: {stats_data['mean_gap_percent']:.2f}%
  - RL Achieves: {100 - stats_data['mean_gap_percent']:.1f}% of LP optimal

{'='*70}
PERFORMANCE BY SCENARIO TYPE:
{'='*70}

"""
        
        for scenario_type in df['scenario_type'].unique():
            subset = df[df['scenario_type'] == scenario_type]
            report += f"""{scenario_type}:
  - Tests: {len(subset)}
  - LP Mean: {subset['lp_savings_percent'].mean():.2f}% (±{subset['lp_savings_percent'].std():.2f}%)
  - RL Mean: {subset['rl_savings_percent'].mean():.2f}% (±{subset['rl_savings_percent'].std():.2f}%)

"""
        
        report += f"""
{'='*70}
STATISTICAL TESTS:
{'='*70}

Paired t-test (LP vs RL):
  - t-statistic: {stats_data['lp_rl_ttest'].statistic:.4f}
  - p-value: {stats_data['lp_rl_ttest'].pvalue:.6f}
  - Significant: {'Yes' if stats_data['lp_rl_ttest'].pvalue < 0.05 else 'No'}

Normality Test (Shapiro-Wilk):
  - W-statistic: {stats_data['savings_normality'].statistic:.4f}
  - p-value: {stats_data['savings_normality'].pvalue:.4f}

{'='*70}
KEY FINDINGS:
{'='*70}

1. LP consistently delivers {stats_data['lp_mean_savings_percent']:.1f}% average savings
2. RL achieves {100 - stats_data['mean_gap_percent']:.1f}% of LP optimal performance
3. Annual savings potential: ${stats_data['lp_mean_savings'] * 300:.2f}/year
4. Stable performance across diverse scenarios (std {stats_data['lp_std_savings_percent']:.1f}%)
"""
        
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"✓ Report saved: {save_path}")
    
    def save_results(self, save_path='robust_evaluation_data.json'):
        """Save all results to JSON"""
        results_clean = [{
            'scenario_id': r['scenario_id'],
            'day': r['day'],
            'scenario_type': r['scenario_type'],
            'num_appliances': r['num_appliances'],
            'baseline_cost': r['baseline_cost'],
            'lp_cost': r['lp_cost'],
            'rl_cost': r['rl_cost'],
            'lp_savings': r['lp_savings'],
            'rl_savings': r['rl_savings'],
            'lp_savings_percent': r['lp_savings_percent'],
            'rl_savings_percent': r['rl_savings_percent'],
            'execution_time': r['execution_time']
        } for r in self.results]
        
        with open(save_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        print(f"✓ Data saved: {save_path}")
    
    def export_latex_table(self, save_path='robust_evaluation_table.tex'):
        """Export results as LaTeX table"""
        stats_data = self.calculate_statistics()
        
        latex = r"""\begin{table}[h]
\centering
\caption{Robust Evaluation Results}
\label{tab:robust_eval}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{LP} & \textbf{RL} \\
\hline
Mean Savings (\%) & """ + f"{stats_data['lp_mean_savings_percent']:.2f}" + r""" & """ + f"{stats_data['rl_mean_savings_percent']:.2f}" + r""" \\
Std Dev (\%) & """ + f"{stats_data['lp_std_savings_percent']:.2f}" + r""" & """ + f"{stats_data['rl_std_savings_percent']:.2f}" + r""" \\
Optimality Gap (\%) & 0.00 & """ + f"{stats_data['mean_gap_percent']:.2f}" + r""" \\
\hline
\end{tabular}
\end{table}
"""
        
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"✓ LaTeX table saved: {save_path}")


def main():
    print("="*70)
    print("ROBUST EVALUATION SYSTEM")
    print("="*70)
    
    # Load agents
    print("\nLoading agents...")
    agent1 = PriceForecastingAgent()
    agent1.load('agent1/agent1')
    
    agent2 = PowerForecastingAgent()
    agent2.load('agent2/agent2')
    
    agent3 = ApplianceProfilingAgent()
    agent3.load('agent3/agent3')
    print("✓ All agents loaded")
    
    # Run evaluation
    evaluator = RobustEvaluator(agent1, agent2, agent3)
    evaluator.run_evaluation(max_power=10.0)
    
    # Generate outputs
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)
    
    evaluator.plot_comprehensive_results('robust_evaluation_results.png')
    evaluator.generate_report('robust_evaluation_report.txt')
    evaluator.save_results('robust_evaluation_data.json')
    evaluator.export_latex_table('robust_evaluation_table.tex')
    
    # Summary
    stats_data = evaluator.calculate_statistics()
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\n✓ Tested {stats_data['total_scenarios']} scenarios")
    print(f"✓ LP Savings: {stats_data['lp_mean_savings_percent']:.2f}% ± {stats_data['lp_std_savings_percent']:.2f}%")
    print(f"✓ RL Savings: {stats_data['rl_mean_savings_percent']:.2f}% ± {stats_data['rl_std_savings_percent']:.2f}%")
    print(f"✓ Annual Potential: ${stats_data['lp_mean_savings'] * 300:.2f}/year")


if __name__ == "__main__":
    main()