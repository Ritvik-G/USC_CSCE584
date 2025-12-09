import sys
import os

# Import all agents
from agent1.agent1_price_forecasting import PriceForecastingAgent
from agent2.agent2_power_forecasting import PowerForecastingAgent
from agent3.agent3_appliance_profiling import ApplianceProfilingAgent
from agent4.agent4_scheduler import SchedulingOptimizerAgent


def main():
    """Run multi-agent system integration test"""
    
    print("="*70)
    print("SMART APPLIANCE SCHEDULER - MULTI-AGENT SYSTEM")
    print("="*70)
    
    # Load agents
    print("\nLoading agents...")
    agent1 = PriceForecastingAgent()
    agent1.load('agent1/agent1')
    print(f"✓ Agent 1 loaded - Test: ${agent1.predict(45, 14):.6f}/kWh")
    
    agent2 = PowerForecastingAgent()
    agent2.load('agent2/agent2')
    print(f"✓ Agent 2 loaded - Test: {agent2.predict(45, 14):.3f} kW")
    
    agent3 = ApplianceProfilingAgent()
    agent3.load('agent3/agent3')
    print(f"✓ Agent 3 loaded - Schedulable: {agent3.get_schedulable_appliances()}")
    
    # Configure and run scheduler
    print("\n" + "="*70)
    print("RUNNING SCHEDULER")
    print("="*70)
    
    scheduler = SchedulingOptimizerAgent(agent1, agent2, agent3)
    
    test_day = 45
    appliances = ['Washing Machine', 'Dishwasher']
    max_power = 10.0
    
    print(f"\nConfig: Day {test_day}, Appliances {appliances}, Power limit {max_power} kW")
    
    results = scheduler.compare(
        day=test_day,
        appliances=appliances,
        max_power=max_power
    )
    
    # Generate outputs
    scheduler.plot_results('agent4_results.png')
    scheduler.generate_report('agent4_report.txt')
    scheduler.save('agent4')
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    lp = results['lp']
    rl = results['rl']
    baseline = lp['baseline_cost']
    
    print(f"\nBaseline: ${baseline:.4f}")
    print(f"\nLP Solution: ${lp['total_cost']:.4f} (saves ${lp['savings']:.4f}, {lp['savings']/baseline*100:.1f}%)")
    for app, details in sorted(lp['schedule'].items(), key=lambda x: x[1]['start_hour']):
        print(f"  • {app}: {details['start_hour']}:00, ${details['cost']:.4f}")
    
    print(f"\nRL Solution: ${rl['total_cost']:.4f} (saves ${rl['savings']:.4f}, {rl['savings']/baseline*100:.1f}%)")
    for app, details in sorted(rl['schedule'].items(), key=lambda x: x[1]['start_hour']):
        print(f"  • {app}: {details['start_hour']}:00, ${details['cost']:.4f}")
    
    gap = abs(rl['total_cost'] - lp['total_cost'])
    gap_pct = gap/lp['total_cost']*100
    print(f"\nRL Optimality Gap: {gap_pct:.2f}% {'✓' if gap_pct < 5 else '⚠'}")
    print(f"Annual Savings Potential: ${lp['savings'] * 300:.2f}")
    
    print("\n" + "="*70)
    print("✓ TEST COMPLETE - Files: agent4_results.png, agent4_report.txt")
    print("="*70)
    
    return {
        'agent1': agent1,
        'agent2': agent2,
        'agent3': agent3,
        'scheduler': scheduler,
        'results': results
    }


if __name__ == "__main__":
    try:
        system = main()
        print("\n✓ System ready for further use")
        print("Example: system['scheduler'].compare(day=100, appliances=['Washing Machine'], max_power=8.0)")
        
    except FileNotFoundError as e:
        print(f"\n✗ Missing file: {e}")
        print("Required folders: agent1/, agent2/, agent3/ with model files")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()