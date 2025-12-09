import sys
import os

# Import all agents
from MAS.Old.agent1_price_forecasting_old import PriceForecastingAgent
from MAS.Old.agent2_power_forecasting_old import PowerForecastingAgent
from MAS.Old.agent3_appliance_profiling_old import ApplianceProfilingAgent
from MAS.Old.agent4_scheduler_old import SchedulingOptimizerAgent


def main():
    print("="*70)
    print("SMART APPLIANCE SCHEDULER - MULTI-AGENT SYSTEM")
    print("="*70)
    
    # =========================================================================
    # STEP 1: LOAD AGENT 1 (PRICE FORECASTING)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: AGENT 1 - PRICE FORECASTING")
    print("="*70)
    
    agent1 = PriceForecastingAgent()
    
    print("Loading Agent 1 from agent1/ folder...")
    agent1.load('agent1/agent1')
    
    print(f"✓ Agent 1 loaded!")
    print(f"  Test prediction: Day 45, Hour 14 = ${agent1.predict(45, 14):.6f}/kWh")
    
    # =========================================================================
    # STEP 2: LOAD AGENT 2 (POWER FORECASTING)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: AGENT 2 - POWER FORECASTING")
    print("="*70)
    
    agent2 = PowerForecastingAgent()
    
    print("Loading Agent 2 from agent2/ folder...")
    agent2.load('agent2/agent2')
    
    print(f"✓ Agent 2 loaded!")
    print(f"  Test prediction: Day 45, Hour 14 = {agent2.predict(45, 14):.3f} kW")
    
    # =========================================================================
    # STEP 3: LOAD AGENT 3 (APPLIANCE PROFILING)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: AGENT 3 - APPLIANCE PROFILING")
    print("="*70)
    
    agent3 = ApplianceProfilingAgent()
    
    print("Loading Agent 3 from agent3/ folder...")
    agent3.load('agent3/agent3')
    
    schedulable = agent3.get_schedulable_appliances()
    print(f"✓ Agent 3 loaded!")
    print(f"  Schedulable appliances: {schedulable}")
    
    # =========================================================================
    # STEP 4: RUN AGENT 4 (SCHEDULING OPTIMIZER)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: AGENT 4 - SCHEDULING OPTIMIZER")
    print("="*70)
    
    # Create scheduler with all 3 agents
    scheduler = SchedulingOptimizerAgent(agent1, agent2, agent3)
    
    # Define scheduling parameters
    test_day = 45
    appliances_to_schedule = ['Washing Machine', 'Dishwasher']
    max_power_limit = 10.0  # kW
    
    print(f"\nScheduling Configuration:")
    print(f"  Day: {test_day}")
    print(f"  Appliances: {appliances_to_schedule}")
    print(f"  Power limit: {max_power_limit} kW")
    
    # Run comparison
    print("\nRunning LP and RL schedulers...")
    results = scheduler.compare(
        day=test_day,
        appliances=appliances_to_schedule,
        max_power=max_power_limit
    )
    
    # Generate outputs
    print("\nGenerating visualizations and reports...")
    scheduler.plot_results('agent4_results.png')
    scheduler.generate_report('agent4_report.txt')
    scheduler.save('agent4')
    
    # =========================================================================
    # STEP 5: SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY - FINAL RESULTS")
    print("="*70)
    
    lp = results['lp']
    rl = results['rl']
    
    print(f"\nDAY {test_day} OPTIMIZATION RESULTS:")
    print(f"\nBaseline Cost (no optimization): ${lp['baseline_cost']:.4f}")
    
    print(f"\n{'─'*70}")
    print("LINEAR PROGRAMMING (Optimal Solution):")
    print(f"{'─'*70}")
    print(f"  Total Cost: ${lp['total_cost']:.4f}")
    print(f"  Savings: ${lp['savings']:.4f} ({lp['savings']/lp['baseline_cost']*100:.1f}%)")
    print(f"  Annual Savings (300 uses): ${lp['savings'] * 300:.2f}")
    print(f"\n  Optimal Schedule:")
    for app, details in sorted(lp['schedule'].items(), key=lambda x: x[1]['start_hour']):
        print(f"    • {app}: Start at {details['start_hour']}:00, Cost ${details['cost']:.4f}")
    
    print(f"\n{'─'*70}")
    print("REINFORCEMENT LEARNING (Learned Solution):")
    print(f"{'─'*70}")
    print(f"  Total Cost: ${rl['total_cost']:.4f}")
    print(f"  Savings: ${rl['savings']:.4f} ({rl['savings']/rl['baseline_cost']*100:.1f}%)")
    print(f"  Annual Savings (300 uses): ${rl['savings'] * 300:.2f}")
    print(f"\n  Learned Schedule:")
    for app, details in sorted(rl['schedule'].items(), key=lambda x: x[1]['start_hour']):
        print(f"    • {app}: Start at {details['start_hour']}:00, Cost ${details['cost']:.4f}")
    
    gap = abs(rl['total_cost'] - lp['total_cost'])
    print(f"\n{'─'*70}")
    print(f"LP vs RL Comparison:")
    print(f"  Cost Difference: ${gap:.4f}")
    print(f"  RL Optimality Gap: {gap/lp['total_cost']*100:.2f}%")
    
    if gap / lp['total_cost'] < 0.05:
        print(f"  ✓ RL solution is within 5% of optimal!")
    else:
        print(f"  ⚠ RL solution is {gap/lp['total_cost']*100:.1f}% from optimal")
    
    # =========================================================================
    # STEP 6: FILES GENERATED
    # =========================================================================
    print("\n" + "="*70)
    print("OUTPUT FILES GENERATED")
    print("="*70)
    
    print("\nAgent 4 Outputs (in Code/ folder):")
    print("  ✓ agent4_results.png - Schedule comparison visualization")
    print("  ✓ agent4_report.txt - Detailed comparison report")
    print("  ✓ agent4_results.json - Results data")
    
    print("\n" + "="*70)
    print("✓ MULTI-AGENT SYSTEM TEST COMPLETE!")
    print("="*70)
    
    print("\nKey Findings:")
    print(f"  • Baseline cost: ${lp['baseline_cost']:.4f}")
    print(f"  • Optimized cost (LP): ${lp['total_cost']:.4f}")
    print(f"  • Savings per use: ${lp['savings']:.4f}")
    print(f"  • Annual savings potential: ${lp['savings'] * 300:.2f}")
    print(f"  • RL performance: {100 - (gap/lp['total_cost']*100):.1f}% of optimal")
    
    print("\nNext steps:")
    print("  1. Review agent4_results.png for visual comparison")
    print("  2. Read agent4_report.txt for detailed analysis")
    print("  3. Test with different days and appliances")
    print("  4. Integrate into Streamlit dashboard")
    
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
        print("\n" + "="*70)
        print("✓ All agents ready! System returned for further use.")
        print("="*70)
        
        # Example: Use the loaded system
        print("\nExample - Test different scenario:")
        print(">>> system['scheduler'].compare(day=100, appliances=['Washing Machine'], max_power=8.0)")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Missing file - {e}")
        print("\nCheck that these folders exist:")
        print("  Code/agent1/ (with agent1_model.pkl, agent1_scalers.pkl, agent1_metrics.json)")
        print("  Code/agent2/ (with agent2_model.pkl, agent2_scalers.pkl, agent2_metrics.json)")
        print("  Code/agent3/ (with agent3_profiles.json, agent3_statistics.json)")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()