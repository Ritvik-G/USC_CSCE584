"""
Agent 3: Appliance Profiling
Statistical analysis of appliance usage patterns from device consumption data
No machine learning - pure data aggregation and profiling
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set publication-quality plot style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
sns.set_palette("husl")


class ApplianceProfilingAgent:
    
    # Appliance categories
    ESSENTIAL_APPLIANCES = {"Fridge", "Heater", "Air Conditioning", "Lights"}
    NECESSARY_APPLIANCES = {"Oven", "Microwave", "TV", "Computer"}
    EXPENDABLE_APPLIANCES = {"Washing Machine", "Dishwasher"}
    
    def __init__(self):
        self.df = None
        self.profiles = {}
        self.statistics = {}
        
    def load_data(self, filepath):
        """Load device consumption data"""
        print("Loading device consumption data...")
        self.df = pd.read_csv(filepath)
        
        print(f"Loaded {len(self.df)} appliance usage records")
        print(f"Columns: {self.df.columns.tolist()}")
        print(f"Appliance types: {sorted(self.df['Appliance Type'].unique())}")
        print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        # Parse time and extract hour
        self.df['Hour'] = pd.to_datetime(self.df['Time'], format='%H:%M', errors='coerce').dt.hour + 1
        
        # Parse date
        self.df['DateTime'] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d', errors='coerce')
        self.df['DayOfWeek'] = self.df['DateTime'].dt.dayofweek + 1  # 1=Monday, 7=Sunday
        
        return self.df
    
    def categorize_appliance(self, appliance_type):
        """Determine category of appliance"""
        if appliance_type in self.ESSENTIAL_APPLIANCES:
            return "ESSENTIAL"
        elif appliance_type in self.NECESSARY_APPLIANCES:
            return "NECESSARY"
        elif appliance_type in self.EXPENDABLE_APPLIANCES:
            return "EXPENDABLE"
        else:
            return "OTHER"
    
    def build_profiles(self):
        """Build statistical profiles for each appliance type"""
        print("\nBuilding appliance profiles...")
        
        appliance_types = self.df['Appliance Type'].unique()
        
        for appliance in appliance_types:
            print(f"  Profiling: {appliance}")
            
            # Filter data for this appliance
            app_df = self.df[self.df['Appliance Type'] == appliance]
            
            # Basic statistics
            energy_values = app_df['Energy Consumption (kWh)'].values
            
            # Usage by hour
            hourly_usage = app_df.groupby('Hour').size().to_dict()
            total_occurrences = len(app_df)
            hourly_frequency = {h: hourly_usage.get(h, 0) / total_occurrences 
                               for h in range(1, 25)}
            
            # Peak usage hours (top 5)
            peak_hours = sorted(hourly_frequency.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
            peak_hours_list = [h for h, _ in peak_hours]
            
            # Typical usage windows (hours with >5% of occurrences)
            typical_hours = [h for h, freq in hourly_frequency.items() if freq > 0.05]
            
            # Temperature correlation (for climate-dependent appliances)
            temp_corr = None
            if 'Outdoor Temperature (°C)' in app_df.columns:
                temp_corr = app_df[['Energy Consumption (kWh)', 'Outdoor Temperature (°C)']].corr().iloc[0, 1]
            
            # Season usage
            season_usage = app_df.groupby('Season').size().to_dict()
            
            # Household size effect
            household_stats = app_df.groupby('Household Size')['Energy Consumption (kWh)'].agg(['mean', 'count']).to_dict()
            
            # Estimate average duration and power draw from data
            # We have energy consumption per event, need to estimate how long it ran
            # Approach: Look at typical usage patterns and energy values
            
            avg_energy = np.mean(energy_values)
            median_energy = np.median(energy_values)
            
            # For appliances that run continuously (like Fridge), estimate hourly power
            # For appliances that run in cycles, estimate per-use power
            
            if appliance == 'Fridge':
                # Fridge runs continuously, energy is per hour typically
                typical_power = median_energy  # Energy per event ~ power per hour
                estimated_duration = 24  # Runs all day
            elif appliance in ['Heater', 'Air Conditioning', 'Lights']:
                # These run for variable durations, estimate based on median
                typical_power = median_energy  # Rough estimate
                estimated_duration = 1.0  # Varies widely
            else:
                # For cycle appliances (dishwasher, washing machine, etc.)
                # Estimate power draw and duration
                # Common appliance cycles: 1-3 hours typically
                
                # Use the relationship: Energy (kWh) = Power (kW) × Duration (hours)
                # Assume typical durations based on appliance type
                if appliance in ['Washing Machine', 'Dishwasher']:
                    estimated_duration = 2.0  # Typical cycle time
                    typical_power = avg_energy / estimated_duration
                elif appliance in ['Oven', 'Microwave']:
                    estimated_duration = 0.5  # 30 min typical
                    typical_power = avg_energy / estimated_duration
                elif appliance in ['TV', 'Computer']:
                    estimated_duration = 3.0  # Typical usage session
                    typical_power = avg_energy / estimated_duration
                else:
                    # Generic estimate
                    estimated_duration = 1.0
                    typical_power = avg_energy
            
            # Build profile
            profile = {
                'appliance_type': appliance,
                'category': self.categorize_appliance(appliance),
                'can_schedule': appliance in self.EXPENDABLE_APPLIANCES,
                
                # Energy statistics
                'avg_energy_kwh': float(np.mean(energy_values)),
                'std_energy_kwh': float(np.std(energy_values)),
                'min_energy_kwh': float(np.min(energy_values)),
                'max_energy_kwh': float(np.max(energy_values)),
                'median_energy_kwh': float(np.median(energy_values)),
                
                # Power and duration
                'estimated_power_kw': float(typical_power),
                'estimated_duration_hours': float(estimated_duration),
                
                # Usage patterns
                'total_occurrences': int(total_occurrences),
                'peak_hours': peak_hours_list,
                'typical_usage_hours': typical_hours,
                'hourly_frequency': {int(k): float(v) for k, v in hourly_frequency.items()},
                
                # Environmental factors
                'temperature_correlation': float(temp_corr) if temp_corr is not None else None,
                'season_usage': {str(k): int(v) for k, v in season_usage.items()},
                
                # Scheduling constraints (for expendable appliances)
                'scheduling_constraints': self._get_scheduling_constraints(appliance, typical_hours)
            }
            
            self.profiles[appliance] = profile
        
        print(f"\nProfiled {len(self.profiles)} appliance types")
        return self.profiles
    
    def _get_scheduling_constraints(self, appliance, typical_hours):
        """Define scheduling constraints for appliances"""
        
        if appliance not in self.EXPENDABLE_APPLIANCES:
            return None
        
        # Default constraints
        constraints = {
            'min_start_hour': 1,
            'max_start_hour': 24,
            'max_delay_hours': 24,
            'preferred_windows': []
        }
        
        # Appliance-specific constraints
        if appliance == 'Dishwasher':
            # Usually after dinner, before bedtime
            constraints['preferred_windows'] = [(18, 23)]
            constraints['max_delay_hours'] = 8
            
        elif appliance == 'Washing Machine':
            # Morning or evening, avoid late night
            constraints['preferred_windows'] = [(6, 10), (17, 22)]
            constraints['max_delay_hours'] = 12
        
        return constraints
    
    def calculate_statistics(self):
        """Calculate overall statistics across all appliances"""
        
        self.statistics = {
            'total_records': len(self.df),
            'unique_appliances': len(self.profiles),
            'unique_homes': self.df['Home ID'].nunique(),
            'date_range': {
                'start': str(self.df['Date'].min()),
                'end': str(self.df['Date'].max())
            },
            'category_breakdown': {
                'ESSENTIAL': len([a for a, p in self.profiles.items() if p['category'] == 'ESSENTIAL']),
                'NECESSARY': len([a for a, p in self.profiles.items() if p['category'] == 'NECESSARY']),
                'EXPENDABLE': len([a for a, p in self.profiles.items() if p['category'] == 'EXPENDABLE']),
                'OTHER': len([a for a, p in self.profiles.items() if p['category'] == 'OTHER'])
            },
            'schedulable_appliances': [a for a, p in self.profiles.items() if p['can_schedule']],
            'total_energy_observed': float(self.df['Energy Consumption (kWh)'].sum()),
            'avg_energy_per_event': float(self.df['Energy Consumption (kWh)'].mean())
        }
        
        return self.statistics
    
    def plot_results(self, save_path='agent3_results.png'):
        """Generate publication-quality plots"""
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Energy consumption by appliance type
        ax1 = plt.subplot(2, 3, 1)
        energy_by_appliance = self.df.groupby('Appliance Type')['Energy Consumption (kWh)'].mean().sort_values(ascending=False)
        colors = ['#C73E1D' if app in self.EXPENDABLE_APPLIANCES else 
                 '#F18F01' if app in self.NECESSARY_APPLIANCES else 
                 '#2E86AB' for app in energy_by_appliance.index]
        energy_by_appliance.plot(kind='barh', ax=ax1, color=colors)
        ax1.set_xlabel('Average Energy per Use (kWh)', fontweight='bold')
        ax1.set_ylabel('Appliance Type', fontweight='bold')
        ax1.set_title('Average Energy Consumption by Appliance', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Usage frequency by hour (aggregate)
        ax2 = plt.subplot(2, 3, 2)
        hourly_all = self.df.groupby('Hour').size()
        ax2.plot(hourly_all.index, hourly_all.values, linewidth=2, marker='o', color='#A23B72')
        ax2.set_xlabel('Hour of Day', fontweight='bold')
        ax2.set_ylabel('Number of Appliance Uses', fontweight='bold')
        ax2.set_title('Overall Appliance Usage Pattern (24-Hour)', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 25, 4))
        
        # 3. Expendable appliances usage patterns
        ax3 = plt.subplot(2, 3, 3)
        for appliance in self.EXPENDABLE_APPLIANCES:
            if appliance in self.df['Appliance Type'].values:
                app_hourly = self.df[self.df['Appliance Type'] == appliance].groupby('Hour').size()
                ax3.plot(app_hourly.index, app_hourly.values, linewidth=2, marker='o', label=appliance)
        ax3.set_xlabel('Hour of Day', fontweight='bold')
        ax3.set_ylabel('Usage Frequency', fontweight='bold')
        ax3.set_title('Schedulable Appliances - Usage Patterns', fontweight='bold', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(0, 25, 4))
        
        # 4. Energy by category
        ax4 = plt.subplot(2, 3, 4)
        self.df['Category'] = self.df['Appliance Type'].apply(self.categorize_appliance)
        category_energy = self.df.groupby('Category')['Energy Consumption (kWh)'].sum()
        colors_cat = ['#2E86AB', '#F18F01', '#C73E1D', '#6A4C93']
        category_energy.plot(kind='pie', ax=ax4, autopct='%1.1f%%', colors=colors_cat, startangle=90)
        ax4.set_ylabel('')
        ax4.set_title('Total Energy by Category', fontweight='bold', fontsize=12)
        
        # 5. Temperature effect on climate appliances
        ax5 = plt.subplot(2, 3, 5)
        climate_apps = ['Heater', 'Air Conditioning']
        for app in climate_apps:
            if app in self.df['Appliance Type'].values:
                app_df = self.df[self.df['Appliance Type'] == app]
                temps = app_df['Outdoor Temperature (°C)'].values
                energy = app_df['Energy Consumption (kWh)'].values
                ax5.scatter(temps, energy, alpha=0.5, s=20, label=app)
        ax5.set_xlabel('Outdoor Temperature (°C)', fontweight='bold')
        ax5.set_ylabel('Energy Consumption (kWh)', fontweight='bold')
        ax5.set_title('Climate Appliances vs Temperature', fontweight='bold', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Appliance category distribution
        ax6 = plt.subplot(2, 3, 6)
        category_counts = self.df['Category'].value_counts()
        colors_bar = ['#2E86AB', '#F18F01', '#C73E1D', '#6A4C93']
        category_counts.plot(kind='bar', ax=ax6, color=colors_bar)
        ax6.set_xlabel('Category', fontweight='bold')
        ax6.set_ylabel('Number of Usage Events', fontweight='bold')
        ax6.set_title('Appliance Usage Events by Category', fontweight='bold', fontsize=12)
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Agent 3: Appliance Profiling - Statistical Analysis', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nPlot saved: {save_path}")
        plt.show()
    
    def generate_report(self, save_path='agent3_report.txt'):
        """Generate text report for paper"""
        
        report = f"""
{'='*70}
AGENT 3: APPLIANCE PROFILING - STATISTICAL ANALYSIS REPORT
{'='*70}

METHODOLOGY:
- Data Source: Device-level consumption records
- Analysis Type: Statistical aggregation and pattern identification
- No machine learning used - pure descriptive statistics

DATASET OVERVIEW:
- Total Records: {self.statistics['total_records']:,}
- Unique Appliances: {self.statistics['unique_appliances']}
- Unique Homes: {self.statistics['unique_homes']}
- Date Range: {self.statistics['date_range']['start']} to {self.statistics['date_range']['end']}
- Total Energy Observed: {self.statistics['total_energy_observed']:.2f} kWh

APPLIANCE CATEGORIES:
- ESSENTIAL (always running/critical): {self.statistics['category_breakdown']['ESSENTIAL']} types
- NECESSARY (regular use, limited flexibility): {self.statistics['category_breakdown']['NECESSARY']} types
- EXPENDABLE (schedulable for optimization): {self.statistics['category_breakdown']['EXPENDABLE']} types
- OTHER: {self.statistics['category_breakdown']['OTHER']} types

SCHEDULABLE APPLIANCES:
{chr(10).join(['  - ' + a for a in self.statistics['schedulable_appliances']])}

DETAILED APPLIANCE PROFILES:

"""
        
        # Add detailed profiles for each appliance
        for appliance, profile in sorted(self.profiles.items()):
            report += f"""
{'-'*70}
{appliance.upper()} ({profile['category']})
{'-'*70}
Energy Consumption:
  - Average: {profile['avg_energy_kwh']:.3f} kWh per use
  - Range: {profile['min_energy_kwh']:.3f} - {profile['max_energy_kwh']:.3f} kWh
  - Std Dev: {profile['std_energy_kwh']:.3f} kWh

Power & Duration:
  - Estimated Power: {profile['estimated_power_kw']:.2f} kW
  - Estimated Duration: {profile['estimated_duration_hours']:.2f} hours

Usage Patterns:
  - Total Occurrences: {profile['total_occurrences']}
  - Peak Usage Hours: {', '.join(map(str, profile['peak_hours']))}
  - Typical Hours: {', '.join(map(str, profile['typical_usage_hours']))}

"""
            if profile['can_schedule']:
                report += f"""Scheduling Information:
  - Can be scheduled: YES
  - Preferred Windows: {profile['scheduling_constraints']['preferred_windows']}
  - Max Delay: {profile['scheduling_constraints']['max_delay_hours']} hours

"""
            
            if profile['temperature_correlation'] is not None:
                report += f"""Environmental Factors:
  - Temperature Correlation: {profile['temperature_correlation']:.3f}

"""
        
        report += f"""
{'='*70}
INTERPRETATION:

The profiling analysis identifies {len(self.statistics['schedulable_appliances'])} appliances 
that can be scheduled for energy cost optimization. These appliances account for a 
significant portion of flexible household energy consumption.

Essential appliances (Fridge, Heater, AC, Lights) cannot be scheduled and represent
baseline household consumption that must be met regardless of electricity prices.

The temporal usage patterns reveal clear peak hours for different appliance types,
which can inform intelligent scheduling algorithms to shift expendable loads to
off-peak, lower-cost time periods.

APPLICATIONS:
- Provides appliance energy profiles for cost calculation
- Identifies schedulable vs. fixed loads
- Defines temporal constraints for optimization algorithms
- Enables household energy consumption estimation
- Supports personalized scheduling based on appliance ownership

{'='*70}
"""
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"Report saved: {save_path}")
    
    def get_profile(self, appliance_type):
        """Get profile for specific appliance"""
        return self.profiles.get(appliance_type, None)
    
    def get_schedulable_appliances(self):
        """Get list of appliances that can be scheduled"""
        return [app for app, profile in self.profiles.items() if profile['can_schedule']]
    
    def estimate_appliance_cost(self, appliance_type, price_per_kwh):
        """Estimate cost to run an appliance at given price"""
        profile = self.get_profile(appliance_type)
        if profile:
            return profile['avg_energy_kwh'] * price_per_kwh
        return None
    
    def save(self, prefix='agent3'):
        """Save profiles and statistics"""
        with open(f'{prefix}_profiles.json', 'w') as f:
            json.dump(self.profiles, f, indent=2)
        with open(f'{prefix}_statistics.json', 'w') as f:
            json.dump(self.statistics, f, indent=2)
        print(f"\nSaved: {prefix}_profiles.json, {prefix}_statistics.json")
    
    def load(self, prefix='agent3'):
        """Load profiles and statistics"""
        with open(f'{prefix}_profiles.json', 'r') as f:
            self.profiles = json.load(f)
        with open(f'{prefix}_statistics.json', 'r') as f:
            self.statistics = json.load(f)
        print(f"Loaded: {prefix}_profiles.json")


if __name__ == "__main__":
    # Initialize agent
    agent = ApplianceProfilingAgent()
    
    # Load data
    agent.load_data('../Data/device_consumption.csv')
    
    # Build profiles
    agent.build_profiles()
    
    # Calculate statistics
    agent.calculate_statistics()
    
    # Generate visualizations
    agent.plot_results('agent3_results.png')
    
    # Generate text report
    agent.generate_report('agent3_report.txt')
    
    # Example usage
    print("\n" + "="*60)
    print("EXAMPLE USAGE")
    print("="*60)
    
    # Get schedulable appliances
    schedulable = agent.get_schedulable_appliances()
    print(f"\nSchedulable appliances: {schedulable}")
    
    # Get profile for washing machine
    if 'Washing Machine' in agent.profiles:
        wm_profile = agent.get_profile('Washing Machine')
        print(f"\nWashing Machine Profile:")
        print(f"  Average energy: {wm_profile['avg_energy_kwh']:.3f} kWh")
        print(f"  Estimated duration: {wm_profile['estimated_duration_hours']:.2f} hours")
        print(f"  Peak hours: {wm_profile['peak_hours']}")
        
        # Estimate cost at different prices
        cheap_price = 0.050  # $0.05/kWh
        expensive_price = 0.080  # $0.08/kWh
        
        cheap_cost = agent.estimate_appliance_cost('Washing Machine', cheap_price)
        expensive_cost = agent.estimate_appliance_cost('Washing Machine', expensive_price)
        
        print(f"\n  Cost at ${cheap_price:.3f}/kWh: ${cheap_cost:.4f}")
        print(f"  Cost at ${expensive_price:.3f}/kWh: ${expensive_cost:.4f}")
        print(f"  Savings by scheduling: ${expensive_cost - cheap_cost:.4f} ({((expensive_cost - cheap_cost)/expensive_cost * 100):.1f}%)")
    
    # Save
    agent.save('agent3')