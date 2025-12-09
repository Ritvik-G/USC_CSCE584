"""
Agent 3: Appliance Profiling
Statistical analysis of appliance energy usage patterns from device consumption data
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_palette("husl")


class ApplianceProfilingAgent:
    
    ESSENTIAL_APPLIANCES = {"Fridge", "Heater", "Air Conditioning", "Lights"}
    NECESSARY_APPLIANCES = {"Oven", "Microwave", "TV", "Computer"}
    EXPENDABLE_APPLIANCES = {"Washing Machine", "Dishwasher"}
    
    def __init__(self):
        self.df = None
        self.profiles = {}
        self.statistics = {}
        
    def load_data(self, filepath):
        """Load device consumption data"""
        self.df = pd.read_csv(filepath)
        
        # Parse time and extract hour
        self.df['Hour'] = pd.to_datetime(self.df['Time'], format='%H:%M', errors='coerce').dt.hour + 1
        
        # Parse date and extract day of week
        self.df['DateTime'] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d', errors='coerce')
        self.df['DayOfWeek'] = self.df['DateTime'].dt.dayofweek + 1
        
        return self.df
    
    def categorize_appliance(self, appliance_type):
        """Determine appliance category"""
        if appliance_type in self.ESSENTIAL_APPLIANCES:
            return "ESSENTIAL"
        elif appliance_type in self.NECESSARY_APPLIANCES:
            return "NECESSARY"
        elif appliance_type in self.EXPENDABLE_APPLIANCES:
            return "EXPENDABLE"
        else:
            return "OTHER"
    
    def build_profiles(self):
        """Build statistical profiles for each appliance"""
        appliance_types = self.df['Appliance Type'].unique()
        
        # Typical power draws (kW) for duration estimation
        typical_power_draws = {
            'Washing Machine': 1.5, 'Dishwasher': 1.8, 'Fridge': 0.15,
            'Oven': 2.5, 'Microwave': 1.2, 'Heater': 2.0,
            'Air Conditioning': 3.5, 'Lights': 0.1, 'TV': 0.15, 'Computer': 0.2
        }
        
        for appliance in appliance_types:
            app_df = self.df[self.df['Appliance Type'] == appliance]
            energy_values = app_df['Energy Consumption (kWh)'].values
            
            # Usage by hour
            hourly_usage = app_df.groupby('Hour').size().to_dict()
            total_occurrences = len(app_df)
            hourly_frequency = {h: hourly_usage.get(h, 0) / total_occurrences for h in range(1, 25)}
            
            # Peak hours (top 5)
            peak_hours = sorted(hourly_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
            peak_hours_list = [h for h, _ in peak_hours]
            
            # Typical usage windows (hours with >5% frequency)
            typical_hours = [h for h, freq in hourly_frequency.items() if freq > 0.05]
            
            # Temperature correlation
            temp_corr = None
            if 'Outdoor Temperature (°C)' in app_df.columns:
                temp_corr = app_df[['Energy Consumption (kWh)', 'Outdoor Temperature (°C)']].corr().iloc[0, 1]
            
            # Season usage
            season_usage = app_df.groupby('Season').size().to_dict()
            
            # Estimate duration and power
            avg_energy = np.mean(energy_values)
            median_energy = np.median(energy_values)
            
            if appliance == 'Fridge':
                typical_power = median_energy
                estimated_duration = 24
            elif appliance in ['Heater', 'Air Conditioning', 'Lights']:
                typical_power = median_energy
                estimated_duration = 1.0
            elif appliance in ['Washing Machine', 'Dishwasher']:
                estimated_duration = 2.0
                typical_power = avg_energy / estimated_duration
            elif appliance in ['Oven', 'Microwave']:
                estimated_duration = 0.5
                typical_power = avg_energy / estimated_duration
            elif appliance in ['TV', 'Computer']:
                estimated_duration = 3.0
                typical_power = avg_energy / estimated_duration
            else:
                estimated_duration = 1.0
                typical_power = avg_energy
            
            # Build profile
            self.profiles[appliance] = {
                'appliance_type': appliance,
                'category': self.categorize_appliance(appliance),
                'can_schedule': appliance in self.EXPENDABLE_APPLIANCES,
                'avg_energy_kwh': float(np.mean(energy_values)),
                'std_energy_kwh': float(np.std(energy_values)),
                'min_energy_kwh': float(np.min(energy_values)),
                'max_energy_kwh': float(np.max(energy_values)),
                'median_energy_kwh': float(np.median(energy_values)),
                'estimated_power_kw': float(typical_power),
                'estimated_duration_hours': float(estimated_duration),
                'total_occurrences': int(total_occurrences),
                'peak_hours': peak_hours_list,
                'typical_usage_hours': typical_hours,
                'hourly_frequency': {int(k): float(v) for k, v in hourly_frequency.items()},
                'temperature_correlation': float(temp_corr) if temp_corr is not None else None,
                'season_usage': {str(k): int(v) for k, v in season_usage.items()},
                'scheduling_constraints': self._get_scheduling_constraints(appliance, typical_hours)
            }
        
        return self.profiles
    
    def _get_scheduling_constraints(self, appliance, typical_hours):
        """Define scheduling constraints for expendable appliances"""
        if appliance not in self.EXPENDABLE_APPLIANCES:
            return None
        
        constraints = {
            'min_start_hour': 1,
            'max_start_hour': 24,
            'max_delay_hours': 24,
            'preferred_windows': []
        }
        
        if appliance == 'Dishwasher':
            constraints['preferred_windows'] = [(18, 23)]
            constraints['max_delay_hours'] = 8
        elif appliance == 'Washing Machine':
            constraints['preferred_windows'] = [(6, 10), (17, 22)]
            constraints['max_delay_hours'] = 12
        
        return constraints
    
    def calculate_statistics(self):
        """Calculate overall statistics"""
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
        """Generate visualization"""
        fig = plt.figure(figsize=(16, 10))
        
        # Energy by appliance
        ax1 = plt.subplot(2, 3, 1)
        energy_by_appliance = self.df.groupby('Appliance Type')['Energy Consumption (kWh)'].mean().sort_values(ascending=False)
        colors = ['#C73E1D' if app in self.EXPENDABLE_APPLIANCES else 
                 '#F18F01' if app in self.NECESSARY_APPLIANCES else 
                 '#2E86AB' for app in energy_by_appliance.index]
        energy_by_appliance.plot(kind='barh', ax=ax1, color=colors)
        ax1.set_xlabel('Avg Energy (kWh)', fontweight='bold')
        ax1.set_ylabel('Appliance', fontweight='bold')
        ax1.set_title('Energy by Appliance', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Overall usage pattern
        ax2 = plt.subplot(2, 3, 2)
        hourly_all = self.df.groupby('Hour').size()
        ax2.plot(hourly_all.index, hourly_all.values, linewidth=2, marker='o', color='#A23B72')
        ax2.set_xlabel('Hour', fontweight='bold')
        ax2.set_ylabel('Usage Count', fontweight='bold')
        ax2.set_title('24-Hour Usage Pattern', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 25, 4))
        
        # Expendable appliances patterns
        ax3 = plt.subplot(2, 3, 3)
        for appliance in self.EXPENDABLE_APPLIANCES:
            if appliance in self.df['Appliance Type'].values:
                app_hourly = self.df[self.df['Appliance Type'] == appliance].groupby('Hour').size()
                ax3.plot(app_hourly.index, app_hourly.values, linewidth=2, marker='o', label=appliance)
        ax3.set_xlabel('Hour', fontweight='bold')
        ax3.set_ylabel('Usage Frequency', fontweight='bold')
        ax3.set_title('Schedulable Appliances', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(0, 25, 4))
        
        # Energy by category
        ax4 = plt.subplot(2, 3, 4)
        self.df['Category'] = self.df['Appliance Type'].apply(self.categorize_appliance)
        category_energy = self.df.groupby('Category')['Energy Consumption (kWh)'].sum()
        colors_cat = ['#2E86AB', '#F18F01', '#C73E1D', '#6A4C93']
        category_energy.plot(kind='pie', ax=ax4, autopct='%1.1f%%', colors=colors_cat, startangle=90)
        ax4.set_ylabel('')
        ax4.set_title('Energy by Category', fontweight='bold')
        
        # Temperature effect
        ax5 = plt.subplot(2, 3, 5)
        climate_apps = ['Heater', 'Air Conditioning']
        for app in climate_apps:
            if app in self.df['Appliance Type'].values:
                app_df = self.df[self.df['Appliance Type'] == app]
                temps = app_df['Outdoor Temperature (°C)'].values
                energy = app_df['Energy Consumption (kWh)'].values
                ax5.scatter(temps, energy, alpha=0.5, s=20, label=app)
        ax5.set_xlabel('Temperature (°C)', fontweight='bold')
        ax5.set_ylabel('Energy (kWh)', fontweight='bold')
        ax5.set_title('Climate Appliances', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Usage by category
        ax6 = plt.subplot(2, 3, 6)
        category_counts = self.df['Category'].value_counts()
        colors_bar = ['#2E86AB', '#F18F01', '#C73E1D', '#6A4C93']
        category_counts.plot(kind='bar', ax=ax6, color=colors_bar)
        ax6.set_xlabel('Category', fontweight='bold')
        ax6.set_ylabel('Usage Events', fontweight='bold')
        ax6.set_title('Events by Category', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Agent 3: Appliance Profiling', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def generate_report(self, save_path='agent3_report.txt'):
        """Generate text report"""
        report = f"""
{'='*70}
AGENT 3: APPLIANCE PROFILING
{'='*70}

Total Records: {self.statistics['total_records']:,}
Unique Appliances: {self.statistics['unique_appliances']}
Schedulable: {len(self.statistics['schedulable_appliances'])}

SCHEDULABLE APPLIANCES:
{', '.join(self.statistics['schedulable_appliances'])}

APPLIANCE PROFILES:

"""
        
        for appliance, profile in sorted(self.profiles.items()):
            report += f"""
{appliance} ({profile['category']}):
  Avg Energy: {profile['avg_energy_kwh']:.3f} kWh
  Est Power: {profile['estimated_power_kw']:.2f} kW
  Est Duration: {profile['estimated_duration_hours']:.1f} hours
  Peak Hours: {', '.join(map(str, profile['peak_hours'][:3]))}
  Schedulable: {'Yes' if profile['can_schedule'] else 'No'}
"""
        
        report += f"\n{'='*70}\n"
        
        with open(save_path, 'w') as f:
            f.write(report)
        print(report)
    
    def get_profile(self, appliance_type):
        """Get profile for specific appliance"""
        return self.profiles.get(appliance_type, None)
    
    def get_schedulable_appliances(self):
        """Get list of schedulable appliances"""
        return [app for app, profile in self.profiles.items() if profile['can_schedule']]
    
    def estimate_appliance_cost(self, appliance_type, price_per_kwh):
        """Estimate cost to run appliance"""
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
    
    def load(self, prefix='agent3'):
        """Load profiles and statistics"""
        with open(f'{prefix}_profiles.json', 'r') as f:
            self.profiles = json.load(f)
        with open(f'{prefix}_statistics.json', 'r') as f:
            self.statistics = json.load(f)


if __name__ == "__main__":
    agent = ApplianceProfilingAgent()
    agent.load_data('../Data/device_consumption.csv')
    agent.build_profiles()
    agent.calculate_statistics()
    agent.plot_results('agent3_results.png')
    agent.generate_report('agent3_report.txt')
    agent.save('agent3')