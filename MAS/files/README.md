# Multi-Agent Smart Appliance Scheduler - IEEE Paper

## Document Overview

This package contains the **Proposed Methods and Experimental Setup** section for an IEEE conference paper submission on the multi-agent smart appliance scheduling system.

## Files Included

1. **proposed_methods_experimental_setup.tex** - Main LaTeX source file (18 KB)
2. **proposed_methods_experimental_setup.pdf** - Compiled PDF document (184 KB)
3. **README.md** - This file

## Document Structure

The document contains the following sections:

### 1. Introduction
Brief overview of the smart home energy management problem

### 2. Proposed Multi-Agent Architecture
System overview and agent descriptions

### 3. Agent 1: Electricity Price Forecasting
- Problem formulation
- Feature engineering (cyclical time encoding)
- Neural network architecture (6→64→32→16→1)
- Training procedure with Adam optimizer
- Data normalization techniques

### 4. Agent 2: Household Power Forecasting
- Problem formulation
- Enhanced feature engineering with seasonal encoding
- Data aggregation from minute to hourly resolution
- Neural network architecture (8→64→32→16→1)

### 5. Agent 3: Appliance Profiling
- Appliance categorization (Essential/Necessary/Expendable)
- Statistical profile construction
- Energy statistics computation
- Hourly usage frequency analysis
- Peak hours identification
- Power and duration estimation

### 6. Agent 4: Scheduling Optimizer
- Problem formulation
- Linear Programming (LP) formulation with constraints
- Reinforcement Learning (RL) greedy heuristic algorithm
- Baseline schedule definition
- Performance metrics (cost reduction, savings %, optimality gap)

### 7. Experimental Setup
- Dataset descriptions (price, power, device consumption)
- Training configuration and hyperparameters
- Evaluation metrics (R², MAE, RMSE, MAPE)
- Scheduling optimization configuration
- Implementation details (software stack)
- Agent integration workflow
- Performance analysis methodology
- Validation strategy

### 8. Experimental Workflow
Complete algorithmic description of the multi-agent execution pipeline

### 9. Reproducibility
Guidelines for experiment reproduction

## Key Mathematical Formulations

The document includes detailed mathematical formulations for:

1. **Cyclical time encoding** using sine/cosine transformations
2. **Neural network forward pass** equations
3. **MSE loss function** for training
4. **Statistical measures** (mean, std, frequency)
5. **LP objective function**: Minimize ∑ₐ∑ₜ xₐ,ₜ · E(a) · pₜ
6. **Power constraint**: Pₜ + ∑ running appliance power ≤ Pₘₐₓ
7. **Performance metrics**: R², MAE, RMSE, MAPE, savings percentage

## Algorithms Included

### Algorithm 1: RL Greedy Scheduling
Complete pseudocode for the reinforcement learning-based greedy heuristic that schedules appliances at lowest-cost hours while respecting constraints.

### Algorithm 2: Multi-Agent System Execution
Four-phase pipeline covering training, integration, optimization, and analysis.

## Compilation Instructions

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get install texlive-publishers texlive-latex-extra

# Or use TeX Live full installation
sudo apt-get install texlive-full
```

### Compiling the Document
```bash
# Single compilation
pdflatex proposed_methods_experimental_setup.tex

# For proper references (run twice)
pdflatex proposed_methods_experimental_setup.tex
pdflatex proposed_methods_experimental_setup.tex
```

### Using latexmk (recommended)
```bash
latexmk -pdf proposed_methods_experimental_setup.tex
```

## LaTeX Packages Used

- **IEEEtran** - IEEE conference format
- **amsmath, amssymb, amsfonts** - Mathematical symbols and equations
- **algorithm, algorithmic** - Algorithm environments
- **graphicx** - Figure support
- **booktabs** - Professional tables
- **multirow** - Multi-row table cells
- **xcolor, textcomp** - Colors and special characters

## Customization

### Adjusting Content

To modify the document:

1. **Author information** (line 15-18): Update department, university, contact
2. **Title** (line 13): Modify paper title
3. **Abstract** (line 22-27): Update abstract text
4. **Sections**: Add/remove/modify sections as needed
5. **References** (line 410+): Add actual bibliography entries

### Adding Results

To include experimental results:

1. Add results tables in Section 7 (Experimental Setup)
2. Reference result figures (currently placeholders)
3. Add statistical analysis in relevant sections

### Formatting Options

```latex
% Change to journal format
\documentclass[journal]{IEEEtran}

% Add line numbers for review
\usepackage{lineno}
\linenumbers

% Adjust margins (if allowed by conference)
\usepackage[margin=0.75in]{geometry}
```

## Integration with Main Paper

This document is designed to be:
1. **Standalone** - Can be submitted as a supplementary methods document
2. **Integrated** - Can be merged into the main paper
3. **Referenced** - Main paper can reference equation numbers from this document

## Document Statistics

- **Pages**: 9-10 (depending on compilation)
- **Equations**: 40+ numbered equations
- **Algorithms**: 2 detailed algorithms
- **Tables**: 1 hyperparameter table (expandable)
- **Sections**: 9 major sections

## Quality Checks

✓ All equations properly numbered and referenced  
✓ Consistent mathematical notation throughout  
✓ Algorithm pseudocode follows IEEE standards  
✓ Section numbering follows IEEE format  
✓ Citations format ready (requires actual references)  
✓ Table formatting uses booktabs for professional appearance  
✓ Proper use of IEEE formatting commands  

## Correspondence with Code

The mathematical formulations directly correspond to the implementation:

| Document Section | Code File | Key Functions |
|-----------------|-----------|---------------|
| Agent 1 (§3) | agent1_price_forecasting.py | create_features(), train() |
| Agent 2 (§4) | agent2_power_forecasting.py | create_features(), train() |
| Agent 3 (§5) | agent3_appliance_profiling.py | build_profiles() |
| Agent 4 (§6) | agent4_scheduler.py | optimize_lp(), optimize_rl() |
| Integration (§8) | integration_test.py | main() workflow |

## Next Steps

1. **Add Results**: Include experimental results tables and figures
2. **Complete References**: Add proper bibliography entries
3. **Add Figures**: Include system architecture diagram (referenced as Fig. 1)
4. **Expand Discussion**: Add results interpretation and analysis
5. **Peer Review**: Have colleagues review technical accuracy

## Contact

For questions about the mathematical formulations or experimental setup, please contact the corresponding author.

## License

This document is prepared for IEEE conference submission. Copyright will be transferred to IEEE upon acceptance.

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Compiled with**: pdfLaTeX, TeX Live 2023
