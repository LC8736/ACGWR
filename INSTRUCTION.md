Table 1: Comparison of prediction performance for four models

\## Description: Monte Carlo simulation results comparing ACGWR, GWR, MGWR, and LL-GWR models.



Reproduction Steps:

1.Run the performance simulation:

  python performance\_simulation.py --n\_repeat 500 --grid\_size 24

2.The results will be saved in simulation\_results\_comparison.xlsx

3.The Excel file contains R², RMSE, and MAE metrics for all four models





Table 2: Comparison of model fitting performance

\## Description: Georgia dataset analysis results comparing model performance.

Reproduction Steps:

1. Ensure Georgia data is in data/georgia/ directory
2. Run the Georgia analysis:
   python georgia\_analysis.py
3. The results will be saved in georgia\_model\_comparison.xlsx
4. The Excel file contains R², RMSE, and MAE metrics for all four models



Figure 1: Estimation of coefficient surfaces under separable structure

\## Description: 3D surface plots showing true and estimated coefficient surfaces for separable structure.

Reproduction Steps:

1. Run the separable structure simulation:
   python performance\_simulation.py --n\_repeat 500 --grid\_size 24
2. The figure will be automatically generated and saved as beta\_comparison\_separable.jpg in the simulation results directory

 

Figure 2: Estimation of coefficient surfaces under non-separable structure

Description: 3D surface plots showing true and estimated coefficient surfaces for non-separable structure.



Reproduction Steps:

1. The figure is automatically generated alongside Figure 1 during the simulation
2. Saved as beta\_comparison\_non\_separable.jpg in the simulation results directory



Figure 3: True and estimated curves of directional spatial effect functions

\## Description: Comparison of true and estimated f and g functions for separable structure.



Reproduction Steps:

1. The figure is automatically generated during the separable structure simulation
2. Saved as f\_g\_comparison\_separable.jpg in the simulation results directory



Figures 4-10: Directional functions and spatial distributions

\## Description: Seven figures showing directional functions and spatial distributions for each variable in the Georgia analysis.



Reproduction Steps:

1. Run the Georgia analysis:
   python georgia\_analysis.py
2. The figures will be saved in the georgia\_plots/ directory as:

* georgia\_acgwr\_feature\_0.jpg (Intercept term - Figure 4)
* georgia\_acgwr\_feature\_1.jpg (TotPop90 - Figure 5)
* georgia\_acgwr\_feature\_2.jpg (PctRural - Figure 6)
* georgia\_acgwr\_feature\_3.jpg (PctEld - Figure 7)
* georgia\_acgwr\_feature\_4.jpg (PctFB - Figure 8)
* georgia\_acgwr\_feature\_5.jpg (PctPov - Figure 9)
* georgia\_acgwr\_feature\_6.jpg (PctBlack - Figure 10)



\##  Additional Notes

1. For faster testing, reduce the number of Monte Carlo repetitions (e.g., --n\_repeat 10)
2. The Georgia analysis requires the shapefile and CSV data to be in the data/georgia/ directory
3. All visualizations use Times New Roman font for consistency with the paper
4. The simulation results include both separable and non-separable structures
5. The code automatically handles bandwidth selection for all models



\##  Expected Output Files

After running all analyses, you should have:

1. simulation\_results\_comparison.xlsx (Table 1)
2. georgia\_model\_comparison.xlsx (Table 2)
3. Multiple JPG files for Figures 1-10 in their respective directories
4. CSV files with coefficient estimates for further analysis
