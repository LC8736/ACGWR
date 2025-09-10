##### ACGWR: Adaptive Coefficient Geographically Weighted Regression

This repository contains the implementation of ACGWR (Adaptive Coefficient Geographically Weighted Regression) and comparative analysis with other spatial regression models.

##### Project Structure

src/
├── data/
│ └── georgia/ # Georgia dataset
│ ├── GData\_utm.csv # Attribute data
│ └── G\_utm.shp # Shapefile
├── georgia\_plots/ # Output directory for Georgia visualizations
├── simulation\_results/ # Output directory for simulation results
├── acgwr\_model.py # ACGWR model implementation
├── data\_generation.py # Synthetic data generation functions
├── evaluation.py # Performance evaluation metrics
├── georgia\_analysis.py # Georgia dataset analysis
├── llgwr\_model.py # LL-GWR model implementation
├── main.py # Demonstration script
├── performance\_simulation.py # Performance simulation script
├── run\_experiment.py # Experiment runner
└── visualize\_results.py # Visualization functions

##### Quick Start

1. Run the Georgia analysis:
   python georgia\_analysis.py
2. Run performance simulations:
   python performance\_simulation.py --n\_repeat 100 --grid\_size 9 24
3. Run experiments:

   python run\_experiment.py --n\_repeat 100 --grid\_size 24

   ##### Data

   The Georgia dataset is located in src/data/georgia/. Ensure the data files are in this location before running the analysis.

   ##### Output

   Georgia analysis results: georgia\_model\_comparison.xlsx and georgia\_plots/

   Simulation results: simulation\_results/ directory

   Experiment results: simulation\_visualizations/ directory

   

   ##### Notes

1. For full reproduction of the paper results, use the same parameters:

   Number of repetitions: 500
   Grid sizes: 9 and 24 (sample sizes 100 and 625)

2. The complete process may take several hours to complete depending on hardware.
3. All codes are thoroughly commented to explain each step of the analysis.
4. Ensure all data files are in the correct location (src/data/georgia/) before running analyses.
