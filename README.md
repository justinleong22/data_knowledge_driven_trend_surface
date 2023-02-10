# Data-knowledge-driven trend surface analysis

### What is this repository for?
This repo is for a data-knowledge-driven trend surface analysis, where the trend surface analysis is a common task in geostatistics and geological interface modeling. Data-knowledge-driven means that we combine many different data sources including borehole data and geophysical surveys and also geological knowledge such as the delineation from geologists. 

This repo has the trend surface analysis both on the explicit modeling and implicit level-sets modeling. We present three different test cases: 

1. Modeling a Greenland subglacial topography, with ice-penetrating radar measurement, using 2D explicit modeling. 
2. Modeling a magmatic intrusion, with discrete borehole measurements, using 2D implicit modeling. 
3. Modeling palaeovalley structures in Australia, with Airborne Electromagnetic survey using 3D implicit modeling. 


### How do I get set up?
Please download the .zip file or use 

'''
git clone https://github.com/lijingwang/data_knowledge_driven_trend_surface.git
'''

using command line. 

### How to use the code? 
There are three main folders in this repo: data, methods and notebook. 

- The data folder contains all the data for each case. 

- The methods folder have three implementations for data-knowledge-driven trend surface constructions using Metropolis Hastings. Based on your project, where you can choose 2D or 3D model, explicit or implicit. The definition of the density function for multiple datasets are also within these three .py files. 

- The notebook folder should be your best friend. You can learn how the Metropolis Hastings algorithm was used to generate stochastic geological interfaces while meeting the criteria of both data and geological knowledge. 


### Who do I talk to?
Lijing Wang, Stanford University, mollywang52 AT gmail DOT com

