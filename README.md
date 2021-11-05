# Analysis tool for REPLOM

This repository contains a pipeline for analysis of data obtained through REal-time kinetics via binding and Photobleaching LOcalization Microscopy (REPLOM). 
The methodology is described in https://doi.org/10.1101/2021.08.20.457097 and made available here for convenience.
The tool relies on clustering with a euclidean minimum spanning tree which is implemented in the [AstroML](https://www.astroml.org/index.html) package (see [here](https://www.astroml.org/book_figures/chapter6/fig_great_wall_MST.html) for an example use of the clustering methodology). It is therefore under the BSD license. 

# Introduction
The input data consists of STORM localizations in a csv file from blinking fluorophores with timestamps. 

    "frame","x [nm]","y [nm]","intensity [photon]"
    1.0,1031.048741153928,38711.99299011621,771.0507218978817
    1.0,1209.1570168683265,15066.011054392755,1469.2293350289435

The pipeline first use the method Hierarchical Clustering from the [AstroML](https://www.astroml.org/index.html) package to segment the observed data into molecular clusters

![Overview](https://github.com/hatzakislab/REPLOM-analysis-tool/blob/main/Readme_files/plot%20overview.png)

The individual clusters are then analyzed in parallel to estimate growth curves of their area, leading to an output looking like this
![Alt Text](https://github.com/hatzakislab/REPLOM-analysis-tool/blob/main/Readme_files/aggregate.gif)

Finally, the repository includes a fitting functionality which allows estimation of growth parameters for the segmented clusters.
# Example use
The steps below guide you though an example use of how to employ this tool for analysis of REPLOM data. 
## Installation
The scripts rely on a set of libraries. As of now, the individual libraries must be installed by the user using either pip or conda. 
For most this will include (along with subdependencies of these libraries of course): 
<ol>
    <li><code>matplotlib</code></li>
    <li><code>iminuit</code></li>
    <li><code>multiprocess</code></li>
    <li><code>sklearn</code></li>
    <li><code>tqdm</code></li>
</ol>

The animations are built using 'FuncAnimation' from 'matplotlib' which requires ffmpeg. 
If you do not have this on your system, you can run 

    conda install -c conda-forge ffmpeg

Or 

    pip install ffmpeg
    
To install it. 
## Segment clusters
The repository contains an example data set `example_raw data.csv`. 
After having installed the requirements, you may navigate to the cloned repository (if not already there) and run 

    python Automated_aggregate_analysis.py example_raw\ data.csv 0.92

This command calls the script `Automated_aggregate_analysis.py` with the file `example_raw\ data.csv` as input. 
The only input parameter is the distance cutoff percentile used to segment the euclidean minimum spanning tree (see [here](https://www.astroml.org/book_figures/chapter6/fig_great_wall_MST.html) for an example). 
A higher value of this cutoff yields less but larger clusters and conversely for a lower cutoff. 

After having run this command, a folder called `example_raw data` should have appeared, containing an overview plot called `plot overview.png` of the clusters along with coordinates in csv files named for example `Group 0.csv`.
The script will then ask if the clustering is reasonable. 
If you agree, you type `y` and press enter. 
If not, you type in a new cutoff and press enter after which it will recompute the clustering. 

Upon accepting the clusters computed, the script automatically begins analyzing the clusters and computing growth curves and videos. 
While this is run in parallel, the process still takes some time (can take up to hours, but movies should be produced continously).
The end result is a movie for each file called for example `Clustered Group 0.mp4` and a growth curve callled `Group 8 Growth curve` which may be fitted to extract growth parameters. 
## Fit growth curves
Upon completion of clustering, the individual growth curves may be fit. 
The script `fitter-rate calculation.py` contains functions to do so. 
In the current implementation of the repository, the script will run an example which relies on completion of cluster segmentation on the example data with a cutoff of 0.92. 
If done, running the command 

    python fitter-rate\ calculation.py
    
will fit two of the clusters, a symmetric and assymetric cluster with a single growth mode and a switching one respectively.

# Repository overview
To sum up, here is a description of each of the scripts in this repository (apart from stuff relating to readme):
<ol>
  <li>
    <code>Automated_aggregate_analysis.py</code> 
  </li>
    <ol>
      <li>
       Runs clustering analysis on an input csv file of REPLOM observations. 
       Upon completion, it computes growth curves and generates movies for all clusters.
      </li>
    </ol>
  </li>
  <li>
  <code>fitter-rate calculation.py</code>
</li>
    <ol>
        <li>Contains fit functions to fit individual clusters</li>
      </ol>
<li>
    <code>example_raw data.csv</code>
     <ol>
        <li>Example data set for instructional use</li>
      </ol>
</li>
<li>
    <code>mst_clustering.py</code>
     <ol>
        <li>Auxillary script to compute Euclidean Spanning tree clustering, modified from the [AstroML](https://www.astroml.org/index.html)</li>
      </ol>
</li>
</ol>

