# Analysis tool for REPLOM

This repository contains a pipeline for analysis of data obtained through REal-time kinetics via binding and Photobleaching LOcalization Microscopy (REPLOM). 
The methodology is described in https://doi.org/10.1101/2021.08.20.457097 and made available here for convenience.
The tool relies on clustering with a euclidean minimum spanning tree which is implemented in the [AstroML](https://www.astroml.org/index.html) package (see [here](https://www.astroml.org/book_figures/chapter6/fig_great_wall_MST.html) for an example use of the clustering methodology). 

# Explanation
The input data consists of STORM localizations in a csv file from blinking fluorophores with timestamps. 

    "frame","x [nm]","y [nm]","intensity [photon]"
    1.0,1031.048741153928,38711.99299011621,771.0507218978817
    1.0,1209.1570168683265,15066.011054392755,1469.2293350289435

If plotted it looks something like 


The repository consists of two key scripts:

<ol>
  <li>
    <code>Automated_aggregate_analysis.py</code> 
  </li>
    <ol>
      <li>
        
      </li>
    </ol>
  </li>
  <li>
  <code>fitter-rate calculation.py</code>
  
</li>
    <ol>
        <li>Explanation</li>
      </ol>
</ol>
