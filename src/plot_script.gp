set term png
set output '../output/simulation_plot.png'
set datafile separator ","
plot "../output/time_evolution.csv" every::1 using 1:2 with lines title 'Position'
