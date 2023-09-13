set term png
set output '../output/simulation_plot.png'
set datafile separator ","
set ytics nomirror
set ylabel 'Position' font 'Arial, 12'
plot "../output/time_evolution.csv" every::1 using 1:2 with lines title 'Position' lw 2 lt 1 axis x1y1
set y2tics nomirror
set y2label 'Variance' font 'Arial, 12'
plot "../output/time_evolution.csv" every::1 using 1:3 with lines title 'Variance' lw 2 lt 2 axis x1y2
