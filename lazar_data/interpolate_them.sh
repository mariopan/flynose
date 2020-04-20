#! /bin/bash

./interpolate step_1.txt 0 3.5 0.01 step_1.dat
./interpolate step_2.txt 0 3.5 0.01 step_2.dat
./interpolate step_3.txt 0 3.5 0.01 step_3.dat
./interpolate ramp_1.txt 0 3.5 0.01 ramp_1.dat
./interpolate ramp_2.txt 0 3.5 0.01 ramp_2.dat
./interpolate ramp_3.txt 0 3.5 0.01 ramp_3.dat
./interpolate parabola_1.txt 0 3.5 0.01 parabola_1.dat
./interpolate parabola_2.txt 0 3.5 0.01 parabola_2.dat
./interpolate parabola_3.txt 0 3.5 0.01 parabola_3.dat
